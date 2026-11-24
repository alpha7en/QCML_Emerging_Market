"""Three-qubit data-reuploading statevector variational quantum circuits (Sec. 2.1, Sec. 3.2, Fig. 1c).

Each circuit maps x in R^p to |psi(x)> = U(x; theta)|0>^{otimes n} on H = (C^2)^{otimes n}
(n = 3, d = 2^n = 8) under two encoding schedules:
  - Cyclic packed (L = 3): partitions the p scaled inputs into ceil(p / (2n)) sub-blocks of
    R_Y(s_y x_y) R_Z(s_z x_z) rotations followed by Euler R_Z R_Y R_Z gates on each wire;
  - Affine (L = 9): applies affine linear projections w_y x + b_y, w_z x + b_z per wire and layer.
Toggling the directed ring of CNOTs against the identity isolates multipartite entanglement,
quantified via the Meyer-Wallach global measure Q_{MW}(psi) in [0, 1], under both full-Hermitian
readouts <psi(x)|B|psi(x)> + b_0 (B in Herm(C^{2^n})) and local Pauli-{Z, X} readouts.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _ring_cnot_edges(n_qubits: int) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if n_qubits == 2:
        return [(0, 1)]
    return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]


def _ring_cnot_inv_perm(n_qubits: int) -> torch.Tensor:
    """Inverse basis permutation sigma^{-1} on {0, ..., 2^n - 1} induced by the directed CNOT ring
    CNOT_{0->1} ... CNOT_{n-1->0}, so that (U_{ring} psi)_k = psi_{sigma^{-1}(k)}.
    """
    d = 1 << n_qubits
    sigma = list(range(d))
    for ctrl, tgt in _ring_cnot_edges(n_qubits):
        c_bit = 1 << (n_qubits - 1 - ctrl)
        t_bit = 1 << (n_qubits - 1 - tgt)
        sigma = [s ^ t_bit if (s & c_bit) else s for s in sigma]
    inv_sigma = [0] * d
    for j, k in enumerate(sigma):
        inv_sigma[k] = j
    return torch.tensor(inv_sigma, dtype=torch.long)


def _pauli_z_signs(n_qubits: int) -> torch.Tensor:
    """Eigenvalues (-1)^{z_q} of local Pauli Z_q on computational basis states |z>."""
    d = 1 << n_qubits
    return torch.tensor(
        [[1.0 if ((z >> (n_qubits - 1 - q)) & 1) == 0 else -1.0 for q in range(n_qubits)] for z in range(d)],
        dtype=torch.float32,
    )


def _pauli_x_flips(n_qubits: int) -> torch.Tensor:
    """Bit-flip involutions z -> z xor 2^{n-1-q} representing local Pauli X_q on |z>."""
    d = 1 << n_qubits
    return torch.tensor(
        [[z ^ (1 << (n_qubits - 1 - q)) for z in range(d)] for q in range(n_qubits)],
        dtype=torch.long,
    )


def _apply_rz_all(psi: torch.Tensor, angles: torch.Tensor, z_signs: torch.Tensor) -> torch.Tensor:
    """Tensor product bigotimes_{q=0}^{n-1} exp(-i (theta_q / 2) Z_q) diagonal in the computational basis."""
    z_s = z_signs.to(dtype=angles.dtype, device=angles.device)
    phases = -0.5 * (angles @ z_s.T)
    return psi * torch.complex(torch.cos(phases), torch.sin(phases))


def _apply_ry_all(psi: torch.Tensor, angles: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Tensor product bigotimes_{q=0}^{n-1} exp(-i (theta_q / 2) Y_q) acting wire-by-wire."""
    bs, d = psi.shape
    half = 0.5 * angles
    cy, sy = torch.cos(half), torch.sin(half)
    batched = angles.ndim == 2
    psi_v = psi.view([bs] + [2] * n_qubits)
    for q in range(n_qubits):
        dim = q + 1
        if batched:
            cq = cy[:, q].view([bs] + [1] * (n_qubits - 1))
            sq = sy[:, q].view([bs] + [1] * (n_qubits - 1))
        else:
            cq, sq = cy[q], sy[q]
        p0, p1 = psi_v.unbind(dim=dim)
        psi_v = torch.stack((cq * p0 - sq * p1, sq * p0 + cq * p1), dim=dim)
    return psi_v.reshape(bs, d)


def meyer_wallach_entanglement(psi: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Meyer-Wallach global multipartite entanglement Q_{MW}(psi) = 2 (1 - (1/n) sum_{q=1}^n Tr(rho_q^2))
    in [0, 1], where rho_q = Tr_{!= q}(|psi><psi|) is the single-qubit reduced state (Sec. 3.2).
    """
    bs, d = psi.shape
    psi_v = psi.view([bs] + [2] * n_qubits)
    purity_sum = torch.zeros(bs, dtype=psi.real.dtype, device=psi.device)
    for q in range(n_qubits):
        perm = [0, q + 1] + [i + 1 for i in range(n_qubits) if i != q]
        mat = psi_v.permute(perm).reshape(bs, 2, d // 2)
        rho = torch.bmm(mat, mat.conj().transpose(1, 2))
        purity = (rho.real * rho.real + rho.imag * rho.imag).sum(dim=(1, 2))
        purity_sum = purity_sum + purity
    return torch.clamp(2.0 * (1.0 - purity_sum / float(n_qubits)), min=0.0, max=1.0)


class StatevectorVQC(nn.Module):
    """Noiseless n-qubit data-reuploading statevector circuit (Sec. 3.2, Fig. 1b)."""

    VALID_ENCODINGS = ("affine", "cyclic_packed")
    VALID_READOUTS = ("full_hermitian", "local_zx")

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 3,
        n_layers: int = 9,
        entangle: bool = True,
        encoding: str = "affine",
        readout: str = "full_hermitian",
        seed: int = 101,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if encoding == "cyclic":
            encoding = "cyclic_packed"
        if encoding not in self.VALID_ENCODINGS:
            raise ValueError(f"encoding must be in {self.VALID_ENCODINGS}, got {encoding!r}")
        if readout not in self.VALID_READOUTS:
            raise ValueError(f"readout must be in {self.VALID_READOUTS}, got {readout!r}")
        if n_qubits < 2 or n_qubits > 6:
            raise ValueError(f"n_qubits must be in [2, 6], got {n_qubits}")

        self.in_features = int(in_features)
        self.n_qubits = int(n_qubits)
        self.d = 1 << self.n_qubits
        self.n_layers = int(n_layers)
        self.entangle = bool(entangle)
        self.encoding = encoding
        self.readout_mode = readout
        self.real_dtype = dtype
        self.n_cnot_per_step = len(_ring_cnot_edges(self.n_qubits)) if self.entangle else 0

        self.register_buffer("ring_cnot_inv_perm", _ring_cnot_inv_perm(self.n_qubits))
        self.register_buffer("z_signs", _pauli_z_signs(self.n_qubits))
        self.register_buffer("x_flips", _pauli_x_flips(self.n_qubits))
        tri_r, tri_c = torch.triu_indices(self.d, self.d, offset=1)
        self.register_buffer("_tri_r", tri_r, persistent=False)
        self.register_buffer("_tri_c", tri_c, persistent=False)
        self._n_offdiag = int(tri_r.numel())

        gen = torch.Generator().manual_seed(int(seed))

        if self.encoding == "cyclic_packed":
            self.feats_per_subblock = 2 * self.n_qubits
            self.n_subblocks = int(math.ceil(self.in_features / float(self.feats_per_subblock)))
            self.n_total_steps = self.n_layers * self.n_subblocks
            self.input_scales = nn.Parameter(
                (0.22 + 0.05 * torch.randn(self.n_layers, self.in_features, generator=gen, dtype=dtype))
            )
            self.var_euler = nn.Parameter(
                0.10 * torch.randn(self.n_layers, self.n_subblocks, self.n_qubits, 3, generator=gen, dtype=dtype)
            )
        else:
            self.feats_per_subblock = self.in_features
            self.n_subblocks = 1
            self.n_total_steps = self.n_layers
            scale_std = 0.08 / math.sqrt(float(self.in_features))
            self.w_y = nn.Parameter(
                scale_std * torch.randn(self.n_layers, self.n_qubits, self.in_features, generator=gen, dtype=dtype)
            )
            self.w_z = nn.Parameter(
                scale_std * torch.randn(self.n_layers, self.n_qubits, self.in_features, generator=gen, dtype=dtype)
            )
            self.var_euler = nn.Parameter(
                0.08 * torch.randn(self.n_layers, self.n_qubits, 3, generator=gen, dtype=dtype)
            )

        if self.readout_mode == "full_hermitian":
            # Full observable B in Herm(C^{2^n}) plus scalar offset b_0 (2^{2n} + 1 real parameters).
            s_b = 0.10 / math.sqrt(float(self.d))
            self.b_diag = nn.Parameter(s_b * torch.randn((self.d,), generator=gen, dtype=dtype))
            self.b_off_re = nn.Parameter(s_b * torch.randn((self._n_offdiag,), generator=gen, dtype=dtype))
            self.b_off_im = nn.Parameter(s_b * torch.randn((self._n_offdiag,), generator=gen, dtype=dtype))
            self.b_bias = nn.Parameter(torch.zeros((1,), dtype=dtype))
            self.readout_linear = None
        else:
            self.register_parameter("b_diag", None)
            self.register_parameter("b_off_re", None)
            self.register_parameter("b_off_im", None)
            self.register_parameter("b_bias", None)
            self.readout_linear = nn.Linear(2 * self.n_qubits, 1, dtype=dtype)
            with torch.no_grad():
                self.readout_linear.weight.copy_(
                    (0.10 / math.sqrt(2.0 * self.n_qubits))
                    * torch.randn(1, 2 * self.n_qubits, generator=gen, dtype=dtype)
                )
                self.readout_linear.bias.zero_()

    def circuit_parameters(self) -> list[nn.Parameter]:
        readout_ids = {id(p) for p in self.readout_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in readout_ids]

    def readout_parameters(self) -> list[nn.Parameter]:
        if self.readout_mode == "full_hermitian":
            return [self.b_diag, self.b_off_re, self.b_off_im, self.b_bias]
        return list(self.readout_linear.parameters())

    def get_readout_observable(self) -> torch.Tensor | None:
        if self.readout_mode != "full_hermitian":
            return None
        re = torch.diag(self.b_diag)
        r, c = self._tri_r, self._tri_c
        re[r, c] = self.b_off_re
        re[c, r] = self.b_off_re
        im = torch.zeros((self.d, self.d), dtype=self.b_diag.dtype, device=self.b_diag.device)
        im[r, c] = self.b_off_im
        im[c, r] = -self.b_off_im
        return torch.complex(re, im)

    def circuit_spec(self) -> dict[str, int]:
        n_2q = self.n_total_steps * self.n_cnot_per_step
        cnot_moments = self.n_cnot_per_step
        if self.encoding == "cyclic_packed":
            depth = self.n_total_steps * (5 + cnot_moments)
        else:
            depth = self.n_total_steps * (3 + cnot_moments)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "n_qubits": int(self.n_qubits),
            "depth": int(depth),
            "n_2q_gates": int(n_2q),
            "n_params": int(n_params),
        }

    def evolve_state(self, x: torch.Tensor, init_psi: torch.Tensor | None = None) -> torch.Tensor:
        x_r = x.to(dtype=self.real_dtype)
        bs = x_r.shape[0]
        c_dtype = torch.complex128 if x_r.dtype == torch.float64 else torch.complex64

        if init_psi is None:
            psi = torch.zeros(bs, self.d, dtype=c_dtype, device=x_r.device)
            psi[:, 0] = 1.0
        else:
            psi = init_psi.to(dtype=c_dtype, device=x_r.device)

        if self.encoding == "cyclic_packed":
            for l_idx in range(self.n_layers):
                scaled_x = x_r * self.input_scales[l_idx]
                for b_idx in range(self.n_subblocks):
                    start = b_idx * self.feats_per_subblock
                    end = min(start + self.feats_per_subblock, self.in_features)
                    chunk = scaled_x[:, start:end]
                    if end - start < self.feats_per_subblock:
                        pad = torch.zeros(
                            bs, self.feats_per_subblock - (end - start), dtype=x_r.dtype, device=x_r.device
                        )
                        chunk = torch.cat([chunk, pad], dim=1)
                    ang_y = chunk[:, : self.n_qubits]
                    ang_z = chunk[:, self.n_qubits : 2 * self.n_qubits]

                    # R_Y precedes R_Z so the initial computational state |0>^{otimes n} acquires polar angle
                    # before relative Z-phase rotation, followed by universal SU(2) Euler angles R_Z R_Y R_Z.
                    psi = _apply_ry_all(psi, ang_y, self.n_qubits)
                    psi = _apply_rz_all(psi, ang_z, self.z_signs)
                    eul = self.var_euler[l_idx, b_idx]
                    psi = _apply_rz_all(psi, eul[:, 0], self.z_signs)
                    psi = _apply_ry_all(psi, eul[:, 1], self.n_qubits)
                    psi = _apply_rz_all(psi, eul[:, 2], self.z_signs)
                    if self.entangle:
                        psi = psi[:, self.ring_cnot_inv_perm]
        else:
            for l_idx in range(self.n_layers):
                eul = self.var_euler[l_idx]
                ang_y = x_r @ self.w_y[l_idx].T + eul[:, 1]
                ang_z = x_r @ self.w_z[l_idx].T + eul[:, 2]
                psi = _apply_rz_all(psi, eul[:, 0], self.z_signs)
                psi = _apply_ry_all(psi, ang_y, self.n_qubits)
                psi = _apply_rz_all(psi, ang_z, self.z_signs)
                if self.entangle:
                    psi = psi[:, self.ring_cnot_inv_perm]

        return psi

    def readout_from_state(self, psi: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "full_hermitian":
            B_mat = self.get_readout_observable()
            exp_val = torch.einsum("ba,ac,bc->b", psi.conj(), B_mat.to(dtype=psi.dtype), psi).real
            return (exp_val.to(dtype=self.real_dtype) + self.b_bias).unsqueeze(-1)
        probs = psi.real * psi.real + psi.imag * psi.imag
        exp_z = probs @ self.z_signs.to(dtype=probs.dtype, device=probs.device)
        psi_flipped = psi[:, self.x_flips]
        exp_x = (psi.conj().unsqueeze(1) * psi_flipped).real.sum(dim=2)
        obs = torch.cat([exp_z, exp_x], dim=1).to(dtype=self.real_dtype)
        return self.readout_linear(obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout_from_state(self.evolve_state(x))

    def readout_l2_penalty(self) -> torch.Tensor:
        """Dimension-normalized Frobenius penalty (1/2^n)||B||_F^2 (or 1/(2n)||w||_2^2 for local Pauli readout)."""
        if self.readout_mode == "full_hermitian":
            frob_sq = (
                torch.sum(self.b_diag ** 2)
                + 2.0 * torch.sum(self.b_off_re ** 2)
                + 2.0 * torch.sum(self.b_off_im ** 2)
            )
            return frob_sq / float(self.d)
        return torch.sum(self.readout_linear.weight ** 2) / float(2 * self.n_qubits)


def _train_single_vqc(
    X_tr: np.ndarray,
    y_tr_s: np.ndarray,
    X_va: np.ndarray | None,
    y_va_s: np.ndarray | None,
    *,
    encoding: str,
    n_qubits: int,
    n_layers: int,
    entangle: bool,
    readout: str,
    wd_readout: float,
    wd_circuit: float,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[StatevectorVQC, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = StatevectorVQC(
        in_features=X_tr.shape[1],
        n_qubits=n_qubits,
        n_layers=n_layers,
        entangle=entangle,
        encoding=encoding,
        readout=readout,
        seed=seed,
        dtype=torch.float32,
    )
    opt = torch.optim.AdamW(
        [
            {"params": model.circuit_parameters(), "lr": lr, "weight_decay": wd_circuit},
            {"params": model.readout_parameters(), "lr": lr * 1.5, "weight_decay": 0.0},
        ]
    )
    X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.asarray(y_tr_s, dtype=np.float32)).reshape(-1, 1)
    has_val = X_va is not None and y_va_s is not None and len(X_va) > 0
    X_va_t = None if not has_val else torch.from_numpy(np.asarray(X_va, dtype=np.float32))
    y_va_t = None if not has_val else torch.from_numpy(np.asarray(y_va_s, dtype=np.float32)).reshape(-1, 1)

    n_tr = len(X_tr_t)
    gen = torch.Generator().manual_seed(int(seed))
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=gen)
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(X_tr_t[idx])
            loss = torch.mean((pred - y_tr_t[idx]) ** 2) + float(wd_readout) * model.readout_l2_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_mse = float(torch.mean((model(X_va_t) - y_va_t) ** 2).item())
            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if has_val:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def fit_predict_vqc(
    encoding: str,
    n_qubits: int,
    n_layers: int,
    entangle: bool,
    readout: str,
    fold_dict: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Inner-validation selection of readout regularization and evaluation of out-of-sample
    forecasts, Meyer-Wallach entanglement Q_{MW}, and state infidelity relative to the training
    mean density matrix rho_bar_{tr} (Sec. 3.2, 4.1).
    """
    X_tr = np.asarray(fold_dict["X_tr"], dtype=np.float32)
    y_tr = np.asarray(fold_dict["y_tr"], dtype=np.float64).ravel()
    X_va = None if fold_dict.get("X_va") is None else np.asarray(fold_dict["X_va"], dtype=np.float32)
    y_va = None if fold_dict.get("y_va") is None else np.asarray(fold_dict["y_va"], dtype=np.float64).ravel()
    X_te = np.asarray(fold_dict["X_te"], dtype=np.float32)

    y_mu = float(y_tr.mean())
    y_sd = float(y_tr.std()) if float(y_tr.std()) >= 1e-8 else 1.0
    y_tr_s = (y_tr - y_mu) / y_sd
    y_va_s = None if y_va is None else (y_va - y_mu) / y_sd

    wd_grid = [0.05, 0.30]
    hpo_epochs = 10
    full_epochs = 24
    patience = 5
    lr = 2.5e-3 if encoding == "affine" else 4e-3
    wd_circuit = 1e-3

    if X_va is None or len(X_va) == 0:
        best_wd = wd_grid[0]
        model, best_val = _train_single_vqc(
            X_tr,
            y_tr_s,
            X_va,
            y_va_s,
            encoding=encoding,
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangle=entangle,
            readout=readout,
            wd_readout=best_wd,
            wd_circuit=wd_circuit,
            lr=lr,
            batch_size=512,
            max_epochs=full_epochs,
            patience=patience,
            seed=seed,
        )
    else:
        best_hpo = float("inf")
        best_wd = wd_grid[0]
        for wd in wd_grid:
            _, v_mse = _train_single_vqc(
                X_tr,
                y_tr_s,
                X_va,
                y_va_s,
                encoding=encoding,
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangle=entangle,
                readout=readout,
                wd_readout=wd,
                wd_circuit=wd_circuit,
                lr=lr,
                batch_size=512,
                max_epochs=hpo_epochs,
                patience=3,
                seed=seed,
            )
            if v_mse < best_hpo:
                best_hpo = v_mse
                best_wd = wd
        model, best_val = _train_single_vqc(
            X_tr,
            y_tr_s,
            X_va,
            y_va_s,
            encoding=encoding,
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangle=entangle,
            readout=readout,
            wd_readout=best_wd,
            wd_circuit=wd_circuit,
            lr=lr,
            batch_size=512,
            max_epochs=full_epochs,
            patience=patience,
            seed=seed,
        )

    @torch.no_grad()
    def _eval_batched(X_arr: np.ndarray, bs: int = 2048) -> tuple[np.ndarray, torch.Tensor]:
        model.eval()
        preds_l: list[np.ndarray] = []
        psis_l: list[torch.Tensor] = []
        for i0 in range(0, len(X_arr), bs):
            xb = torch.from_numpy(np.asarray(X_arr[i0 : i0 + bs], dtype=np.float32))
            psi_b = model.evolve_state(xb)
            p_b = model.readout_from_state(psi_b).squeeze(-1).cpu().numpy() * y_sd + y_mu
            preds_l.append(p_b.astype(np.float64))
            psis_l.append(psi_b.to(dtype=torch.complex128).cpu())
        return np.concatenate(preds_l, axis=0), torch.cat(psis_l, dim=0)

    pred_te, psi_te = _eval_batched(X_te)
    pred_va = np.empty((0,), dtype=np.float64) if (X_va is None or len(X_va) == 0) else _eval_batched(X_va)[0]
    q_mw_mean = float(meyer_wallach_entanglement(psi_te, n_qubits).mean().item())

    # Principal eigenstate |psi_ref> of the empirical training state rho_bar_{tr} = N_{tr}^{-1} sum_s |psi(x_s)><psi(x_s)|.
    _, psi_tr = _eval_batched(X_tr)
    rho_bar_tr = (psi_tr.T @ psi_tr.conj()) / float(max(1, len(psi_tr)))
    rho_bar_tr = 0.5 * (rho_bar_tr + rho_bar_tr.mH)
    _, vecs_ref = torch.linalg.eigh(rho_bar_tr)
    psi_ref = vecs_ref[:, -1]

    @torch.no_grad()
    def ood_fn(X_ood: np.ndarray) -> dict[str, np.ndarray]:
        pred_o, psi_o = _eval_batched(np.asarray(X_ood, dtype=np.float32))
        ov = torch.abs(torch.einsum("ba,a->b", psi_o.conj(), psi_ref))
        infid_ref = torch.clamp(1.0 - ov * ov, min=0.0, max=1.0).cpu().numpy()
        fid_rho = torch.einsum("ba,ac,bc->b", psi_o.conj(), rho_bar_tr, psi_o).real
        infid_rho = torch.clamp(1.0 - fid_rho, min=0.0, max=1.0).cpu().numpy()
        return {
            "pred": pred_o,
            "infidelity": infid_ref.astype(np.float64),
            "rho_infidelity": infid_rho.astype(np.float64),
        }

    spec = model.circuit_spec()
    return {
        "pred_te": pred_te,
        "pred_va": pred_va,
        "n_qubits": int(spec["n_qubits"]),
        "depth": int(spec["depth"]),
        "n_2q_gates": int(spec["n_2q_gates"]),
        "n_params": int(spec["n_params"]),
        "q_mw_mean": q_mw_mean,
        "best_val_loss": float(best_val),
        "best_wd_readout": float(best_wd),
        "ood_fn": ood_fn,
    }
