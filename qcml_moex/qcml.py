"""Matrix-form Quantum Cognition Machine Learning, operator-swap limits, and spectral geometry
(Sec. 2 and Appendix A of the paper).

For an input x in R^p and feature operators A_1, ..., A_p in Herm(C^d), the positive semidefinite
error Hamiltonian decomposes into a shifted operator and a scalar radial shift [Eq. (1)]:
    H(x) = (1/2) sum_{k=1}^p (A_k - x_k I_d)^2 = H_tilde(x) + (1/2) ||x||_2^2 I_d,
    H_tilde(x) = C_0(A) - sum_{k=1}^p x_k A_k,    C_0(A) = (1/2) sum_{k=1}^p A_k^2.
Observable expectations y_hat_m(x) = Tr(rho_beta(x) B_m) are evaluated either in the projective
ground state rho_infty(x) = |psi_0(x)><psi_0(x)| via a Tikhonov-damped Rayleigh-Schrodinger
resolvent [Eqs. (A.4)-(A.8)] or in the finite-temperature Gibbs state rho_beta(x) via the
Daleckii-Krein divided-difference kernel [Eq. (2), Sec. A.2]. Restricting A_k, B_m to Sym(R^d)
sets Berry curvature F_{jk}(x) = 0; restricting to diag(R^d) sets [A_j, A_k] = 0 and reduces
Eq. (2) to a radial-basis Boltzmann mixture over d centroids in R^p.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class _TikhonovGroundState(torch.autograd.Function):
    """Projective ground state rho_0(x) = |psi_0><psi_0| with Tikhonov-damped resolvent adjoint
    R_0^eps = sum_{a >= 1} [Delta_a / (Delta_a^2 + eps_gap^2)] |psi_a><psi_a| [Eqs. (A.4), (A.7)].
    """

    @staticmethod
    def forward(
        ctx: Any, H_tilde: torch.Tensor, eps_gap: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evals, evecs = torch.linalg.eigh(H_tilde)
        e0 = evals[:, 0]
        delta_e = evals[:, 1] - evals[:, 0]
        psi0 = evecs[:, :, 0]
        rho0 = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
        ctx.save_for_backward(evals, evecs)
        ctx.eps_gap = float(eps_gap)
        return e0, delta_e, rho0

    @staticmethod
    def backward(
        ctx: Any,
        grad_e0: torch.Tensor | None,
        grad_delta_e: torch.Tensor | None,
        grad_rho0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        evals, evecs = ctx.saved_tensors
        eps_gap = ctx.eps_gap
        psi0 = evecs[:, :, 0]
        grad_h = torch.zeros_like(evecs)

        if grad_rho0 is not None:
            # U(1)-invariant sensitivity -(R_0^eps B_bar rho_0 + rho_0 B_bar R_0^eps) [Eq. (A.4)].
            b_bar = 0.5 * (grad_rho0 + grad_rho0.mH)
            b_psi0 = torch.einsum("bij,bj->bi", b_bar, psi0)
            overlaps = torch.einsum("bia,bi->ba", evecs.conj(), b_psi0)
            gaps = evals - evals[:, :1]
            inv_gaps = gaps / (gaps * gaps + eps_gap * eps_gap)
            chi = torch.einsum("bia,ba->bi", evecs, overlaps * inv_gaps.to(overlaps.dtype))
            grad_h = grad_h - (
                chi.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
                + psi0.unsqueeze(-1) * chi.conj().unsqueeze(-2)
            )

        if grad_e0 is not None:
            # Hellmann-Feynman differential dE_0 = Tr(rho_0 dH_tilde) [Eq. (A.2)].
            rho0 = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
            grad_h = grad_h + grad_e0.to(grad_h.dtype)[:, None, None] * rho0

        if grad_delta_e is not None:
            psi1 = evecs[:, :, 1]
            rho1 = psi1.unsqueeze(-1) * psi1.conj().unsqueeze(-2)
            rho0 = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
            grad_h = grad_h + grad_delta_e.to(grad_h.dtype)[:, None, None] * (rho1 - rho0)

        return grad_h, None


class _GibbsDensityState(torch.autograd.Function):
    """Thermal density state rho_beta(x) = exp(-beta H) / Tr exp(-beta H) [Eq. (2)] differentiated
    via the Daleckii-Krein spectral divided-difference kernel (Sec. A.2).
    """

    @staticmethod
    def forward(
        ctx: Any, H_tilde: torch.Tensor, beta: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evals, evecs = torch.linalg.eigh(H_tilde)
        e0 = evals[:, 0]
        delta_e = evals[:, 1] - evals[:, 0]
        shifted = evals - evals[:, :1]
        weights = torch.softmax(-beta * shifted, dim=-1)
        rho_beta = torch.einsum("bia,ba,bja->bij", evecs, weights.to(evecs.dtype), evecs.conj())
        ctx.save_for_backward(evals, evecs, weights, rho_beta)
        ctx.beta = float(beta)
        return e0, delta_e, rho_beta

    @staticmethod
    def backward(
        ctx: Any,
        grad_e0: torch.Tensor | None,
        grad_delta_e: torch.Tensor | None,
        grad_rho: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        evals, evecs, weights, rho_beta = ctx.saved_tensors
        beta = ctx.beta
        grad_h = torch.zeros_like(evecs)

        if grad_rho is not None:
            b_bar = 0.5 * (grad_rho + grad_rho.mH)
            b_eig = evecs.mH @ b_bar @ evecs
            b_diag = torch.diagonal(b_eig, dim1=-2, dim2=-1).real
            exp_b = torch.sum(weights * b_diag, dim=-1)

            # Divided difference -(beta/2)(w_a + w_c) tanh(u)/u with u = (beta/2)(E_a - E_c)
            # and Taylor expansion 1 - u^2/3 near u = 0.
            diff_e = evals.unsqueeze(-1) - evals.unsqueeze(-2)
            u = 0.5 * beta * diff_e
            small = u.abs() < 1e-4
            u_safe = torch.where(small, torch.ones_like(u), u)
            tanh_over_u = torch.where(small, 1.0 - (u * u) / 3.0, torch.tanh(u_safe) / u_safe)
            w_sum = weights.unsqueeze(-1) + weights.unsqueeze(-2)
            kernel = -0.5 * beta * w_sum * tanh_over_u

            scaled_eig = kernel.to(b_eig.dtype) * b_eig
            grad_h = (
                grad_h
                + (evecs @ scaled_eig @ evecs.mH)
                + (beta * exp_b).to(grad_h.dtype)[:, None, None] * rho_beta
            )

        if grad_e0 is not None:
            psi0 = evecs[:, :, 0]
            rho0 = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
            grad_h = grad_h + grad_e0.to(grad_h.dtype)[:, None, None] * rho0

        if grad_delta_e is not None:
            psi0 = evecs[:, :, 0]
            psi1 = evecs[:, :, 1]
            rho0 = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
            rho1 = psi1.unsqueeze(-1) * psi1.conj().unsqueeze(-2)
            grad_h = grad_h + grad_delta_e.to(grad_h.dtype)[:, None, None] * (rho1 - rho0)

        return grad_h, None


def _normalized_frobenius_sq(
    diag: torch.Tensor,
    off_re: torch.Tensor | None,
    off_im: torch.Tensor | None,
    hilbert_dim: int,
) -> torch.Tensor:
    """Mean-square eigenvalue norm (1/d)||O||_F^2 = (1/d)(sum_a O_{aa}^2 + 2 sum_{a<b} |O_{ab}|^2)
    equating shrinkage per degree of freedom across operator classes [Eq. (3), Sec. A.3].
    """
    frob_sq = torch.sum(diag * diag, dim=-1)
    if off_re is not None:
        frob_sq = frob_sq + 2.0 * torch.sum(off_re * off_re, dim=-1)
    if off_im is not None:
        frob_sq = frob_sq + 2.0 * torch.sum(off_im * off_im, dim=-1)
    return frob_sq / float(hilbert_dim)


class QCMLModel(nn.Module):
    """Matrix-form QCML over Herm(C^d), Sym(R^d), and commuting diag(R^d) (Sec. 2.1)."""

    VALID_MODES = ("hermitian", "real_sym", "commuting_diag")

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        hilbert_dim: int = 8,
        operator_mode: str = "hermitian",
        beta: float | None = None,
        eps_gap: float = 1e-5,
        unsupervised: bool = False,
        init_scale_a: float = 0.5,
        init_scale_b: float = 0.1,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if operator_mode not in self.VALID_MODES:
            raise ValueError(f"operator_mode must be in {self.VALID_MODES}, got {operator_mode!r}")
        if hilbert_dim < 2:
            raise ValueError(f"hilbert_dim must be >= 2, got {hilbert_dim}")

        self.p = int(n_features)
        self.q = int(n_outputs)
        self.d = int(hilbert_dim)
        self.operator_mode = operator_mode
        self.beta = None if beta is None else float(beta)
        self.eps_gap = float(eps_gap)
        self.unsupervised = bool(unsupervised)
        self.real_dtype = dtype
        self.complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        tri_r, tri_c = torch.triu_indices(self.d, self.d, offset=1)
        self.register_buffer("_tri_r", tri_r, persistent=False)
        self.register_buffer("_tri_c", tri_c, persistent=False)
        self._n_offdiag = int(tri_r.numel())

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))

        # Wigner d^{-1/2} variance scaling keeps initial operator norms O(1) across d.
        scale_a = init_scale_a / math.sqrt(self.d) if operator_mode != "commuting_diag" else init_scale_a
        scale_b = init_scale_b / math.sqrt(self.d) if operator_mode != "commuting_diag" else init_scale_b

        self.a_diag = nn.Parameter(torch.randn((self.p, self.d), generator=gen, dtype=dtype) * scale_a)
        if operator_mode in ("hermitian", "real_sym"):
            self.a_off_re = nn.Parameter(
                torch.randn((self.p, self._n_offdiag), generator=gen, dtype=dtype) * scale_a
            )
        else:
            self.register_parameter("a_off_re", None)
        if operator_mode == "hermitian":
            self.a_off_im = nn.Parameter(
                torch.randn((self.p, self._n_offdiag), generator=gen, dtype=dtype) * scale_a
            )
        else:
            self.register_parameter("a_off_im", None)

        if self.unsupervised:
            self.register_parameter("b_diag", None)
            self.register_parameter("b_off_re", None)
            self.register_parameter("b_off_im", None)
        else:
            self.b_diag = nn.Parameter(torch.randn((self.q, self.d), generator=gen, dtype=dtype) * scale_b)
            if operator_mode in ("hermitian", "real_sym"):
                self.b_off_re = nn.Parameter(
                    torch.randn((self.q, self._n_offdiag), generator=gen, dtype=dtype) * scale_b
                )
            else:
                self.register_parameter("b_off_re", None)
            if operator_mode == "hermitian":
                self.b_off_im = nn.Parameter(
                    torch.randn((self.q, self._n_offdiag), generator=gen, dtype=dtype) * scale_b
                )
            else:
                self.register_parameter("b_off_im", None)

    def _assemble_operators(
        self,
        diag: torch.Tensor,
        off_re: torch.Tensor | None,
        off_im: torch.Tensor | None,
    ) -> torch.Tensor:
        k_cnt = diag.shape[0]
        if self.operator_mode == "commuting_diag":
            return torch.diag_embed(diag)
        re = torch.diag_embed(diag)
        r, c = self._tri_r, self._tri_c
        re[:, r, c] = off_re
        re[:, c, r] = off_re
        if self.operator_mode == "real_sym":
            return re
        im = torch.zeros((k_cnt, self.d, self.d), dtype=self.real_dtype, device=diag.device)
        im[:, r, c] = off_im
        im[:, c, r] = -off_im
        return torch.complex(re, im)

    def get_feature_operators(self) -> torch.Tensor:
        return self._assemble_operators(self.a_diag, self.a_off_re, self.a_off_im)

    def get_target_operators(self) -> torch.Tensor | None:
        if self.unsupervised:
            return None
        return self._assemble_operators(self.b_diag, self.b_off_re, self.b_off_im)

    def parameter_counts(self) -> tuple[int, int]:
        """Raw real parameter count and identifiable degrees of freedom modulo PU(d), PO(d), or S_d
        conjugation orbits [Eq. (A.9)].
        """
        d, p = self.d, self.p
        q_eff = 0 if self.unsupervised else self.q
        if self.operator_mode == "hermitian":
            n_params = (p + q_eff) * (d * d)
            n_eff = (p + q_eff - 1) * (d * d) + 1
        elif self.operator_mode == "real_sym":
            sym_d = d * (d + 1) // 2
            so_d = d * (d - 1) // 2
            n_params = (p + q_eff) * sym_d
            n_eff = n_params - so_d
        else:
            n_params = (p + q_eff) * d
            n_eff = n_params
        return int(n_params), int(n_eff)

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        *,
        return_x_bar: bool = False,
        return_quantum_var: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Evaluate y_hat(x), E_0(x), E_0(x) - (1/2)||x||_2^2, Delta E(x), and optional position
        moments x_bar_k = Tr(rho A_k) and sigma^2 = sum_k [Tr(rho A_k^2) - x_bar_k^2].
        """
        x_r = x.to(dtype=self.real_dtype)
        half_x_norm2 = 0.5 * torch.sum(x_r * x_r, dim=-1)

        if self.operator_mode == "commuting_diag":
            # In the commuting basis A_k = diag(a_{k,0}, ..., a_{k,d-1}), eigenvalues equal
            # E_alpha(x) = (1/2)||a_{.,alpha} - x||_2^2 [Sec. 2.1].
            a_d = self.a_diag
            c0_d = 0.5 * torch.sum(a_d * a_d, dim=0)
            e_tilde_all = c0_d.unsqueeze(0) - x_r @ a_d
            e_all = e_tilde_all + half_x_norm2.unsqueeze(-1)
            top2_vals, top2_idx = torch.topk(e_all, k=2, dim=-1, largest=False, sorted=True)
            e0 = top2_vals[:, 0]
            e0_shifted = e0 - half_x_norm2
            delta_e = top2_vals[:, 1] - top2_vals[:, 0]

            if self.beta is None:
                best_idx = top2_idx[:, 0]
                if self.unsupervised:
                    y_hat = torch.zeros((len(x_r), self.q), dtype=self.real_dtype, device=x_r.device)
                else:
                    y_hat = self.b_diag.T[best_idx]
                out: dict[str, torch.Tensor] = {
                    "y_hat": y_hat,
                    "E_0": e0,
                    "E_0_shifted": e0_shifted,
                    "delta_E": delta_e,
                }
                if return_x_bar or return_quantum_var:
                    out["x_bar"] = a_d.T[best_idx]
                    if return_quantum_var:
                        out["quantum_var"] = torch.zeros_like(e0)
                return out

            w = torch.softmax(-self.beta * (e_all - e0.unsqueeze(-1)), dim=-1)
            if self.unsupervised:
                y_hat = torch.zeros((len(x_r), self.q), dtype=self.real_dtype, device=x_r.device)
            else:
                y_hat = w @ self.b_diag.T
            out = {"y_hat": y_hat, "E_0": e0, "E_0_shifted": e0_shifted, "delta_E": delta_e}
            if return_x_bar or return_quantum_var:
                x_bar = w @ a_d.T
                out["x_bar"] = x_bar
                if return_quantum_var:
                    x2_bar = w @ (a_d * a_d).T
                    out["quantum_var"] = torch.clamp(torch.sum(x2_bar - x_bar * x_bar, dim=-1), min=0.0)
            return out

        A_ops = self.get_feature_operators()
        C0 = 0.5 * torch.sum(A_ops @ A_ops, dim=0)
        C0 = 0.5 * (C0 + C0.mH)

        x_op = x_r.to(dtype=A_ops.dtype)
        H_tilde = C0.unsqueeze(0) - (x_op @ A_ops.reshape(self.p, self.d * self.d)).reshape(
            -1, self.d, self.d
        )

        if self.beta is None:
            e0_shifted, delta_e, rho = _TikhonovGroundState.apply(H_tilde, self.eps_gap)
        else:
            e0_shifted, delta_e, rho = _GibbsDensityState.apply(H_tilde, self.beta)

        e0 = e0_shifted + half_x_norm2
        if self.unsupervised:
            y_hat = torch.zeros((len(x_r), self.q), dtype=self.real_dtype, device=x_r.device)
        else:
            B_ops = self.get_target_operators()
            y_hat = torch.einsum("bij,mji->bm", rho, B_ops).real

        out = {"y_hat": y_hat, "E_0": e0, "E_0_shifted": e0_shifted, "delta_E": delta_e}
        if return_x_bar or return_quantum_var:
            x_bar = torch.einsum("bij,kji->bk", rho, A_ops).real
            out["x_bar"] = x_bar
            if return_quantum_var:
                # 2 E_0(x) = ||x_bar - x||_2^2 + sigma^2(rho), with sum_k Tr(rho A_k^2) = 2 Tr(rho C_0).
                two_tr_c0 = 2.0 * torch.einsum("bij,ji->b", rho, C0).real
                out["quantum_var"] = torch.clamp(two_tr_c0 - torch.sum(x_bar * x_bar, dim=-1), min=0.0)
        return out

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        *,
        lambda_e: float = 0.05,
        lambda_reg_a: float = 8e-3,
        lambda_reg_b: float | list[float] | torch.Tensor = 0.60,
        candelori_w: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Scale-invariant regularized objective L(A, B) [Eq. (3)], pure ground-energy objective
        (lambda_E = 1), or Candelori variance-relaxed manifold objective ||x_bar - x||_2^2 + w_var sigma^2.
        """
        need_candelori = candelori_w is not None
        diag = self.forward_with_diagnostics(
            x, return_x_bar=need_candelori, return_quantum_var=need_candelori
        )

        if need_candelori:
            x_r = x.to(dtype=self.real_dtype)
            bias_sq = torch.sum((diag["x_bar"] - x_r) ** 2, dim=-1)
            total = torch.mean(bias_sq + float(candelori_w) * diag["quantum_var"])
        elif self.unsupervised:
            total = torch.mean(diag["E_0"])
        else:
            y_t = y.to(dtype=self.real_dtype)
            if y_t.ndim == 1:
                y_t = y_t.unsqueeze(-1)
            total = torch.mean((diag["y_hat"] - y_t) ** 2)
            if lambda_e != 0.0:
                total = total + float(lambda_e) * torch.mean(diag["E_0"])

        if lambda_reg_a != 0.0:
            frob_a_norm = _normalized_frobenius_sq(
                self.a_diag, self.a_off_re, self.a_off_im, self.d
            )
            total = total + float(lambda_reg_a) * torch.sum(frob_a_norm)

        if not self.unsupervised and not need_candelori:
            frob_b_norm = _normalized_frobenius_sq(
                self.b_diag, self.b_off_re, self.b_off_im, self.d
            )
            if isinstance(lambda_reg_b, (int, float)):
                if float(lambda_reg_b) != 0.0:
                    total = total + float(lambda_reg_b) * torch.sum(frob_b_norm)
            else:
                w_b = torch.as_tensor(lambda_reg_b, dtype=self.real_dtype, device=x.device)
                total = total + torch.sum(w_b * frob_b_norm)

        return total, diag

    @torch.no_grad()
    def commutator_diagnostics(self, p_mkt: int = 12) -> dict[str, float]:
        """Frobenius-normalized commutator ratio kappa_nc(A) in [0, sqrt(2)] (Bottcher-Wenzel bound)
        and cross-modal market-news sub-sum kappa_nc^{mkt x news} [Eq. (5)].
        """
        A = self.get_feature_operators()
        norms = torch.linalg.matrix_norm(A, ord="fro")

        if self.p > p_mkt:
            mkt_mean = float(norms[:p_mkt].mean().item())
            news_mean = float(norms[p_mkt:].mean().item())
            norm_ratio = news_mean / (mkt_mean + 1e-12)
        else:
            norm_ratio = 0.0

        if self.operator_mode == "commuting_diag" or self.p < 2:
            return {
                "kappa_nc": 0.0,
                "kappa_nc_mkt_news": 0.0,
                "norm_ratio_news_mkt": float(norm_ratio),
            }

        num_all, den_all = 0.0, 1e-12
        num_mn, den_mn = 0.0, 1e-12
        for j in range(self.p):
            Aj = A[j]
            nj = float(norms[j].item())
            for k in range(j + 1, self.p):
                Ak = A[k]
                c_norm = float(torch.linalg.matrix_norm(Aj @ Ak - Ak @ Aj, ord="fro").item())
                prod = nj * float(norms[k].item())
                num_all += c_norm
                den_all += prod
                if j < p_mkt <= k:
                    num_mn += c_norm
                    den_mn += prod

        return {
            "kappa_nc": float(num_all / den_all),
            "kappa_nc_mkt_news": float(num_mn / den_mn) if self.p > p_mkt else 0.0,
            "norm_ratio_news_mkt": float(norm_ratio),
        }

    @torch.no_grad()
    def hermiticity_error(self) -> float:
        """Maximum Frobenius deviation ||O - O^dagger||_F across all feature and target operators."""
        A = self.get_feature_operators()
        err = float(torch.max(torch.linalg.matrix_norm(A - A.mH, ord="fro")).item())
        B = self.get_target_operators()
        if B is not None:
            err = max(err, float(torch.max(torch.linalg.matrix_norm(B - B.mH, ord="fro")).item()))
        return err

    @torch.no_grad()
    def export_state(
        self, y_mu: np.ndarray, y_sd: np.ndarray, is_1d: bool
    ) -> dict[str, Any]:
        A_np = self.get_feature_operators().detach().cpu().numpy()
        B_t = self.get_target_operators()
        B_np = None if B_t is None else B_t.detach().cpu().numpy()
        return {
            "operator_mode": self.operator_mode,
            "d": self.d,
            "p": self.p,
            "q": self.q,
            "beta": self.beta,
            "eps_gap": self.eps_gap,
            "unsupervised": self.unsupervised,
            "A_ops": A_np,
            "B_ops": B_np,
            "a_diag": self.a_diag.detach().cpu().numpy(),
            "b_diag": None if self.b_diag is None else self.b_diag.detach().cpu().numpy(),
            "y_mu": np.asarray(y_mu, dtype=np.float64),
            "y_sd": np.asarray(y_sd, dtype=np.float64),
            "is_1d": bool(is_1d),
        }


def make_gue_state(n_features: int, d: int, seed: int = 101) -> dict[str, Any]:
    """Untrained Gaussian Unitary Ensemble realization GUE(d) with E[(A_j)_{ab}(A_j)_{cd}] = sigma^2 delta_{ad} delta_{bc}
    and theoretical RMS commutator index kappa_nc^{GUE}(d) = sqrt((2/d)(1 - d^{-2})) [Eq. (5), Sec. A.3].
    """
    rng = np.random.default_rng(int(seed))
    scale = 0.5 / math.sqrt(float(d))
    re = rng.normal(0.0, scale, size=(n_features, d, d))
    im = rng.normal(0.0, scale, size=(n_features, d, d))
    M = re + 1j * im
    A_ops = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))
    return {
        "operator_mode": "hermitian",
        "d": int(d),
        "p": int(n_features),
        "q": 1,
        "beta": None,
        "eps_gap": 1e-5,
        "unsupervised": True,
        "A_ops": A_ops.astype(np.complex128),
        "B_ops": None,
        "a_diag": np.diagonal(A_ops, axis1=-2, axis2=-1).real.copy(),
        "b_diag": None,
        "y_mu": np.zeros((1, 1), dtype=np.float64),
        "y_sd": np.ones((1, 1), dtype=np.float64),
        "is_1d": True,
    }


@torch.no_grad()
def extract_spectral_observables(
    model_state: dict[str, Any],
    X: np.ndarray,
    date_idx: np.ndarray | None = None,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Spectral observables E_0(x), E_0(x) - (1/2)||x||_2^2, Delta E(x), scale-free gap
    Delta E_tilde(x) = Delta E(x) / sqrt(1 + ||x||_2^2) [Eq. (4)], and U(1)-gauge-invariant
    Fubini-Study step arccos|<psi_bar_t | psi_bar_{t-1}>| between principal eigenstates of
    daily mean ground projectors rho_bar_t (Sec. 2.2).
    """
    X_np = np.asarray(X, dtype=np.float64)
    n = len(X_np)
    d = int(model_state["d"])
    p = int(model_state["p"])
    q = int(model_state["q"])
    mode = str(model_state["operator_mode"])
    beta = model_state.get("beta", None)
    unsupervised = bool(model_state.get("unsupervised", False))
    y_mu = np.asarray(model_state.get("y_mu", np.zeros((1, q))), dtype=np.float64)
    y_sd = np.asarray(model_state.get("y_sd", np.ones((1, q))), dtype=np.float64)
    is_1d = bool(model_state.get("is_1d", q == 1))

    norm2 = np.sum(X_np * X_np, axis=1)
    half_norm2 = 0.5 * norm2

    e0_list: list[np.ndarray] = []
    e0_sh_list: list[np.ndarray] = []
    de_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []

    if date_idx is not None:
        di_np = np.asarray(date_idx)
        uniq_dates = np.unique(di_np)
        date_to_pos = {int(dv): idx for idx, dv in enumerate(uniq_dates)}
        day_pos_all = np.array([date_to_pos[int(dv)] for dv in di_np], dtype=np.int64)
        n_days = len(uniq_dates)
        rho_dtype = np.complex128 if mode == "hermitian" else np.float64
        if mode == "commuting_diag":
            day_diag_rho = np.zeros((n_days, d), dtype=np.float64)
        else:
            day_rho = np.zeros((n_days, d, d), dtype=rho_dtype)
        day_counts = np.zeros(n_days, dtype=np.int64)
    else:
        uniq_dates = None
        day_pos_all = None
        psi0_chunks: list[np.ndarray] = []

    if mode == "commuting_diag":
        a_d = torch.from_numpy(np.asarray(model_state["a_diag"], dtype=np.float64))
        b_d = (
            None
            if unsupervised or model_state.get("b_diag") is None
            else torch.from_numpy(np.asarray(model_state["b_diag"], dtype=np.float64))
        )
        c0_d = 0.5 * torch.sum(a_d * a_d, dim=0)
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            xb = torch.from_numpy(X_np[i0:i1])
            hn2 = torch.from_numpy(half_norm2[i0:i1])
            e_sh_all = c0_d.unsqueeze(0) - xb @ a_d
            e_all = e_sh_all + hn2.unsqueeze(-1)
            top2_v, top2_i = torch.topk(e_all, k=2, dim=-1, largest=False, sorted=True)
            e0_b = top2_v[:, 0]
            e0_sh_b = e0_b - hn2
            de_b = top2_v[:, 1] - top2_v[:, 0]
            best_i = top2_i[:, 0]
            if beta is None:
                w = torch.zeros_like(e_all)
                w.scatter_(1, best_i.unsqueeze(-1), 1.0)
            else:
                w = torch.softmax(-float(beta) * (e_all - e0_b.unsqueeze(-1)), dim=-1)
            if b_d is None:
                yp_b = np.zeros((i1 - i0, q), dtype=np.float64)
            else:
                yp_b = (w @ b_d.T).cpu().numpy()

            e0_list.append(e0_b.cpu().numpy())
            e0_sh_list.append(e0_sh_b.cpu().numpy())
            de_list.append(de_b.cpu().numpy())
            pred_list.append(yp_b)

            w_np = w.cpu().numpy()
            if date_idx is not None:
                pos_b = day_pos_all[i0:i1]
                np.add.at(day_diag_rho, pos_b, w_np)
                np.add.at(day_counts, pos_b, 1)
            else:
                psi_b = np.zeros((i1 - i0, d), dtype=np.float64)
                psi_b[np.arange(i1 - i0), best_i.cpu().numpy()] = 1.0
                psi0_chunks.append(psi_b)
    else:
        op_dtype = torch.complex128 if mode == "hermitian" else torch.float64
        A_ops = torch.as_tensor(model_state["A_ops"], dtype=op_dtype)
        B_ops = (
            None
            if unsupervised or model_state.get("B_ops") is None
            else torch.as_tensor(model_state["B_ops"], dtype=op_dtype)
        )
        C0 = 0.5 * torch.sum(A_ops @ A_ops, dim=0)
        C0 = 0.5 * (C0 + C0.mH)
        A_flat = A_ops.reshape(p, d * d)

        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            xb = torch.from_numpy(X_np[i0:i1]).to(dtype=op_dtype)
            H_tilde = C0.unsqueeze(0) - (xb @ A_flat).reshape(-1, d, d)
            evals, evecs = torch.linalg.eigh(H_tilde)
            e0_sh_b = evals[:, 0].cpu().numpy()
            e0_b = e0_sh_b + half_norm2[i0:i1]
            de_b = (evals[:, 1] - evals[:, 0]).cpu().numpy()
            psi0 = evecs[:, :, 0]

            if beta is None:
                rho = psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)
            else:
                w = torch.softmax(-float(beta) * (evals - evals[:, :1]), dim=-1)
                rho = torch.einsum("bia,ba,bja->bij", evecs, w.to(op_dtype), evecs.conj())

            if B_ops is None:
                yp_b = np.zeros((i1 - i0, q), dtype=np.float64)
            else:
                yp_b = torch.einsum("bij,mji->bm", rho, B_ops).real.cpu().numpy()

            e0_list.append(e0_b)
            e0_sh_list.append(e0_sh_b)
            de_list.append(de_b)
            pred_list.append(yp_b)

            if date_idx is not None:
                rho0_np = (psi0.unsqueeze(-1) * psi0.conj().unsqueeze(-2)).cpu().numpy()
                pos_b = day_pos_all[i0:i1]
                np.add.at(day_rho, pos_b, rho0_np)
                np.add.at(day_counts, pos_b, 1)
            else:
                psi0_chunks.append(psi0.cpu().numpy())

    e0_arr = np.concatenate(e0_list, axis=0)
    e0_sh_arr = np.concatenate(e0_sh_list, axis=0)
    de_arr = np.maximum(0.0, np.concatenate(de_list, axis=0))
    de_tilde_arr = de_arr / np.sqrt(1.0 + norm2)

    pred_s = np.concatenate(pred_list, axis=0)
    pred_raw = pred_s * y_sd + y_mu if not unsupervised else np.zeros_like(pred_s)
    if is_1d:
        pred_raw = pred_raw[:, 0]

    if date_idx is not None:
        n_days = len(uniq_dates)
        if mode == "commuting_diag":
            psi_days = np.zeros((n_days, d), dtype=np.float64)
            best_d = np.argmax(day_diag_rho / np.maximum(day_counts[:, None], 1), axis=1)
            psi_days[np.arange(n_days), best_d] = 1.0
        else:
            rho_mean = day_rho / np.maximum(day_counts[:, None, None], 1)
            _, vecs_d = np.linalg.eigh(rho_mean)
            psi_days = vecs_d[:, :, -1]
        fs_step = np.zeros(n_days, dtype=np.float64)
        if n_days > 1:
            overlaps = np.abs(np.sum(np.conj(psi_days[1:]) * psi_days[:-1], axis=1))
            fs_step[1:] = np.arccos(np.clip(overlaps, 0.0, 1.0))
    else:
        psi_all = np.concatenate(psi0_chunks, axis=0)
        fs_step = np.zeros(n, dtype=np.float64)
        if n > 1:
            overlaps = np.abs(np.sum(np.conj(psi_all[1:]) * psi_all[:-1], axis=1))
            fs_step[1:] = np.arccos(np.clip(overlaps, 0.0, 1.0))

    return {
        "E0": e0_arr,
        "E0_shifted": e0_sh_arr,
        "dE": de_arr,
        "dE_tilde": de_tilde_arr,
        "fubini_study_step": fs_step,
        "unique_dates": uniq_dates,
        "pred": pred_raw,
    }


def train_qcml_single(
    X_tr: np.ndarray,
    y_tr_s: np.ndarray | None,
    X_va: np.ndarray | None,
    y_va_s: np.ndarray | None,
    *,
    d: int,
    mode: str,
    q: int,
    beta: float | None,
    lambda_e: float,
    lambda_reg_a: float,
    lambda_reg_b: float | list[float],
    candelori_w: float | None,
    unsupervised: bool,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    eps_gap: float = 1e-5,
) -> tuple[QCMLModel, float]:
    """Minimize Eq. (3) (or unsupervised manifold objectives) with inner-validation early stopping."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    p = int(X_tr.shape[1])

    model = QCMLModel(
        n_features=p,
        n_outputs=q,
        hilbert_dim=d,
        operator_mode=mode,
        beta=beta,
        eps_gap=eps_gap,
        unsupervised=unsupervised,
        dtype=dtype,
        seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float64)).to(dtype=dtype)
    y_tr_t = None if y_tr_s is None else torch.from_numpy(np.asarray(y_tr_s, dtype=np.float64)).to(dtype=dtype)
    has_val = X_va is not None and len(X_va) > 0 and (unsupervised or (y_va_s is not None and len(y_va_s) > 0))
    X_va_t = None if not has_val else torch.from_numpy(np.asarray(X_va, dtype=np.float64)).to(dtype=dtype)
    y_va_t = (
        None
        if (not has_val or y_va_s is None)
        else torch.from_numpy(np.asarray(y_va_s, dtype=np.float64)).to(dtype=dtype)
    )

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
            loss, _ = model.compute_loss(
                X_tr_t[idx],
                None if y_tr_t is None else y_tr_t[idx],
                lambda_e=lambda_e,
                lambda_reg_a=lambda_reg_a,
                lambda_reg_b=lambda_reg_b,
                candelori_w=candelori_w,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                if candelori_w is not None:
                    vd = model.forward_with_diagnostics(X_va_t, return_x_bar=True, return_quantum_var=True)
                    val_metric = float(
                        torch.mean(
                            torch.sum((vd["x_bar"] - X_va_t) ** 2, dim=-1)
                            + float(candelori_w) * vd["quantum_var"]
                        ).item()
                    )
                elif unsupervised:
                    vd = model.forward_with_diagnostics(X_va_t)
                    val_metric = float(torch.mean(vd["E_0"]).item())
                else:
                    vd = model.forward_with_diagnostics(X_va_t)
                    val_metric = float(torch.mean((vd["y_hat"] - y_va_t) ** 2).item())

            if val_metric < best_val - 1e-6:
                best_val = val_metric
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


def fit_predict_qcml(
    mode: str,
    d: int,
    fold_dict: dict[str, Any],
    seed: int,
    lambda_e: float = 0.05,
    beta: float | None = None,
    multi_output: bool = False,
    candelori_w: float | None = None,
) -> dict[str, Any]:
    """Nested inner-validation selection of target shrinkage lambda_{reg, B} in Eq. (3) and
    out-of-sample evaluation on a single walk-forward fold (Sec. 3.2).
    """
    X_tr = np.asarray(fold_dict["X_tr"], dtype=np.float64)
    X_va = None if fold_dict.get("X_va") is None else np.asarray(fold_dict["X_va"], dtype=np.float64)
    X_te = np.asarray(fold_dict["X_te"], dtype=np.float64)
    p_mkt = int(fold_dict.get("p_mkt", 12))

    unsupervised = (candelori_w is not None) or (float(lambda_e) >= 0.9999)
    if unsupervised:
        q = 1
        is_1d = True
        y_mu = np.zeros((1, 1), dtype=np.float64)
        y_sd = np.ones((1, 1), dtype=np.float64)
        y_tr_s = None
        y_va_s = None
    else:
        y_tr_raw = np.asarray(fold_dict["y_tr"], dtype=np.float64)
        y_va_raw = None if fold_dict.get("y_va") is None else np.asarray(fold_dict["y_va"], dtype=np.float64)
        is_1d = (y_tr_raw.ndim == 1) and (not multi_output)
        y_tr_2d = y_tr_raw[:, None] if y_tr_raw.ndim == 1 else y_tr_raw
        y_va_2d = None if y_va_raw is None else (y_va_raw[:, None] if y_va_raw.ndim == 1 else y_va_raw)
        q = int(y_tr_2d.shape[1])
        y_mu = y_tr_2d.mean(axis=0, keepdims=True)
        y_sd = np.where(y_tr_2d.std(axis=0, keepdims=True) < 1e-8, 1.0, y_tr_2d.std(axis=0, keepdims=True))
        y_tr_s = (y_tr_2d - y_mu) / y_sd
        y_va_s = None if y_va_2d is None else (y_va_2d - y_mu) / y_sd

    lambda_reg_a = 1e-3 if unsupervised else 8e-3
    lr = 1e-2 if unsupervised else (5e-3 if multi_output else 3e-3)
    batch_size = 512
    hpo_epochs = 12
    full_epochs = 24 if unsupervised else 28
    patience = 5

    if unsupervised:
        model, best_val = train_qcml_single(
            X_tr,
            None,
            X_va,
            None,
            d=d,
            mode=mode,
            q=q,
            beta=beta,
            lambda_e=lambda_e,
            lambda_reg_a=lambda_reg_a,
            lambda_reg_b=0.0,
            candelori_w=candelori_w,
            unsupervised=True,
            lr=lr,
            batch_size=batch_size,
            max_epochs=full_epochs,
            patience=patience,
            seed=seed,
        )
        chosen_lreg_b: Any = 0.0
    else:
        if multi_output and q > 1:
            # Stronger shrinkage on the low-SNR return observable B_1 than on volatility/drawdown B_2..B_4 (Supplementary S2).
            grid: list[Any] = [
                [0.08] + [0.05] * (q - 1),
                [0.40] + [0.08] * (q - 1),
                [1.20] + [0.20] * (q - 1),
            ]
        else:
            grid = [0.15, 0.60, 2.0]

        if len(grid) == 1 or X_va is None or len(X_va) == 0:
            chosen_lreg_b = grid[0]
            model, best_val = train_qcml_single(
                X_tr,
                y_tr_s,
                X_va,
                y_va_s,
                d=d,
                mode=mode,
                q=q,
                beta=beta,
                lambda_e=lambda_e,
                lambda_reg_a=lambda_reg_a,
                lambda_reg_b=chosen_lreg_b,
                candelori_w=None,
                unsupervised=False,
                lr=lr,
                batch_size=batch_size,
                max_epochs=full_epochs,
                patience=patience,
                seed=seed,
            )
        else:
            best_hpo_val = float("inf")
            chosen_lreg_b = grid[0]
            for cand in grid:
                _, cand_val = train_qcml_single(
                    X_tr,
                    y_tr_s,
                    X_va,
                    y_va_s,
                    d=d,
                    mode=mode,
                    q=q,
                    beta=beta,
                    lambda_e=lambda_e,
                    lambda_reg_a=lambda_reg_a,
                    lambda_reg_b=cand,
                    candelori_w=None,
                    unsupervised=False,
                    lr=lr,
                    batch_size=batch_size,
                    max_epochs=hpo_epochs,
                    patience=3,
                    seed=seed,
                )
                if cand_val < best_hpo_val:
                    best_hpo_val = cand_val
                    chosen_lreg_b = cand

            model, best_val = train_qcml_single(
                X_tr,
                y_tr_s,
                X_va,
                y_va_s,
                d=d,
                mode=mode,
                q=q,
                beta=beta,
                lambda_e=lambda_e,
                lambda_reg_a=lambda_reg_a,
                lambda_reg_b=chosen_lreg_b,
                candelori_w=None,
                unsupervised=False,
                lr=lr,
                batch_size=batch_size,
                max_epochs=full_epochs,
                patience=patience,
                seed=seed,
            )

    model_state = model.export_state(y_mu=y_mu, y_sd=y_sd, is_1d=is_1d)
    n_params, n_eff = model.parameter_counts()
    comm = model.commutator_diagnostics(p_mkt=p_mkt)

    te_obs = extract_spectral_observables(model_state, X_te, date_idx=None)
    pred_te = te_obs["pred"]
    if X_va is not None and len(X_va) > 0:
        pred_va = extract_spectral_observables(model_state, X_va, date_idx=None)["pred"]
    else:
        pred_va = np.empty((0,) if is_1d else (0, q), dtype=np.float64)

    return {
        "pred_te": np.asarray(pred_te, dtype=np.float64),
        "pred_va": np.asarray(pred_va, dtype=np.float64),
        "n_params": int(n_params),
        "n_eff_params": int(n_eff),
        "kappa_nc": float(comm["kappa_nc"]),
        "kappa_nc_mkt_news": float(comm["kappa_nc_mkt_news"]),
        "norm_ratio_news_mkt": float(comm["norm_ratio_news_mkt"]),
        "hermiticity_err": float(model.hermiticity_error()),
        "best_val_loss": float(best_val),
        "best_lambda_reg_b": chosen_lreg_b,
        "model_state": model_state,
    }
