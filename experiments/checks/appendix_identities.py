"""Numerical verification of the analytical identities in Appendix A (Secs. A.1-A.3):
  1. Hellmann-Feynman energy differentials and Rayleigh-Schrodinger operator gradients
     vs central finite differences across Herm(C^d), Sym(R^d), and diag(R^d) (Eqs. A.2-A.5).
  2. Feature Jacobian J_{mk}(x) and vanishing Berry curvature F_{jk}(x) = 0 on Sym(R^d) (Eq. A.6).
  3. Tikhonov resolvent spectral bound ||R_0^eps||_2 <= (2 eps_gap)^{-1}, uniform sensitivity
     bound at exact degeneracies (Delta E = 0), and Daleckii-Krein finite-beta Gibbs gradients
     vs central finite differences (Eqs. A.7-A.8).
  4. Simultaneous PU(d)/PO(d) conjugation invariance, identifiable parameter counts N_eff (Eq. A.9),
     Bottcher-Wenzel bound <= sqrt(2), and GUE RMS ratio sqrt((2/d)(1 - d^{-2})).

Usage: python -I experiments/checks/appendix_identities.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcml_moex import QCMLModel, extract_spectral_observables, make_gue_state


def _build_hamiltonian_single(x_vec: torch.Tensor, A_ops: torch.Tensor) -> torch.Tensor:
    d = A_ops.shape[-1]
    eye = torch.eye(d, dtype=A_ops.dtype, device=A_ops.device)
    diffs = A_ops - x_vec.to(A_ops.dtype)[:, None, None] * eye.unsqueeze(0)
    H = 0.5 * torch.sum(diffs @ diffs, dim=0)
    return 0.5 * (H + H.mH)


def _check_hellmann_feynman_and_jacobian() -> dict[str, float]:
    p, d = 5, 4
    model = QCMLModel(
        n_features=p, n_outputs=1, hilbert_dim=d, operator_mode="hermitian", eps_gap=1e-12, dtype=torch.float64, seed=101
    )
    x = torch.randn(1, p, dtype=torch.float64, requires_grad=True)
    A = model.get_feature_operators()
    B = model.get_target_operators()
    H = _build_hamiltonian_single(x[0], A)
    evals, evecs = torch.linalg.eigh(H)
    psi0 = evecs[:, 0]

    # Eq. (A.2): dE_0 / dx_k = x_k - <psi_0 | A_k | psi_0>
    grad_x_auto = torch.autograd.grad(evals[0], x)[0][0]
    x_bar = torch.stack([torch.real(torch.vdot(psi0, A[k] @ psi0)) for k in range(p)])
    hf_x_err = float(torch.max(torch.abs(grad_x_auto - (x.detach()[0] - x_bar))).item())

    # Eq. (A.6): J_{1k}(x) = 2 Re <psi_0 | A_k R_0 B_1 | psi_0>
    deltas = evals[1:] - evals[0]
    V_exc = evecs[:, 1:]
    R0 = (V_exc * (1.0 / deltas).unsqueeze(0)) @ V_exc.conj().T
    B0 = B[0]
    J_analytic = np.array(
        [float(2.0 * torch.real(torch.vdot(psi0, A[k] @ R0 @ B0 @ psi0)).item()) for k in range(p)]
    )
    eps_fd = 1e-6
    J_fd = np.zeros(p, dtype=np.float64)
    for k in range(p):
        xp = x.detach().clone()
        xm = x.detach().clone()
        xp[0, k] += eps_fd
        xm[0, k] -= eps_fd
        yp = model.forward_with_diagnostics(xp)["y_hat"]
        ym = model.forward_with_diagnostics(xm)["y_hat"]
        J_fd[k] = float((yp[0, 0] - ym[0, 0]).item()) / (2.0 * eps_fd)
    jac_err = float(np.max(np.abs(J_analytic - J_fd)))

    # Eq. (A.6): Berry curvature F_{jk}(x) = -2 Im Q_{jk}(x) vanishes identically on Sym(R^d)
    model_real = QCMLModel(
        n_features=p, n_outputs=1, hilbert_dim=d, operator_mode="real_sym", eps_gap=1e-12, dtype=torch.float64, seed=102
    )
    Ar = model_real.get_feature_operators()
    Hr = _build_hamiltonian_single(x.detach()[0], Ar)
    ev_r, vc_r = torch.linalg.eigh(Hr)
    psi0_r = vc_r[:, 0]
    V_exc_r = vc_r[:, 1:]
    R0_r = (V_exc_r * (1.0 / (ev_r[1:] - ev_r[0])).unsqueeze(0)) @ V_exc_r.T
    Q_mat = torch.zeros((p, p), dtype=torch.float64)
    for j in range(p):
        for k in range(p):
            Q_mat[j, k] = torch.dot(psi0_r, Ar[j] @ (R0_r @ R0_r) @ Ar[k] @ psi0_r)
    berry_asym_real = float(torch.max(torch.abs(Q_mat - Q_mat.T)).item())
    return {"hf_dx_err": hf_x_err, "jacobian_fd_err": jac_err, "real_sym_berry_err": berry_asym_real}


def _check_operator_gradients_vs_fd() -> dict[str, float]:
    errs: dict[str, float] = {}
    eps_fd = 1e-6
    for label, mode, beta in [
        ("herm_tikhonov", "hermitian", None),
        ("realsym_tikhonov", "real_sym", None),
        ("herm_gibbs_dk", "hermitian", 2.5),
        ("diag_gibbs", "commuting_diag", 2.5),
    ]:
        torch.manual_seed(202)
        m = QCMLModel(
            n_features=4,
            n_outputs=2,
            hilbert_dim=4,
            operator_mode=mode,
            beta=beta,
            eps_gap=1e-8,
            dtype=torch.float64,
            seed=202,
        )
        X = torch.randn(6, 4, dtype=torch.float64)
        Y = torch.randn(6, 2, dtype=torch.float64)

        def _loss_fn() -> torch.Tensor:
            loss_t, _ = m.compute_loss(X, Y, lambda_e=0.05, lambda_reg_a=1e-3, lambda_reg_b=1e-3)
            return loss_t

        m.zero_grad()
        _loss_fn().backward()

        max_rel = 0.0
        for p_tensor in m.parameters():
            assert p_tensor.grad is not None
            g_auto = p_tensor.grad.detach().clone().reshape(-1)
            flat = p_tensor.data.reshape(-1)
            for idx in range(min(12, flat.numel())):
                orig = float(flat[idx].item())
                flat[idx] = orig + eps_fd
                lp = float(_loss_fn().item())
                flat[idx] = orig - eps_fd
                lm = float(_loss_fn().item())
                flat[idx] = orig
                g_fd = (lp - lm) / (2.0 * eps_fd)
                denom = max(abs(float(g_auto[idx].item())), abs(g_fd), 1e-6)
                max_rel = max(max_rel, abs(float(g_auto[idx].item()) - g_fd) / denom)
        errs[label] = max_rel
    return errs


def _check_resolvent_bounds() -> dict[str, float]:
    eps_gap = 1e-5
    theo_max = 1.0 / (2.0 * eps_gap)
    deltas = np.geomspace(1e-12, 1e1, 20000)
    damped = deltas / (deltas**2 + eps_gap**2)
    attained_at_eps = eps_gap / (eps_gap**2 + eps_gap**2)

    # Exact level-crossing stress: H = 0 => Delta_alpha = 0 => G_{tilde H} - lambda_E rho_0 = 0
    m = QCMLModel(
        n_features=3, n_outputs=1, hilbert_dim=4, operator_mode="hermitian", eps_gap=eps_gap, dtype=torch.float64, seed=7
    )
    with torch.no_grad():
        for par in m.parameters():
            par.zero_()
    X0 = torch.zeros(2, 3, dtype=torch.float64)
    Y0 = torch.ones(2, 1, dtype=torch.float64)
    loss, _ = m.compute_loss(X0, Y0, lambda_e=0.05, lambda_reg_a=0.0, lambda_reg_b=0.0)
    loss.backward()
    finite_at_crossing = all(torch.isfinite(par.grad).all().item() for par in m.parameters() if par.grad is not None)
    return {
        "resolvent_bound_excess": float(np.max(damped) - theo_max),
        "attained_rel_err": abs(attained_at_eps - theo_max) / theo_max,
        "crossing_grad_finite": float(finite_at_crossing),
    }


def _check_gauge_neff_and_gue() -> dict[str, object]:
    # Exact parameter counts (N_raw, N_eff) on M+N (p=18, q=1) vs Appendix A.3 Eq. (A.9)
    counts = {
        "herm_d8": QCMLModel(n_features=18, n_outputs=1, hilbert_dim=8, operator_mode="hermitian").parameter_counts(),
        "realsym_d11": QCMLModel(n_features=18, n_outputs=1, hilbert_dim=11, operator_mode="real_sym").parameter_counts(),
        "diag_d64": QCMLModel(n_features=18, n_outputs=1, hilbert_dim=64, operator_mode="commuting_diag").parameter_counts(),
    }
    assert counts["herm_d8"] == (1216, 1153)
    assert counts["realsym_d11"] == (1254, 1199)
    assert counts["diag_d64"] == (1216, 1216)

    # Simultaneous unitary conjugation O -> V O V^dagger leaves E_0, Delta E, yhat invariant
    m_ref = QCMLModel(
        n_features=6, n_outputs=1, hilbert_dim=8, operator_mode="hermitian", dtype=torch.float64, seed=303
    )
    st = m_ref.export_state(np.array([0.0]), np.array([1.0]), True)
    rng = np.random.default_rng(404)
    Z = (rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))) / math.sqrt(2.0)
    Q_unitary, R_tri = np.linalg.qr(Z)
    phases = np.diag(R_tri) / np.abs(np.diag(R_tri))
    V = Q_unitary * phases[None, :]

    st_conj = dict(st)
    st_conj["A_ops"] = np.einsum("ab,kbc,dc->kad", V, st["A_ops"], V.conj())
    st_conj["B_ops"] = np.einsum("ab,mbc,dc->mad", V, st["B_ops"], V.conj())
    X_probe = rng.standard_normal((32, 6))
    obs_1 = extract_spectral_observables(st, X_probe)
    obs_2 = extract_spectral_observables(st_conj, X_probe)
    conj_err = max(
        float(np.max(np.abs(obs_1["E0"] - obs_2["E0"]))),
        float(np.max(np.abs(obs_1["dE"] - obs_2["dE"]))),
        float(np.max(np.abs(obs_1["pred"] - obs_2["pred"]))),
    )

    # GUE(d=8) Wick RMS ratio sqrt((2/d)(1 - d^{-2})) and Bottcher-Wenzel bound <= sqrt(2)
    d = 8
    theo_gue = math.sqrt((2.0 / d) * (1.0 - 1.0 / (d**2)))
    gue_draws = []
    max_pair_ratio = 0.0
    for s in range(50):
        gs = make_gue_state(n_features=18, d=d, seed=1000 + s)
        A = gs["A_ops"]
        norms = np.sqrt(np.sum(np.abs(A) ** 2, axis=(1, 2)))
        pair_sq = []
        for j in range(18):
            for k in range(j + 1, 18):
                comm = A[j] @ A[k] - A[k] @ A[j]
                r = float(np.linalg.norm(comm, ord="fro") / (norms[j] * norms[k]))
                max_pair_ratio = max(max_pair_ratio, r)
                pair_sq.append(r**2)
        gue_draws.append(math.sqrt(float(np.mean(pair_sq))))
    rms_gue_mc = math.sqrt(float(np.mean(np.square(gue_draws))))

    return {
        "counts_mn_p18": counts,
        "pu_d_conjugation_err": conj_err,
        "bottcher_wenzel_max": max_pair_ratio,
        "gue_d8_theory": round(theo_gue, 6),
        "gue_d8_mc_rms": round(rms_gue_mc, 6),
    }


def main() -> int:
    hf = _check_hellmann_feynman_and_jacobian()
    grad_fd = _check_operator_gradients_vs_fd()
    res_b = _check_resolvent_bounds()
    gauge = _check_gauge_neff_and_gue()

    print("Appendix A analytical & numerical identity check:")
    print(f"  [A.1] Hellmann-Feynman dE0/dx error         : {hf['hf_dx_err']:.2e}")
    print(f"  [A.1] Feature Jacobian J(x) vs FD error     : {hf['jacobian_fd_err']:.2e}")
    print(f"  [A.1] Sym(R^d) Berry curvature |F_jk| max   : {hf['real_sym_berry_err']:.2e}")
    for k, v in grad_fd.items():
        print(f"  [A.1-A.2] Autograd vs FD ({k:<16s}) : {v:.2e}")
    print(f"  [A.2] Tikhonov ||R_0^eps||_2 bound excess   : {res_b['resolvent_bound_excess']:.2e}")
    print(f"  [A.2] Degeneracy (Delta E=0) finite grads   : {bool(res_b['crossing_grad_finite'])}")
    print(f"  [A.3] M+N (p=18, q=1) (N_raw, N_eff)        : {gauge['counts_mn_p18']}")
    print(f"  [A.3] PU(8) conjugation invariance error    : {gauge['pu_d_conjugation_err']:.2e}")
    print(f"  [A.3] Bottcher-Wenzel max ratio (<= 1.4142) : {gauge['bottcher_wenzel_max']:.4f}")
    print(
        f"  [A.3] GUE(d=8) kappa_nc theory vs MC RMS    : "
        f"{gauge['gue_d8_theory']:.4f} vs {gauge['gue_d8_mc_rms']:.4f}"
    )

    ok = (
        hf["hf_dx_err"] < 1e-12
        and hf["jacobian_fd_err"] < 1e-6
        and hf["real_sym_berry_err"] < 1e-12
        and all(v < 1e-5 for v in grad_fd.values())
        and res_b["resolvent_bound_excess"] <= 1e-9
        and res_b["crossing_grad_finite"] == 1.0
        and float(gauge["pu_d_conjugation_err"]) < 1e-12
        and float(gauge["bottcher_wenzel_max"]) <= math.sqrt(2.0) + 1e-12
        and abs(float(gauge["gue_d8_theory"]) - float(gauge["gue_d8_mc_rms"])) < 0.01
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
