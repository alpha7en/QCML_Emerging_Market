"""Classical control ladder M_0 (Sec. 3.2) and synthetic manifold controls (Supplementary Material S1).

Defines parameter-free cross-sectional factors (Mom5, LowVol, Mom21, Mom21-LowVol), ridge
regression, random Fourier features kernel ridge regression (Bochner cosine expansion of the
shift-invariant Gaussian RBF kernel), histogram gradient-boosted trees, LightGBM, single-hidden-
layer ReLU and tanh perceptrons, and a bounded gated bilinear network h = tanh(U LN(x)) odot
tanh(V LN(x)) matched to the (p + q) * 64 real parameter count of d = 8 Hermitian QCML.
Also implements the synthetic positive controls on S^1, S^2, and the d = 3 spin-1 quadrupole
y = u_x u_y on S^2 (Supplementary Material S1, Table S1).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .qcml import extract_spectral_observables, train_qcml_single

MARKET_COL: dict[str, int] = {
    "ret_1d": 0,
    "ret_5d": 1,
    "ret_21d": 2,
    "ret_63d": 3,
    "vol_21d": 4,
    "turnover_shock": 5,
    "beta_252d": 6,
    "cs_rank_ret_21d": 7,
    "cs_rank_vol_21d": 8,
    "cs_rank_adv_21d": 9,
    "imoex_ret_5d": 10,
    "imoex_vol_21d": 11,
}


class _IsoMLP(nn.Module):
    """Single-hidden-layer perceptron f(x) = W_2 sigma(W_1 x + b_1) + b_2 with hidden width chosen
    to match the real parameter budget (p + q) d^2 at d = 8 (Sec. 3.2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1, act: str = "tanh") -> None:
        super().__init__()
        self.act = act
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x)) if self.act == "tanh" else torch.relu(self.fc1(x))
        return self.fc2(h)


class _BoundedFiLMMLP(nn.Module):
    """Layer-normalized multiplicative network f(x) = W_o (tanh(U LN(x)) odot tanh(V LN(x))) + b_o
    whose hidden coordinates lie in [-1, 1]^h, bounding forecasts by ||W_o||_1 + |b_o| (Sec. 3.2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc_u = nn.Linear(in_dim, hidden_dim)
        self.fc_v = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ln(x)
        h = torch.tanh(self.fc_u(u)) * torch.tanh(self.fc_v(u))
        return self.out(h)


def _train_torch_tabular(
    model_ctor: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray | None,
    y_va: np.ndarray | None,
    X_te: np.ndarray,
    *,
    seed: int,
    wd_grid: list[float],
    lr: float = 2e-3,
    batch_size: int = 512,
    max_epochs: int = 36,
    patience: int = 6,
) -> tuple[np.ndarray, np.ndarray, int, Any]:
    is_1d = y_tr.ndim == 1
    y_tr_2d = y_tr[:, None].astype(np.float64) if is_1d else y_tr.astype(np.float64)
    y_va_2d = (
        None
        if (y_va is None or len(y_va) == 0)
        else (y_va[:, None].astype(np.float64) if y_va.ndim == 1 else y_va.astype(np.float64))
    )
    y_mu = y_tr_2d.mean(axis=0, keepdims=True)
    y_sd = np.where(y_tr_2d.std(axis=0, keepdims=True) < 1e-8, 1.0, y_tr_2d.std(axis=0, keepdims=True))
    y_tr_s = ((y_tr_2d - y_mu) / y_sd).astype(np.float32)
    y_va_s = None if y_va_2d is None else ((y_va_2d - y_mu) / y_sd).astype(np.float32)

    X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32))
    y_tr_t = torch.from_numpy(y_tr_s)
    has_val = X_va is not None and y_va_s is not None and len(X_va) > 0
    X_va_t = None if not has_val else torch.from_numpy(np.asarray(X_va, dtype=np.float32))
    y_va_t = None if not has_val else torch.from_numpy(y_va_s)

    wds = wd_grid[:1] if not has_val else wd_grid
    best_overall_val = float("inf")
    best_model: nn.Module | None = None

    for wd in wds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        m = model_ctor()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(int(seed))
        best_v = float("inf")
        best_st = {k: v.detach().clone() for k, v in m.state_dict().items()}
        bad = 0
        for _ in range(max_epochs):
            m.train()
            perm = torch.randperm(len(X_tr_t), generator=gen)
            for i0 in range(0, len(X_tr_t), batch_size):
                ix = perm[i0 : i0 + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(m(X_tr_t[ix]), y_tr_t[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
            if has_val:
                m.eval()
                with torch.no_grad():
                    v = float(nn.functional.mse_loss(m(X_va_t), y_va_t).item())
                if v < best_v - 1e-6:
                    best_v = v
                    best_st = {k: v_.detach().clone() for k, v_ in m.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if has_val:
            m.load_state_dict(best_st)
        m.eval()
        if best_v < best_overall_val or best_model is None:
            best_overall_val = best_v
            best_model = m

    n_params = sum(param.numel() for param in best_model.parameters() if param.requires_grad)

    @torch.no_grad()
    def _pred(X_arr: np.ndarray) -> np.ndarray:
        best_model.eval()
        xt = torch.from_numpy(np.asarray(X_arr, dtype=np.float32))
        out = best_model(xt).cpu().numpy().astype(np.float64) * y_sd + y_mu
        return out[:, 0] if is_1d else out

    pred_te = _pred(X_te)
    pred_va = (
        np.empty((0,) if is_1d else (0, y_tr_2d.shape[1]), dtype=np.float64)
        if not has_val
        else _pred(X_va)
    )
    return pred_te, pred_va, int(n_params), lambda X_o: {"pred": _pred(X_o)}


def fit_predict_classical(
    model_name: str,
    fold_dict: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit any classical control in M_0 (Tier 0 and Tier 1, scalar or q = 4 multi-output; Sec. 3.2)."""
    X_tr = np.asarray(fold_dict["X_tr"], dtype=np.float64)
    y_tr = np.asarray(fold_dict["y_tr"], dtype=np.float64)
    X_va = None if fold_dict.get("X_va") is None else np.asarray(fold_dict["X_va"], dtype=np.float64)
    y_va = None if fold_dict.get("y_va") is None else np.asarray(fold_dict["y_va"], dtype=np.float64)
    X_te = np.asarray(fold_dict["X_te"], dtype=np.float64)
    has_val = X_va is not None and y_va is not None and len(X_va) > 0
    p = int(X_tr.shape[1])

    if model_name in ("zero", "mom5", "lowvol", "mom21", "mom21_lowvol"):
        tr_mean = float(np.mean(y_tr))

        def _tier0(X_arr: np.ndarray) -> np.ndarray:
            if model_name == "zero":
                return np.full(len(X_arr), tr_mean, dtype=np.float64)
            if model_name == "mom5":
                return X_arr[:, MARKET_COL["ret_5d"]].astype(np.float64)
            if model_name == "lowvol":
                return (-X_arr[:, MARKET_COL["vol_21d"]]).astype(np.float64)
            if model_name == "mom21":
                return X_arr[:, MARKET_COL["ret_21d"]].astype(np.float64)
            return (X_arr[:, MARKET_COL["ret_21d"]] - X_arr[:, MARKET_COL["vol_21d"]]).astype(np.float64)

        pred_te = _tier0(X_te)
        pred_va = np.empty((0,), dtype=np.float64) if not has_val else _tier0(X_va)
        return {
            "pred_te": pred_te,
            "pred_va": pred_va,
            "n_params": 0,
            "model_name": model_name,
            "ood_fn": lambda X_o: {"pred": _tier0(np.asarray(X_o, dtype=np.float64))},
        }

    if model_name == "ridge":
        y_tr_1d = y_tr.ravel()
        alphas = [100.0] if not has_val else [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
        best_a, best_mse = alphas[0], float("inf")
        if has_val:
            y_va_1d = y_va.ravel()
            for a in alphas:
                m = Ridge(alpha=a).fit(X_tr, y_tr_1d)
                mse = float(np.mean((y_va_1d - m.predict(X_va)) ** 2))
                if mse < best_mse:
                    best_mse, best_a = mse, a
        mdl = Ridge(alpha=best_a).fit(X_tr, y_tr_1d)
        pred_te = np.asarray(mdl.predict(X_te), dtype=np.float64)
        pred_va = np.empty((0,), dtype=np.float64) if not has_val else np.asarray(mdl.predict(X_va), dtype=np.float64)
        return {
            "pred_te": pred_te,
            "pred_va": pred_va,
            "n_params": int(p + 1),
            "model_name": model_name,
            "best_alpha": float(best_a),
            "ood_fn": lambda X_o: {"pred": np.asarray(mdl.predict(np.asarray(X_o, dtype=np.float64)), dtype=np.float64)},
        }

    if model_name == "ridge_multi":
        y_tr_2d = y_tr[:, None] if y_tr.ndim == 1 else y_tr
        y_va_2d = None if not has_val else (y_va[:, None] if y_va.ndim == 1 else y_va)
        q = int(y_tr_2d.shape[1])
        y_mu = y_tr_2d.mean(axis=0, keepdims=True)
        y_sd = np.where(y_tr_2d.std(axis=0, keepdims=True) < 1e-8, 1.0, y_tr_2d.std(axis=0, keepdims=True))
        y_tr_s = (y_tr_2d - y_mu) / y_sd
        y_va_s = None if y_va_2d is None else (y_va_2d - y_mu) / y_sd
        alphas = [100.0] if not has_val else [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
        best_a, best_mse = alphas[0], float("inf")
        if has_val:
            for a in alphas:
                m = Ridge(alpha=a).fit(X_tr, y_tr_s)
                mse = float(np.mean((y_va_s - m.predict(X_va)) ** 2))
                if mse < best_mse:
                    best_mse, best_a = mse, a
        mdl = Ridge(alpha=best_a).fit(X_tr, y_tr_s)
        pred_te = mdl.predict(X_te) * y_sd + y_mu
        pred_va = np.empty((0, q), dtype=np.float64) if not has_val else (mdl.predict(X_va) * y_sd + y_mu)
        return {
            "pred_te": np.asarray(pred_te, dtype=np.float64),
            "pred_va": np.asarray(pred_va, dtype=np.float64),
            "n_params": int(q * (p + 1)),
            "model_name": model_name,
            "best_alpha": float(best_a),
            "ood_fn": lambda X_o: {"pred": np.asarray(mdl.predict(np.asarray(X_o, dtype=np.float64)) * y_sd + y_mu, dtype=np.float64)},
        }

    if model_name == "rff_krr":
        # Bochner random Fourier feature map phi(x) = sqrt(2/D) cos(W^T x + b), W_{kj} ~ N(0, 2 gamma),
        # approximating k(x, x') = exp(-gamma ||x - x'||_2^2) at D = 1024.
        y_tr_1d = y_tr.ravel()
        y_mu = float(y_tr_1d.mean())
        D = 1024
        gammas = [0.05] if not has_val else [0.02, 0.05, 0.1, 0.25]
        lams = [1.0, 10.0, 100.0, 1000.0]
        rng = np.random.default_rng(int(seed))
        best = (float("inf"), None)
        for g in gammas:
            W = rng.normal(0.0, math.sqrt(2.0 * g), size=(p, D))
            b = rng.uniform(0.0, 2.0 * math.pi, size=(D,))
            scale = math.sqrt(2.0 / float(D))
            Phi_tr = scale * np.cos(X_tr @ W + b)
            G = Phi_tr.T @ Phi_tr
            rhs = Phi_tr.T @ (y_tr_1d - y_mu)
            Phi_va = None if not has_val else (scale * np.cos(X_va @ W + b))
            for lam in lams:
                w = np.linalg.solve(G + lam * np.eye(D), rhs)
                mse = float(np.mean((y_va.ravel() - (y_mu + Phi_va @ w)) ** 2)) if has_val else 0.0
                if mse < best[0]:
                    best = (mse, (g, lam, W.copy(), b.copy(), w.copy()))
        g_b, lam_b, W_b, b_b, w_b = best[1]
        scale = math.sqrt(2.0 / float(D))

        def _pred_rff(X_arr: np.ndarray) -> np.ndarray:
            return (y_mu + scale * np.cos(X_arr @ W_b + b_b) @ w_b).astype(np.float64)

        return {
            "pred_te": _pred_rff(X_te),
            "pred_va": np.empty((0,), dtype=np.float64) if not has_val else _pred_rff(X_va),
            "n_params": int(D + 1),
            "model_name": model_name,
            "best_gamma": float(g_b),
            "best_lam": float(lam_b),
            "ood_fn": lambda X_o: {"pred": _pred_rff(np.asarray(X_o, dtype=np.float64))},
        }

    if model_name == "hgbdt":
        y_tr_1d = y_tr.ravel()
        grid = (
            [{"max_depth": 3, "l2_regularization": 100.0, "min_samples_leaf": 100}]
            if not has_val
            else [
                {"max_depth": 3, "l2_regularization": 10.0, "min_samples_leaf": 50},
                {"max_depth": 3, "l2_regularization": 100.0, "min_samples_leaf": 100},
                {"max_depth": 4, "l2_regularization": 100.0, "min_samples_leaf": 100},
            ]
        )
        max_iter = 120
        best_mse, best_cfg, best_mdl = float("inf"), grid[0], None
        for cfg in grid:
            g = HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=max_iter,
                early_stopping=False,
                random_state=int(seed),
                **cfg,
            ).fit(X_tr, y_tr_1d)
            mse = float(np.mean((y_va.ravel() - g.predict(X_va)) ** 2)) if has_val else 0.0
            if mse < best_mse or best_mdl is None:
                best_mse, best_cfg, best_mdl = mse, cfg, g
        return {
            "pred_te": np.asarray(best_mdl.predict(X_te), dtype=np.float64),
            "pred_va": np.empty((0,), dtype=np.float64)
            if not has_val
            else np.asarray(best_mdl.predict(X_va), dtype=np.float64),
            "n_params": int(max_iter * (2 ** best_cfg["max_depth"])),
            "model_name": model_name,
            "best_cfg": best_cfg,
            "ood_fn": lambda X_o: {"pred": np.asarray(best_mdl.predict(np.asarray(X_o, dtype=np.float64)), dtype=np.float64)},
        }

    if model_name == "lgbm":
        import lightgbm as lgb

        y_tr_1d = y_tr.ravel()
        grid = (
            [{"max_depth": 3, "reg_lambda": 100.0, "min_child_samples": 100}]
            if not has_val
            else [
                {"max_depth": 3, "reg_lambda": 10.0, "min_child_samples": 50},
                {"max_depth": 3, "reg_lambda": 100.0, "min_child_samples": 100},
                {"max_depth": 4, "reg_lambda": 100.0, "min_child_samples": 100},
            ]
        )
        n_est = 120
        best_mse, best_cfg, best_mdl = float("inf"), grid[0], None
        for cfg in grid:
            g = lgb.LGBMRegressor(
                n_estimators=n_est,
                learning_rate=0.04,
                num_leaves=2 ** cfg["max_depth"],
                random_state=int(seed),
                n_jobs=1,
                verbose=-1,
                **cfg,
            ).fit(X_tr, y_tr_1d)
            mse = float(np.mean((y_va.ravel() - g.predict(X_va)) ** 2)) if has_val else 0.0
            if mse < best_mse or best_mdl is None:
                best_mse, best_cfg, best_mdl = mse, cfg, g
        return {
            "pred_te": np.asarray(best_mdl.predict(X_te), dtype=np.float64),
            "pred_va": np.empty((0,), dtype=np.float64)
            if not has_val
            else np.asarray(best_mdl.predict(X_va), dtype=np.float64),
            "n_params": int(n_est * (2 ** best_cfg["max_depth"])),
            "model_name": model_name,
            "best_cfg": best_cfg,
            "ood_fn": lambda X_o: {"pred": np.asarray(best_mdl.predict(np.asarray(X_o, dtype=np.float64)), dtype=np.float64)},
        }

    if model_name in ("mlp_relu", "mlp_tanh"):
        target_budget = (p + 1) * 64
        h_dim = max(4, round((target_budget - 1) / float(p + 2)))
        act = "relu" if model_name == "mlp_relu" else "tanh"
        pred_te, pred_va, n_par, ood_fn = _train_torch_tabular(
            lambda: _IsoMLP(p, h_dim, out_dim=1, act=act),
            X_tr,
            y_tr.ravel(),
            X_va,
            None if y_va is None else y_va.ravel(),
            X_te,
            seed=seed,
            wd_grid=[1e-2, 5e-2],
        )
        return {
            "pred_te": pred_te,
            "pred_va": pred_va,
            "n_params": n_par,
            "model_name": model_name,
            "ood_fn": ood_fn,
        }

    if model_name == "film_mlp_iso":
        target_budget = (p + 1) * 64
        h_dim = max(4, round((target_budget - 2 * p - 1) / float(2 * p + 3)))
        pred_te, pred_va, n_par, ood_fn = _train_torch_tabular(
            lambda: _BoundedFiLMMLP(p, h_dim, out_dim=1),
            X_tr,
            y_tr.ravel(),
            X_va,
            None if y_va is None else y_va.ravel(),
            X_te,
            seed=seed,
            wd_grid=[1e-2, 5e-2],
        )
        return {
            "pred_te": pred_te,
            "pred_va": pred_va,
            "n_params": n_par,
            "model_name": model_name,
            "ood_fn": ood_fn,
        }

    if model_name == "mlp_tanh_multi":
        y_tr_2d = y_tr[:, None] if y_tr.ndim == 1 else y_tr
        y_va_2d = None if not has_val else (y_va[:, None] if y_va.ndim == 1 else y_va)
        q = int(y_tr_2d.shape[1])
        target_budget = (p + q) * 64
        h_dim = max(4, round((target_budget - q) / float(p + q + 1)))
        pred_te, pred_va, n_par, ood_fn = _train_torch_tabular(
            lambda: _IsoMLP(p, h_dim, out_dim=q, act="tanh"),
            X_tr,
            y_tr_2d,
            X_va,
            y_va_2d,
            X_te,
            seed=seed,
            wd_grid=[1e-2, 5e-2],
        )
        return {
            "pred_te": pred_te,
            "pred_va": pred_va,
            "n_params": n_par,
            "model_name": model_name,
            "ood_fn": ood_fn,
        }

    raise ValueError(f"Unknown classical model_name: {model_name!r}")


def run_synthetic_controls() -> dict[str, Any]:
    """Synthetic non-commutative manifold controls (Supplementary Material S1, Table S1, Fig. S1a):
      - Unsupervised circle S^1 (d = 2, N = 1200): non-commutative minimum E_0 = 1/4 vs.
        commuting two-centroid Voronoi minimum E_0 = 1/2 - 2/pi^2 approx 0.29736;
      - Supervised SO(3) Bloch map on S^2 (d = 2, q = 3, N = 1500): Pauli su(2) recovery vs.
        real-symmetric (F_{jk} = 0) and commuting diagonal operators;
      - Spin-1 quadrupole y = u_x u_y on S^2 (d = 3, N = 3000, 5 seeds): fixed-d separation vs.
        iso-parameter commuting diagonal parity at d = 9 ((3 + 1) * 9 = 36 parameters).
    """
    rng = np.random.default_rng(42)

    n_s1 = 1200
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n_s1)
    X_s1 = np.column_stack([np.cos(theta), np.sin(theta)])
    n_s1_tr = int(0.7 * n_s1)
    X_s1_tr, X_s1_te = X_s1[:n_s1_tr], X_s1[n_s1_tr:]

    s1_unsup: dict[str, Any] = {}
    for mode in ("hermitian", "real_sym", "commuting_diag"):
        m, _ = train_qcml_single(
            X_s1_tr,
            None,
            None,
            None,
            d=2,
            mode=mode,
            q=1,
            beta=None,
            lambda_e=1.0,
            lambda_reg_a=0.0,
            lambda_reg_b=0.0,
            candelori_w=None,
            unsupervised=True,
            lr=5e-2,
            batch_size=len(X_s1_tr),
            max_epochs=180,
            patience=20,
            seed=101,
        )
        st = m.export_state(np.zeros((1, 1)), np.ones((1, 1)), True)
        obs = extract_spectral_observables(st, X_s1_te)
        s1_unsup[mode] = {
            "test_mean_E0": round(float(np.mean(obs["E0"])), 5),
            "kappa_nc": round(float(m.commutator_diagnostics(p_mkt=2)["kappa_nc"]), 4),
            "hermiticity_err": float(m.hermiticity_error()),
        }

    n_s2 = 1500
    g = rng.normal(size=(n_s2, 3))
    X_s2 = g / np.linalg.norm(g, axis=1, keepdims=True)
    rot = np.array([[0.6, 0.8, 0.0], [-0.48, 0.36, 0.8], [0.64, -0.48, 0.6]], dtype=np.float64)
    y_s2 = X_s2 @ rot.T
    n_s2_tr, n_s2_va = int(0.6 * n_s2), int(0.2 * n_s2)
    X_s2_tr, y_s2_tr = X_s2[:n_s2_tr], y_s2[:n_s2_tr]
    X_s2_va, y_s2_va = X_s2[n_s2_tr : n_s2_tr + n_s2_va], y_s2[n_s2_tr : n_s2_tr + n_s2_va]
    X_s2_te, y_s2_te = X_s2[n_s2_tr + n_s2_va :], y_s2[n_s2_tr + n_s2_va :]

    s2_sup: dict[str, Any] = {}
    for label, mode, beta in [
        ("hermitian", "hermitian", None),
        ("real_sym", "real_sym", None),
        ("commuting_diag_hard", "commuting_diag", None),
        ("commuting_diag_gibbs", "commuting_diag", 2.0),
    ]:
        y_mu = y_s2_tr.mean(axis=0, keepdims=True)
        y_sd = y_s2_tr.std(axis=0, keepdims=True)
        m, _ = train_qcml_single(
            X_s2_tr,
            (y_s2_tr - y_mu) / y_sd,
            X_s2_va,
            (y_s2_va - y_mu) / y_sd,
            d=2,
            mode=mode,
            q=3,
            beta=beta,
            lambda_e=0.05,
            lambda_reg_a=0.0,
            lambda_reg_b=0.0,
            candelori_w=None,
            unsupervised=False,
            lr=4e-2,
            batch_size=128,
            max_epochs=100,
            patience=20,
            seed=101,
        )
        st = m.export_state(y_mu, y_sd, False)
        yp = extract_spectral_observables(st, X_s2_te)["pred"]
        r2 = 1.0 - float(np.sum((y_s2_te - yp) ** 2) / np.sum((y_s2_te - y_s2_te.mean(axis=0)) ** 2))
        np_tot, np_eff = m.parameter_counts()
        s2_sup[label] = {
            "test_r2": round(r2, 5),
            "kappa_nc": round(float(m.commutator_diagnostics(p_mkt=3)["kappa_nc"]), 4),
            "n_params": np_tot,
            "n_eff_params": np_eff,
        }

    seeds_quad = [101, 102, 103, 104, 105]
    n_q = 3000
    n_tr_q, n_va_q = int(0.6 * n_q), int(0.2 * n_q)
    b0b_keys = ("herm_d3", "realsym_d3", "diag_d3", "diag_d9", "ridge", "hgbdt")
    b0b_r2: dict[str, list[float]] = {k: [] for k in b0b_keys}
    b0b_params: dict[str, int] = {}

    for sd in seeds_quad:
        r_s = np.random.default_rng(sd)
        gq = r_s.normal(size=(n_q, 3))
        Xq = gq / np.linalg.norm(gq, axis=1, keepdims=True)
        yq = Xq[:, 0] * Xq[:, 1]
        Xtr, ytr = Xq[:n_tr_q], yq[:n_tr_q]
        Xva, yva = Xq[n_tr_q : n_tr_q + n_va_q], yq[n_tr_q : n_tr_q + n_va_q]
        Xte, yte = Xq[n_tr_q + n_va_q :], yq[n_tr_q + n_va_q :]
        denom = float(np.sum((yte - yte.mean()) ** 2))

        def _r2(p_arr: np.ndarray) -> float:
            return 1.0 - float(np.sum((yte - p_arr.ravel()) ** 2)) / max(denom, 1e-18)

        m_r = Ridge(alpha=1e-3).fit(Xtr, ytr)
        b0b_r2["ridge"].append(_r2(m_r.predict(Xte)))
        b0b_params["ridge"] = 4

        m_g = HistGradientBoostingRegressor(max_iter=120, random_state=sd).fit(Xtr, ytr)
        b0b_r2["hgbdt"].append(_r2(m_g.predict(Xte)))
        b0b_params["hgbdt"] = int(120 * 31)

        y_mu = np.array([[float(ytr.mean())]])
        y_sd = np.array([[float(ytr.std())]])
        ytr_s = ((ytr - y_mu[0, 0]) / y_sd[0, 0])[:, None]
        yva_s = ((yva - y_mu[0, 0]) / y_sd[0, 0])[:, None]

        qcml_specs = [
            ("herm_d3", "hermitian", 3, [None]),
            ("realsym_d3", "real_sym", 3, [None]),
            ("diag_d3", "commuting_diag", 3, [1.0, 2.0, 4.0]),
            ("diag_d9", "commuting_diag", 9, [1.0, 2.0, 4.0, 8.0]),
        ]
        for key, mode, dim, beta_grid in qcml_specs:
            best_v, best_m = float("inf"), None
            for b_cand in beta_grid:
                mc, vc = train_qcml_single(
                    Xtr,
                    ytr_s,
                    Xva,
                    yva_s,
                    d=dim,
                    mode=mode,
                    q=1,
                    beta=b_cand,
                    lambda_e=0.05,
                    lambda_reg_a=0.0,
                    lambda_reg_b=0.0,
                    candelori_w=None,
                    unsupervised=False,
                    lr=3e-2,
                    batch_size=256,
                    max_epochs=140,
                    patience=25,
                    seed=sd,
                )
                if vc < best_v or best_m is None:
                    best_v, best_m = vc, mc
            st = best_m.export_state(y_mu, y_sd, True)
            yp = extract_spectral_observables(st, Xte)["pred"]
            b0b_r2[key].append(_r2(yp))
            b0b_params[key] = best_m.parameter_counts()[0]

    b0b_summary = {
        k: {
            "mean_r2": round(float(np.mean(b0b_r2[k])), 5),
            "std_r2": round(float(np.std(b0b_r2[k], ddof=1)), 5),
            "r2_seeds": [round(float(x), 5) for x in b0b_r2[k]],
            "n_params": int(b0b_params[k]),
        }
        for k in b0b_keys
    }

    return {
        "b0a": {
            "s1_unsupervised": s1_unsup,
            "s1_theory_herm_min": 0.25,
            "s1_theory_diag_min": round(0.5 - 2.0 / (math.pi ** 2), 5),
            "s2_supervised_3way": s2_sup,
        },
        "b0b": b0b_summary,
    }
