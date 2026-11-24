"""Econometric inference, norm-controlled spectral-stress tests, and causal portfolio simulator
(Sec. 3.3, 4.1-4.3, Supplementary Material S3).

Implements pooled daily Spearman RankIC, four-factor Fama-MacBeth orthogonalized RankIC_{FM},
Campbell-Thompson out-of-sample R^2_{OOS} against expanding fold training means, Newey-West
Bartlett HAC covariance (L_{HAC} = 12), Harvey-Leybourne-Newbold Diebold-Mariano statistics,
Clark-West nested MSPE tests, Politis-Romano stationary block bootstrap (mean block b_bar = 10),
Holm-Bonferroni step-down adjustment, Hansen's 95% Model Confidence Set (T_{max}), Gaussian
QMLE GARCH(1,1), five-lag HAC Wald Granger causality controlling for (||x||_2, ||x||_2^2),
overlapping 1/5-tranche weekly portfolio simulation with one-day execution lag, Central Bank of
Russia short-ban masking, borrow fees, and Bailey-Lopez de Prado Deflated Sharpe Ratios.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm, rankdata, t as student_t
from sklearn.decomposition import PCA

from .panel import (
    DATA_SPLIT_SEED,
    FOLD_TEST_WINDOWS,
    L_HAC_DEFAULT,
    N_FOLDS,
    ensure_panels,
    get_fold_data,
    get_ood2022_data,
)


def nw_mean_stats(series: np.ndarray, l_hac: int = L_HAC_DEFAULT) -> dict[str, float]:
    """Sample mean, Newey-West Bartlett long-run variance LRV = gamma_0 + 2 sum_{ell=1}^L (1 - ell/(L+1)) gamma_ell,
    Student-t p-value, and autocorrelation-adjusted effective sample size T_{eff} = min(T, T gamma_0 / LRV).
    """
    z = np.asarray(series, dtype=np.float64)
    z = z[np.isfinite(z)]
    T = len(z)
    if T < 5:
        return {
            "mean": float("nan"), "nw_se": float("nan"), "nw_t": float("nan"),
            "nw_pval": float("nan"), "lrv": float("nan"), "gamma0": float("nan"),
            "t_eff": float(T), "n_days": int(T),
        }
    mu = float(z.mean())
    zc = z - mu
    gamma0 = float(np.dot(zc, zc) / T)
    L = min(int(l_hac), T - 1)
    lrv = gamma0
    for ell in range(1, L + 1):
        w = 1.0 - ell / float(L + 1)
        lrv += 2.0 * w * float(np.dot(zc[ell:], zc[:-ell]) / T)
    lrv_pos = max(lrv, 1e-18)
    se = math.sqrt(lrv_pos / T)
    t_stat = mu / se if se > 0 else 0.0
    pval = float(2.0 * student_t.sf(abs(t_stat), df=T - 1))
    t_eff = float(min(T, max(2.0, T * gamma0 / lrv_pos))) if gamma0 > 1e-18 else float(T)
    return {
        "mean": mu, "nw_se": se, "nw_t": t_stat, "nw_pval": pval,
        "lrv": lrv_pos, "gamma0": gamma0, "t_eff": t_eff, "n_days": int(T),
    }


def _stationary_bootstrap_indices(
    T: int, n_boot: int, mean_block: float = 10.0, seed: int = 202601
) -> np.ndarray:
    """Politis-Romano (1994) geometric-block resampling indices with restart probability p = 1 / b_bar."""
    rng = np.random.default_rng(seed)
    p_new = 1.0 / max(1.0, float(mean_block))
    starts = rng.integers(0, T, size=(n_boot, T))
    is_new = rng.random(size=(n_boot, T)) < p_new
    is_new[:, 0] = True
    idx = np.empty((n_boot, T), dtype=np.int32)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(is_new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def stationary_bootstrap_mean(
    series: np.ndarray,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    seed: int = 202601,
) -> dict[str, float]:
    z = np.asarray(series, dtype=np.float64)
    z = z[np.isfinite(z)]
    T = len(z)
    if T < 5:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "boot_pval": float("nan")}
    idx = _stationary_bootstrap_indices(T, n_boot, mean_block=mean_block, seed=seed)
    boot_means = z[idx].mean(axis=1)
    obs = float(z.mean())
    centered = boot_means - obs
    pval = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "boot_pval": pval,
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    k_trials: int = 40,
    l_hac: int = 4,
    sr_trials_var: float | None = None,
) -> dict[str, float]:
    """Bailey-Lopez de Prado (2014) Deflated Sharpe Ratio correcting for non-Gaussian skewness gamma_3,
    raw kurtosis gamma_4, serial correlation via T_{eff} <= T, and extreme-value threshold SR_0 across
    K_{trials} = 40 strategies (Sec. 3.3).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 10:
        return {"dsr": float("nan"), "sr_per_period": float("nan"), "sr0": float("nan"), "t_eff": float(T)}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return {"dsr": 0.0, "sr_per_period": 0.0, "sr0": 0.0, "t_eff": float(T)}
    sr = mu / sd
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt_raw = float(np.mean(z ** 4))

    nw = nw_mean_stats(r, l_hac=l_hac)
    t_eff = float(min(T, max(5.0, nw["t_eff"])))

    K = max(2, int(k_trials))
    euler_gamma = 0.5772156649015329
    var_sr = float(sr_trials_var) if sr_trials_var is not None else (1.0 / t_eff)
    sr0 = math.sqrt(max(var_sr, 1e-18)) * (
        (1.0 - euler_gamma) * float(norm.ppf(1.0 - 1.0 / K))
        + euler_gamma * float(norm.ppf(1.0 - 1.0 / (K * math.e)))
    )
    denom_sq = max(1e-8, 1.0 - skew * sr + ((kurt_raw - 1.0) / 4.0) * (sr ** 2))
    z_stat = (sr - sr0) * math.sqrt(max(1.0, t_eff - 1.0)) / math.sqrt(denom_sq)
    return {
        "dsr": float(norm.cdf(z_stat)),
        "sr_per_period": sr,
        "sr0": sr0,
        "t_eff": t_eff,
        "k_trials": int(K),
        "skew": skew,
        "kurt_raw": kurt_raw,
    }


def holm_bonferroni(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Holm (1979) step-down family-wise error rate adjusted p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p[i])))
        adj[i] = running
    return adj


def paired_tests(
    daily_a: np.ndarray,
    daily_b: np.ndarray,
    n_boot: int = 2000,
    l_hac: int = L_HAC_DEFAULT,
    horizon: int = 5,
    seed: int = 202601,
) -> dict[str, float]:
    """Paired differential d_t = a_t - b_t evaluated via Newey-West HAC, Harvey-Leybourne-Newbold
    finite-sample Diebold-Mariano factor sqrt((T + 1 - 2h + h(h-1)/T)/T), and stationary bootstrap.
    """
    a = np.asarray(daily_a, dtype=np.float64)
    b = np.asarray(daily_b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    T = len(d)
    if T < 10:
        return {
            "diff_mean": float("nan"), "nw_se": float("nan"), "nw_t": float("nan"),
            "nw_pval": float("nan"), "dm_hln_stat": float("nan"), "dm_hln_pval": float("nan"),
            "boot_ci_lo": float("nan"), "boot_ci_hi": float("nan"), "boot_pval": float("nan"),
            "n_days": int(T),
        }
    nw = nw_mean_stats(d, l_hac=l_hac)
    h = int(horizon)
    hln_factor = math.sqrt(max(1e-9, (T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T))
    dm_stat = nw["nw_t"] * hln_factor
    dm_pval = float(2.0 * student_t.sf(abs(dm_stat), df=T - 1))
    mean_blk = max(float(h), float(math.ceil(T ** (1.0 / 3.0))))
    boot = stationary_bootstrap_mean(d, n_boot=n_boot, mean_block=mean_blk, seed=seed)
    return {
        "diff_mean": nw["mean"],
        "nw_se": nw["nw_se"],
        "nw_t": nw["nw_t"],
        "nw_pval": nw["nw_pval"],
        "dm_hln_stat": float(dm_stat),
        "dm_hln_pval": dm_pval,
        "boot_ci_lo": boot["ci_lo"],
        "boot_ci_hi": boot["ci_hi"],
        "boot_pval": boot["boot_pval"],
        "n_days": int(T),
    }


def clark_west_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench: np.ndarray,
    di: np.ndarray,
    l_hac: int = L_HAC_DEFAULT,
) -> dict[str, float]:
    """Clark-West (2007) adjusted MSPE differential f_{i,t} = (y - y_b)^2 - (y - y_hat)^2 + (y_b - y_hat)^2."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yb = np.asarray(y_bench, dtype=np.float64)
    d = np.asarray(di)
    ok = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yb)
    f_row = (yt - yb) ** 2 - (yt - yp) ** 2 + (yb - yp) ** 2
    ud = np.unique(d[ok])
    f_day = np.array([float(f_row[ok & (d == u)].mean()) for u in ud], dtype=np.float64)
    nw = nw_mean_stats(f_day, l_hac=l_hac)
    cw_stat = nw["nw_t"]
    return {
        "cw_stat": float(cw_stat),
        "cw_pval_one_sided": float(norm.sf(cw_stat)) if np.isfinite(cw_stat) else float("nan"),
        "cw_pval_t_one_sided": float(student_t.sf(cw_stat, df=max(1, nw["n_days"] - 1)))
        if np.isfinite(cw_stat) else float("nan"),
        "n_days": int(nw["n_days"]),
    }


def hansen_mcs_05(
    loss_matrix_dict: dict[str, np.ndarray],
    n_boot: int = 1000,
    block: int = 10,
    alpha: float = 0.05,
    seed: int = 202601,
) -> dict[str, dict[str, Any]]:
    """Hansen-Lunde-Nason (2011) Model Confidence Set under T_{max} elimination with stationary block bootstrap."""
    names = list(loss_matrix_dict.keys())
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: {"mcs_pval": 1.0, "in_mcs_05": True}}

    L_raw = np.column_stack([np.asarray(loss_matrix_dict[k], dtype=np.float64) for k in names])
    L = L_raw[np.all(np.isfinite(L_raw), axis=1)]
    T, M = L.shape
    if T < 10:
        return {k: {"mcs_pval": 1.0, "in_mcs_05": True} for k in names}

    idxs = _stationary_bootstrap_indices(T, n_boot, mean_block=float(block), seed=seed)
    alive = list(range(M))
    pvals = np.ones(M, dtype=np.float64)
    pmax = 0.0

    boot_col_means = L[idxs].mean(axis=1)
    full_col_means = L.mean(axis=0)

    while len(alive) > 1:
        bm = boot_col_means[:, alive]
        fm = full_col_means[alive]
        dbar = fm - fm.mean()
        bd = bm - bm.mean(axis=1, keepdims=True)
        se = np.sqrt(np.maximum(bd.var(axis=0), 1e-18))
        t_i = dbar / se
        stat = float(t_i.max())
        bstat = ((bd - dbar[None, :]) / se[None, :]).max(axis=1)
        p = float(np.mean(bstat >= stat))
        pmax = max(pmax, p)
        worst = int(np.argmax(t_i))
        pvals[alive[worst]] = pmax
        alive.pop(worst)
    pvals[alive[0]] = 1.0

    return {
        names[j]: {"mcs_pval": round(float(pvals[j]), 6), "in_mcs_05": bool(pvals[j] >= alpha)}
        for j in range(M)
    }


def _daily_spearman_and_mse(
    pred: np.ndarray,
    y: np.ndarray,
    day_slices: list[np.ndarray],
    ortho_factors: np.ndarray | None = None,
    min_n: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_days = len(day_slices)
    ic = np.full(n_days, np.nan, dtype=np.float64)
    mse = np.full(n_days, np.nan, dtype=np.float64)
    orth_ic = np.full(n_days, np.nan, dtype=np.float64) if ortho_factors is not None else None

    for k, ix in enumerate(day_slices):
        if len(ix) < min_n:
            continue
        p = pred[ix]
        yy = y[ix]
        mse[k] = float(np.mean((yy - p) ** 2))
        if float(np.std(p)) < 1e-12 or float(np.std(yy)) < 1e-12:
            ic[k] = 0.0
            if orth_ic is not None:
                orth_ic[k] = 0.0
            continue
        rp = rankdata(p)
        ry = rankdata(yy)
        ic[k] = float(np.corrcoef(rp, ry)[0, 1])

        if orth_ic is not None and ortho_factors is not None:
            F = ortho_factors[ix]
            ok_f = np.all(np.isfinite(F), axis=1)
            if int(ok_f.sum()) < max(min_n, F.shape[1] + 3):
                continue
            rp_f = rp[ok_f]
            ry_f = ry[ok_f]
            R_cols = [np.ones(int(ok_f.sum()), dtype=np.float64)]
            for j in range(F.shape[1]):
                col = F[ok_f, j]
                if float(np.std(col)) > 1e-12:
                    R_cols.append(rankdata(col))
            A = np.column_stack(R_cols)
            beta, *_ = np.linalg.lstsq(A, rp_f, rcond=None)
            rp_res = rp_f - A @ beta
            orth_ic[k] = 0.0 if float(np.std(rp_res)) < 1e-12 else float(np.corrcoef(rp_res, ry_f)[0, 1])

    return ic, mse, orth_ic


def _point_eval_single(
    pred: np.ndarray,
    y: np.ndarray,
    tr_mean: np.ndarray,
    di: np.ndarray,
    uniq_days: np.ndarray,
    day_slices: list[np.ndarray],
    fold_bounds: list[tuple[int, int]],
    ortho_factors: np.ndarray | None,
    l_hac: int,
) -> dict[str, Any]:
    ic, mse, orth_ic = _daily_spearman_and_mse(pred, y, day_slices, ortho_factors=ortho_factors)
    nw = nw_mean_stats(ic, l_hac=l_hac)

    err = y - pred
    err_b = y - tr_mean
    sse = float(np.dot(err, err))
    sse_b = float(np.dot(err_b, err_b))
    r2_oos = 0.0 if abs(sse - sse_b) <= 1e-15 * max(sse_b, 1.0) else (1.0 - sse / max(sse_b, 1e-18))
    rmse = math.sqrt(sse / max(len(y), 1))
    mae = float(np.mean(np.abs(err)))

    q90 = float(np.quantile(np.abs(y), 0.90))
    tm = np.abs(y) >= q90
    tail10_rmse = float(math.sqrt(np.mean(err[tm] ** 2))) if np.any(tm) else float("nan")
    pinball_05 = float(np.mean(np.maximum(0.05 * err, (0.05 - 1.0) * err)))

    sd_p = float(np.std(pred))
    sd_y = float(np.std(y))
    corr = float(np.corrcoef(pred, y)[0, 1]) if (sd_p > 1e-12 and sd_y > 1e-12) else 0.0
    scale_ratio = sd_p / max(sd_y, 1e-12)

    fold_ics: list[float] = []
    for lo_d, hi_d in fold_bounds:
        m_f = (uniq_days >= lo_d) & (uniq_days <= hi_d) & np.isfinite(ic)
        fold_ics.append(float(ic[m_f].mean()) if np.any(m_f) else float("nan"))
    valid_f = [v for v in fold_ics if np.isfinite(v)]
    fold_win_rate = float(np.mean([v > 0.0 for v in valid_f])) if valid_f else 0.0

    orth_nw = (
        nw_mean_stats(orth_ic, l_hac=l_hac)
        if orth_ic is not None
        else {"mean": float("nan"), "nw_se": float("nan"), "nw_t": float("nan"), "nw_pval": float("nan")}
    )

    return {
        "rank_ic": nw["mean"],
        "nw_se": nw["nw_se"],
        "nw_t": nw["nw_t"],
        "nw_pval": nw["nw_pval"],
        "oos_r2": float(r2_oos),
        "rmse": float(rmse),
        "mae": mae,
        "tail10_rmse": tail10_rmse,
        "pinball_05": pinball_05,
        "pearson_corr": corr,
        "pred_std_ratio": scale_ratio,
        "orth_fm_ic": orth_nw["mean"],
        "orth_fm_nw_t": orth_nw["nw_t"],
        "orth_fm_nw_pval": orth_nw["nw_pval"],
        "fold_rank_ics": fold_ics,
        "fold_win_rate": fold_win_rate,
        "daily_ic": ic,
        "daily_mse": mse,
        "n_days": int(nw["n_days"]),
        "n_rows": int(len(y)),
    }


def _bootstrap_sharpe_ratio(
    cohort_returns: list[np.ndarray],
    ann_factor: float,
    n_boot: int = 2000,
    mean_block_weeks: float = 2.0,
    seed: int = 202601,
) -> dict[str, float]:
    min_len = min(len(c) for c in cohort_returns if len(c) > 0) if cohort_returns else 0
    if min_len < 10:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "boot_pval": float("nan")}
    mat = np.column_stack([c[:min_len] for c in cohort_returns])
    idx = _stationary_bootstrap_indices(min_len, n_boot, mean_block=mean_block_weeks, seed=seed)
    samp = mat[idx]
    means = samp.mean(axis=1)
    stds = samp.std(axis=1, ddof=1) + 1e-12
    boot_sr = (means / stds * ann_factor).mean(axis=1)

    obs_sr = float((mat.mean(axis=0) / (mat.std(axis=0, ddof=1) + 1e-12) * ann_factor).mean())
    centered = boot_sr - obs_sr
    pval = float(np.mean(np.abs(centered) >= abs(obs_sr)))
    return {
        "ci_lo": float(np.percentile(boot_sr, 2.5)),
        "ci_hi": float(np.percentile(boot_sr, 97.5)),
        "boot_pval": pval,
    }


def evaluate_economics(
    pred: np.ndarray,
    di: np.ndarray,
    si: np.ndarray,
    exec_lag: int = 1,
    horizon: int = 5,
    costs_bps: tuple[int, ...] = (0, 5, 10, 15, 25, 50),
    q: float = 0.20,
    top_n_ls: int = 30,
    borrow_bps_pa: float = 500.0,
    k_trials_dsr: int = 40,
    n_boot: int = 2000,
    seed_boot: int = 202601,
) -> dict[str, Any]:
    """Five-cohort overlapping 1/5-tranche weekly portfolio simulator executing close-t signals at
    close t + exec_lag, drifting weights over (t + 1, t + 1 + h], masking dollar-neutral long-short
    positions during the 2022-02-24..2022-05-31 CBR short ban, and deducting flat/tiered transaction
    costs and borrow fees (Sec. 3.3, 4.3).
    """
    _, panel = ensure_panels()
    logc = panel["log_close"]
    top30_mat = panel["top30_mask"]
    adv63_mat = panel["adv63_med"]
    short_ban = panel["short_ban_mask"]
    rf_daily = panel["cbr_key_rate_daily"]
    T, N = logc.shape

    uniq_days = np.unique(di)
    day_slices = [np.flatnonzero(di == u) for u in uniq_days]
    n_u = len(uniq_days)

    ann_per_yr = 250.0 / float(horizon)
    ann_sqrt = math.sqrt(ann_per_yr)
    borrow_per_period = (borrow_bps_pa / 1e4) * (float(horizon) / 250.0)

    cohorts: list[dict[str, np.ndarray]] = []
    for c0 in range(horizon):
        prev_lo = np.zeros(N, dtype=np.float64)
        prev_ls = np.zeros(N, dtype=np.float64)
        lo_act_g, lo_raw_g, lo_rf, ls_g = [], [], [], []
        nt_lo, nt_ls, tc_tier_lo, tc_tier_ls = [], [], [], []
        ban_flags, dts = [], []

        for k in range(c0, n_u, horizon):
            t = int(uniq_days[k])
            te = t + exec_lag
            tx = te + horizon
            if tx >= T:
                continue

            ix = day_slices[k]
            s_i = si[ix]
            p_i = pred[ix]
            ret_i = np.exp(logc[tx, s_i] - logc[te, s_i]) - 1.0
            ok = np.isfinite(p_i) & np.isfinite(ret_i)
            if int(ok.sum()) < 15:
                continue

            p_ok = p_i[ok]
            r_ok = ret_i[ok]
            s_ok = s_i[ok]
            n_ok = len(p_ok)

            tier_bps_vec = np.where(top30_mat[t], 10.0, 30.0) * 1e-4
            rf_5d = float(np.prod(1.0 + rf_daily[te + 1 : tx + 1]) - 1.0)

            k5 = max(1, int(round(q * n_ok)))
            ord_p = np.argsort(-p_ok, kind="mergesort")
            w_lo = np.zeros(N, dtype=np.float64)
            w_lo[s_ok[ord_p[:k5]]] = 1.0 / k5
            dw_lo = np.abs(w_lo - prev_lo)
            lo_raw = float(r_ok[ord_p[:k5]].mean())
            ew_ret = float(r_ok.mean())

            lo_act_g.append(lo_raw - ew_ret)
            lo_raw_g.append(lo_raw)
            lo_rf.append(rf_5d)
            nt_lo.append(float(dw_lo.sum()))
            tc_tier_lo.append(float(np.dot(dw_lo, tier_bps_vec)))

            w_lo_drifted = np.zeros(N, dtype=np.float64)
            w_lo_drifted[s_ok[ord_p[:k5]]] = (1.0 / k5) * (1.0 + r_ok[ord_p[:k5]])
            tot_w = float(w_lo_drifted.sum())
            prev_lo = w_lo_drifted / tot_w if tot_w > 1e-9 else w_lo

            is_banned = bool(short_ban[te])
            ban_flags.append(is_banned)
            if is_banned:
                w_ls = np.zeros(N, dtype=np.float64)
                dw_ls = np.abs(w_ls - prev_ls)
                ls_g.append(0.0)
                nt_ls.append(float(dw_ls.sum()))
                tc_tier_ls.append(float(np.dot(dw_ls, tier_bps_vec)))
                prev_ls = w_ls
            else:
                adv_ok = adv63_mat[t, s_ok]
                t30_idx = np.argsort(-adv_ok, kind="mergesort")[: min(top_n_ls, n_ok)]
                p3, r3, s3 = p_ok[t30_idx], r_ok[t30_idx], s_ok[t30_idx]
                k3 = max(1, int(round(q * len(t30_idx))))
                ord3 = np.argsort(-p3, kind="mergesort")
                w_ls = np.zeros(N, dtype=np.float64)
                w_ls[s3[ord3[:k3]]] = 1.0 / k3
                w_ls[s3[ord3[-k3:]]] -= 1.0 / k3
                dw_ls = np.abs(w_ls - prev_ls)
                ls_g.append(float(r3[ord3[:k3]].mean() - r3[ord3[-k3:]].mean()))
                nt_ls.append(float(dw_ls.sum()))
                tc_tier_ls.append(float(np.dot(dw_ls, tier_bps_vec)))
                prev_ls = w_ls

            dts.append(t)

        cohorts.append(
            {
                "lo_act_gross": np.asarray(lo_act_g, dtype=np.float64),
                "lo_raw_gross": np.asarray(lo_raw_g, dtype=np.float64),
                "lo_rf": np.asarray(lo_rf, dtype=np.float64),
                "ls_gross": np.asarray(ls_g, dtype=np.float64),
                "nt_lo": np.asarray(nt_lo, dtype=np.float64),
                "nt_ls": np.asarray(nt_ls, dtype=np.float64),
                "tc_tier_lo": np.asarray(tc_tier_lo, dtype=np.float64),
                "tc_tier_ls": np.asarray(tc_tier_ls, dtype=np.float64),
                "ban": np.asarray(ban_flags, dtype=bool),
                "dates": np.asarray(dts, dtype=np.int32),
            }
        )

    def _cohort_metrics(cost_bps: float | None, borrow_p: float) -> dict[str, Any]:
        ir_lo_list, sr_lo_ex_list, sr_ls_list = [], [], []
        lo_net_ser, ls_net_ser = [], []
        for ch in cohorts:
            tc_lo = ch["tc_tier_lo"] if cost_bps is None else (cost_bps * 1e-4 * ch["nt_lo"])
            tc_ls = ch["tc_tier_ls"] if cost_bps is None else (cost_bps * 1e-4 * ch["nt_ls"])
            net_lo_act = ch["lo_act_gross"] - tc_lo
            net_lo_ex = ch["lo_raw_gross"] - ch["lo_rf"] - tc_lo
            bor_vec = np.where(ch["ban"], 0.0, borrow_p)
            net_ls = ch["ls_gross"] - tc_ls - bor_vec
            lo_net_ser.append(net_lo_act)
            ls_net_ser.append(net_ls)
            ir_lo_list.append(float(net_lo_act.mean() / (net_lo_act.std(ddof=1) + 1e-12) * ann_sqrt))
            sr_lo_ex_list.append(float(net_lo_ex.mean() / (net_lo_ex.std(ddof=1) + 1e-12) * ann_sqrt))
            sr_ls_list.append(float(net_ls.mean() / (net_ls.std(ddof=1) + 1e-12) * ann_sqrt))
        return {
            "ir_lo_active": float(np.mean(ir_lo_list)),
            "sharpe_lo_ex_cbr": float(np.mean(sr_lo_ex_list)),
            "sharpe_ls": float(np.mean(sr_ls_list)),
            "lo_net_cohorts": lo_net_ser,
            "ls_net_cohorts": ls_net_ser,
        }

    cost_sweep: dict[str, dict[str, float]] = {}
    lo_net15_cohorts: list[np.ndarray] = []
    ls_net15_cohorts: list[np.ndarray] = []
    for c in costs_bps:
        cm = _cohort_metrics(float(c), borrow_per_period)
        cost_sweep[str(c)] = {
            "ir_lo_active": round(cm["ir_lo_active"], 6),
            "sharpe_lo_ex_cbr": round(cm["sharpe_lo_ex_cbr"], 6),
            "sharpe_ls": round(cm["sharpe_ls"], 6),
        }
        if c == 15:
            lo_net15_cohorts = cm["lo_net_cohorts"]
            ls_net15_cohorts = cm["ls_net_cohorts"]

    tier_cm = _cohort_metrics(None, borrow_per_period)
    cost_sweep["tiered_10_30"] = {
        "ir_lo_active": round(tier_cm["ir_lo_active"], 6),
        "sharpe_lo_ex_cbr": round(tier_cm["sharpe_lo_ex_cbr"], 6),
        "sharpe_ls": round(tier_cm["sharpe_ls"], 6),
    }

    borrow_sens = {}
    for b_bps in (0, 500, 1200):
        bp = (b_bps / 1e4) * (float(horizon) / 250.0)
        borrow_sens[str(b_bps)] = round(_cohort_metrics(15.0, bp)["sharpe_ls"], 6)

    mean_nt_lo = float(np.mean([ch["nt_lo"].mean() for ch in cohorts]))
    mean_nt_ls = float(np.mean([ch["nt_ls"][~ch["ban"]].mean() for ch in cohorts if np.any(~ch["ban"])]))
    mean_g_lo = float(np.mean([ch["lo_act_gross"].mean() for ch in cohorts]))
    mean_g_ls = float(
        np.mean(
            [(ch["ls_gross"] - np.where(ch["ban"], 0.0, borrow_per_period)).mean() for ch in cohorts]
        )
    )
    mean_nt_ls_all = float(np.mean([ch["nt_ls"].mean() for ch in cohorts]))

    be_lo = float(mean_g_lo / max(mean_nt_lo, 1e-9) * 1e4)
    be_ls = float(mean_g_ls / max(mean_nt_ls_all, 1e-9) * 1e4)

    lo_ci = _bootstrap_sharpe_ratio(lo_net15_cohorts, ann_sqrt, n_boot=n_boot, mean_block_weeks=2.0, seed=seed_boot)
    ls_ci = _bootstrap_sharpe_ratio(ls_net15_cohorts, ann_sqrt, n_boot=n_boot, mean_block_weeks=2.0, seed=seed_boot + 1)

    min_w = min(len(c) for c in lo_net15_cohorts)
    tranche_lo_net15 = np.column_stack([c[:min_w] for c in lo_net15_cohorts]).mean(axis=1)
    tranche_ls_net15 = np.column_stack([c[:min_w] for c in ls_net15_cohorts]).mean(axis=1)

    dsr_lo = deflated_sharpe_ratio(tranche_lo_net15, k_trials=k_trials_dsr, l_hac=4)
    dsr_ls = deflated_sharpe_ratio(tranche_ls_net15, k_trials=k_trials_dsr, l_hac=4)

    n_ban_rebal = int(sum(int(ch["ban"].sum()) for ch in cohorts))
    cbr_mean_pct = float(panel["cbr_key_rate_pct"][uniq_days].mean())

    return {
        "exec_lag": int(exec_lag),
        "borrow_bps_pa": float(borrow_bps_pa),
        "cbr_rf_mean_pct": round(cbr_mean_pct, 4),
        "n_short_ban_rebalances": n_ban_rebal,
        "cost_sweep": cost_sweep,
        "ls_borrow_sensitivity_15bps": borrow_sens,
        "gross_ir_lo_active": cost_sweep["0"]["ir_lo_active"],
        "net_ir_lo_active_15bps": cost_sweep["15"]["ir_lo_active"],
        "net_ir_lo_active_15bps_ci": [round(lo_ci["ci_lo"], 5), round(lo_ci["ci_hi"], 5)],
        "net_ir_lo_active_15bps_boot_pval": round(lo_ci["boot_pval"], 5),
        "net_ir_lo_active_tiered": cost_sweep["tiered_10_30"]["ir_lo_active"],
        "net_sharpe_lo_ex_cbr_15bps": cost_sweep["15"]["sharpe_lo_ex_cbr"],
        "gross_sharpe_ls": cost_sweep["0"]["sharpe_ls"],
        "net_sharpe_ls_15bps": cost_sweep["15"]["sharpe_ls"],
        "net_sharpe_ls_15bps_ci": [round(ls_ci["ci_lo"], 5), round(ls_ci["ci_hi"], 5)],
        "net_sharpe_ls_15bps_boot_pval": round(ls_ci["boot_pval"], 5),
        "net_sharpe_ls_tiered": cost_sweep["tiered_10_30"]["sharpe_ls"],
        "breakeven_cost_lo_bps": round(be_lo, 3),
        "breakeven_cost_ls_bps": round(be_ls, 3),
        "annual_turnover_lo_oneway": round(mean_nt_lo * 0.5 * ann_per_yr, 3),
        "annual_turnover_ls_oneway": round(mean_nt_ls * 0.5 * ann_per_yr, 3),
        "dsr_lo_active_15bps": round(dsr_lo["dsr"], 5),
        "dsr_ls_15bps": round(dsr_ls["dsr"], 5),
        "dsr_t_eff_lo": round(dsr_lo["t_eff"], 2),
        "dsr_k_trials": int(k_trials_dsr),
        "weekly_cohort_lo_net15": [c.astype(np.float64) for c in lo_net15_cohorts],
        "weekly_cohort_ls_net15": [c.astype(np.float64) for c in ls_net15_cohorts],
        "weekly_cohort_dates": [ch["dates"] for ch in cohorts],
    }


def evaluate_predictions(
    preds_seeds: np.ndarray,
    y_true: np.ndarray,
    di: np.ndarray,
    si: np.ndarray,
    tr_mean_per_row: np.ndarray,
    ortho_factors: np.ndarray | None = None,
    k_trials_dsr: int = 40,
    n_boot: int = 2000,
    l_hac: int = L_HAC_DEFAULT,
    horizon: int = 5,
    exec_lag: int = 1,
    compute_econ: bool = True,
    seed_boot: int = 202601,
) -> dict[str, Any]:
    """Single-seed (mean +/- sigma_{seed}) and cross-seed ensemble evaluation across all out-of-sample
    stock-days, benchmarking R^2_{OOS} against each fold's expanding training mean y_bar_{tr, f(t)}.
    """
    P = np.asarray(preds_seeds, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    S, N = P.shape
    yt = np.asarray(y_true, dtype=np.float64)
    trm = np.asarray(tr_mean_per_row, dtype=np.float64)
    d = np.asarray(di, dtype=np.int32)
    s_arr = np.asarray(si, dtype=np.int32)
    if trm.ndim == 0:
        trm = np.full(N, float(trm), dtype=np.float64)

    ok = np.isfinite(yt) & np.isfinite(trm) & np.all(np.isfinite(P), axis=0)
    P_ok = P[:, ok]
    y_ok = yt[ok]
    trm_ok = trm[ok]
    d_ok = d[ok]
    s_ok = s_arr[ok]
    ortho_ok = np.asarray(ortho_factors, dtype=np.float64)[ok] if ortho_factors is not None else None

    uniq_days = np.unique(d_ok)
    day_slices = [np.flatnonzero(d_ok == u) for u in uniq_days]

    _, panel = ensure_panels()
    ad = panel["active_dates"]
    fold_bounds: list[tuple[int, int]] = []
    for s_iso, e_iso, _ in FOLD_TEST_WINDOWS:
        ix_f = np.flatnonzero((ad >= s_iso) & (ad <= e_iso))
        fold_bounds.append((int(ix_f.min()), int(ix_f.max())))

    seed_res = [
        _point_eval_single(
            P_ok[j], y_ok, trm_ok, d_ok, uniq_days, day_slices, fold_bounds, ortho_ok, l_hac
        )
        for j in range(S)
    ]
    ens_pred = P_ok.mean(axis=0)
    ens_res = _point_eval_single(
        ens_pred, y_ok, trm_ok, d_ok, uniq_days, day_slices, fold_bounds, ortho_ok, l_hac
    )

    def _agg(key: str) -> tuple[float, float]:
        vals = np.array([sr[key] for sr in seed_res], dtype=np.float64)
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if S > 1 else 0.0)

    ic_s_mu, ic_s_sd = _agg("rank_ic")
    r2_s_mu, r2_s_sd = _agg("oos_r2")
    rmse_s_mu, rmse_s_sd = _agg("rmse")
    t10_s_mu, t10_s_sd = _agg("tail10_rmse")
    pb_s_mu, pb_s_sd = _agg("pinball_05")
    fm_s_mu, fm_s_sd = _agg("orth_fm_ic")
    corr_s_mu, _ = _agg("pearson_corr")
    scale_s_mu, _ = _agg("pred_std_ratio")

    daily_ic_single_mean = np.nanmean(np.stack([sr["daily_ic"] for sr in seed_res], axis=0), axis=0)
    daily_mse_single_mean = np.nanmean(np.stack([sr["daily_mse"] for sr in seed_res], axis=0), axis=0)
    nw_single_pooled = nw_mean_stats(daily_ic_single_mean, l_hac=l_hac)

    fold_ics_single = [
        float(np.nanmean([sr["fold_rank_ics"][f] for sr in seed_res])) for f in range(N_FOLDS)
    ]
    cw_ens = clark_west_test(y_ok, ens_pred, trm_ok, d_ok, l_hac=l_hac)

    out: dict[str, Any] = {
        "n_seeds": int(S),
        "n_oos_rows": int(len(y_ok)),
        "n_oos_days": int(len(uniq_days)),
        "single_seed": {
            "rank_ic_mean": round(ic_s_mu, 6),
            "rank_ic_seed_std": round(ic_s_sd, 6),
            "rank_ic_nw_se": round(nw_single_pooled["nw_se"], 6),
            "rank_ic_nw_t": round(nw_single_pooled["nw_t"], 4),
            "rank_ic_nw_pval": round(nw_single_pooled["nw_pval"], 6),
            "oos_r2_mean": round(r2_s_mu, 6),
            "oos_r2_seed_std": round(r2_s_sd, 6),
            "rmse_mean": round(rmse_s_mu, 6),
            "rmse_seed_std": round(rmse_s_sd, 6),
            "tail10_rmse_mean": round(t10_s_mu, 6),
            "tail10_rmse_seed_std": round(t10_s_sd, 6),
            "pinball_05_mean": round(pb_s_mu, 6),
            "pinball_05_seed_std": round(pb_s_sd, 6),
            "orth_fm_ic_mean": round(fm_s_mu, 6),
            "orth_fm_ic_seed_std": round(fm_s_sd, 6),
            "pearson_corr_mean": round(corr_s_mu, 6),
            "pred_std_ratio_mean": round(scale_s_mu, 6),
            "fold_rank_ics": [round(x, 5) for x in fold_ics_single],
            "fold_win_rate": round(float(np.mean([sr["fold_win_rate"] for sr in seed_res])), 4),
        },
        "ensemble": {
            "rank_ic": round(ens_res["rank_ic"], 6),
            "rank_ic_nw_se": round(ens_res["nw_se"], 6),
            "rank_ic_nw_t": round(ens_res["nw_t"], 4),
            "rank_ic_nw_pval": round(ens_res["nw_pval"], 6),
            "oos_r2": round(ens_res["oos_r2"], 6),
            "rmse": round(ens_res["rmse"], 6),
            "mae": round(ens_res["mae"], 6),
            "tail10_rmse": round(ens_res["tail10_rmse"], 6),
            "pinball_05": round(ens_res["pinball_05"], 6),
            "orth_fm_ic": round(ens_res["orth_fm_ic"], 6),
            "orth_fm_nw_t": round(ens_res["orth_fm_nw_t"], 4),
            "orth_fm_nw_pval": round(ens_res["orth_fm_nw_pval"], 6),
            "pearson_corr": round(ens_res["pearson_corr"], 6),
            "pred_std_ratio": round(ens_res["pred_std_ratio"], 6),
            "fold_rank_ics": [round(x, 5) for x in ens_res["fold_rank_ics"]],
            "fold_win_rate": round(ens_res["fold_win_rate"], 4),
            "clark_west_stat": round(cw_ens["cw_stat"], 4),
            "clark_west_pval": round(cw_ens["cw_pval_one_sided"], 6),
        },
        "oos_rank_ic_single_mean": round(ic_s_mu, 6),
        "oos_rank_ic_seed_std": round(ic_s_sd, 6),
        "oos_rank_ic_ens": round(ens_res["rank_ic"], 6),
        "oos_rank_ic_nw_se": round(ens_res["nw_se"], 6),
        "oos_rank_ic_nw_t": round(ens_res["nw_t"], 4),
        "oos_rank_ic_nw_pval": round(ens_res["nw_pval"], 6),
        "oos_r2": round(r2_s_mu, 6),
        "oos_r2_ens": round(ens_res["oos_r2"], 6),
        "oos_rmse": round(rmse_s_mu, 6),
        "oos_tail10_rmse": round(t10_s_mu, 6),
        "tail_pinball_05": round(pb_s_mu, 6),
        "orth_fm_ic": round(fm_s_mu, 6),
        "orth_fm_ic_ens": round(ens_res["orth_fm_ic"], 6),
        "daily_ic": ens_res["daily_ic"],
        "daily_mse": ens_res["daily_mse"],
        "daily_ic_single_mean": daily_ic_single_mean,
        "daily_mse_single_mean": daily_mse_single_mean,
        "uniq_days": uniq_days,
    }

    if compute_econ:
        econ_ens = evaluate_economics(
            ens_pred, d_ok, s_ok, exec_lag=exec_lag, horizon=horizon,
            k_trials_dsr=k_trials_dsr, n_boot=n_boot, seed_boot=seed_boot,
        )
        if S > 1:
            s_ir, s_ls, s_be = [], [], []
            for j in range(S):
                ej = evaluate_economics(
                    P_ok[j], d_ok, s_ok, exec_lag=exec_lag, horizon=horizon,
                    k_trials_dsr=k_trials_dsr, n_boot=100, seed_boot=seed_boot + j,
                )
                s_ir.append(ej["net_ir_lo_active_15bps"])
                s_ls.append(ej["net_sharpe_ls_15bps"])
                s_be.append(ej["breakeven_cost_lo_bps"])
            econ_ens["single_seed_net_ir_lo_15bps_mean"] = round(float(np.mean(s_ir)), 6)
            econ_ens["single_seed_net_ir_lo_15bps_std"] = round(float(np.std(s_ir, ddof=1)), 6)
            econ_ens["single_seed_net_sharpe_ls_15bps_mean"] = round(float(np.mean(s_ls)), 6)
            econ_ens["single_seed_breakeven_lo_bps_mean"] = round(float(np.mean(s_be)), 3)
        else:
            econ_ens["single_seed_net_ir_lo_15bps_mean"] = econ_ens["net_ir_lo_active_15bps"]
            econ_ens["single_seed_net_ir_lo_15bps_std"] = 0.0
            econ_ens["single_seed_net_sharpe_ls_15bps_mean"] = econ_ens["net_sharpe_ls_15bps"]
            econ_ens["single_seed_breakeven_lo_bps_mean"] = econ_ens["breakeven_cost_lo_bps"]
        out["economics"] = econ_ens

    return out


def _auroc_rank(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney rank statistic U / (n_1 n_0) equal to the empirical area under the ROC curve."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 < 3 or n0 < 3:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))


def _auroc_boot_ci(
    score: np.ndarray, label: np.ndarray, n_boot: int = 1000, block: int = 10, seed: int = 202601
) -> list[float]:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    T = len(s)
    if T < 10:
        return [float("nan"), float("nan")]
    idxs = _stationary_bootstrap_indices(T, n_boot, mean_block=float(block), seed=seed)
    vals = [_auroc_rank(s[idxs[b]], y[idxs[b]]) for b in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)]


def _fit_causal_garch11(returns_pct: np.ndarray) -> tuple[float, float, float, float]:
    """Gaussian quasi-maximum-likelihood GARCH(1,1) h_t = omega + alpha r_{t-1}^2 + beta h_{t-1}
    under stationarity constraint alpha + beta < 0.995 (Sec. 4.2).
    """
    r = np.asarray(returns_pct, dtype=np.float64)
    r = r[np.isfinite(r)]
    r2 = r ** 2
    v0 = max(float(np.var(r)), 1e-4)

    def nll(theta: np.ndarray) -> float:
        omega = math.exp(float(theta[0]))
        ea, eb = math.exp(float(theta[1])), math.exp(float(theta[2]))
        denom = 1.0 + ea + eb
        alpha = 0.995 * ea / denom
        beta = 0.995 * eb / denom
        h = v0
        ll = 0.0
        for x2 in r2:
            ll += math.log(h) + x2 / h
            h = max(1e-8, omega + alpha * x2 + beta * h)
        return 0.5 * ll

    init = np.array([math.log(max(0.05 * v0, 1e-6)), -1.5, 1.5], dtype=np.float64)
    res = minimize(nll, init, method="L-BFGS-B", bounds=[(-14.0, 4.0), (-8.0, 6.0), (-8.0, 6.0)])
    omega = math.exp(float(res.x[0]))
    ea, eb = math.exp(float(res.x[1])), math.exp(float(res.x[2]))
    denom = 1.0 + ea + eb
    alpha = 0.995 * ea / denom
    beta = 0.995 * eb / denom
    h = v0
    for x2 in r2[:-1]:
        h = max(1e-8, omega + alpha * x2 + beta * h)
    return omega, alpha, beta, h


def _wald_hac_granger(y: np.ndarray, Xr: np.ndarray, Xu: np.ndarray, l_hac: int = 5) -> tuple[float, float]:
    """Asymptotic chi^2_k Newey-West HAC Wald statistic W_{Granger} = b_u^T V_{uu}^{-1} b_u for H_0: b_u = 0."""
    n = len(y)
    A = np.column_stack([np.ones(n, dtype=np.float64), Xr, Xu])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    u = y - A @ b
    Au = A * u[:, None]
    S = (Au.T @ Au) / n
    for ell in range(1, min(l_hac, n - 1) + 1):
        G = (Au[ell:].T @ Au[: n - ell]) / n
        w = 1.0 - ell / float(l_hac + 1)
        S += w * (G + G.T)
    ATA_inv = np.linalg.pinv((A.T @ A) / n)
    V = (ATA_inv @ S @ ATA_inv) / n
    k = Xu.shape[1]
    idx = np.arange(A.shape[1] - k, A.shape[1])
    V_sub = V[np.ix_(idx, idx)] + 1e-12 * np.eye(k)
    stat = float(b[idx] @ np.linalg.solve(V_sub, b[idx]))
    return stat, float(chi2.sf(max(0.0, stat), df=k))


def _daily_gue_null_759(
    X_test: np.ndarray,
    day_slices: list[np.ndarray],
    d: int = 8,
    n_draws: int = 10,
    seed: int = 1000,
) -> dict[str, Any]:
    """Untrained GUE(d) random-matrix null across the 759 out-of-sample daily means, quantifying
    Weyl radial-norm collinearity R^2_{OLS}(E_0 ~ ||x||_2^2) and R^2_{OLS}(Delta E ~ [1, ||x||_2, ||x||_2^2])
    (Sec. 2.2, Table 3).
    """
    p = X_test.shape[1]
    n_days = len(day_slices)
    n2_row = np.sum(X_test.astype(np.float64) ** 2, axis=1)
    n1_row = np.sqrt(n2_row)
    n2_day = np.array([float(n2_row[ix].mean()) for ix in day_slices], dtype=np.float64)
    n1_day = np.array([float(n1_row[ix].mean()) for ix in day_slices], dtype=np.float64)

    A_lin = np.column_stack([np.ones(n_days), n2_day])
    A_poly = np.column_stack([np.ones(n_days), n1_day, n2_day])
    X64 = X_test.astype(np.float64)

    kappas, r2_de_n2, r2_de_poly, r2_e0_n2 = [], [], [], []
    for s in range(n_draws):
        rng = np.random.default_rng(seed + s)
        R = rng.normal(scale=1.0 / math.sqrt(2.0 * d), size=(p, d, d))
        I = rng.normal(scale=1.0 / math.sqrt(2.0 * d), size=(p, d, d))
        A_ops = 0.5 * (R + np.swapaxes(R, -1, -2)) + 1j * 0.5 * (I - np.swapaxes(I, -1, -2))
        norms = [float(np.linalg.norm(A_ops[j], ord="fro")) for j in range(p)]
        num, den = 0.0, 1e-12
        for j in range(p):
            for k in range(j + 1, p):
                comm = A_ops[j] @ A_ops[k] - A_ops[k] @ A_ops[j]
                num += float(np.linalg.norm(comm, ord="fro"))
                den += norms[j] * norms[k]
        kappas.append(float(num / den))

        C0 = 0.5 * np.sum(A_ops @ A_ops, axis=0)
        e0_row = np.empty(len(X64), dtype=np.float64)
        de_row = np.empty(len(X64), dtype=np.float64)
        bs = 4096
        for i0 in range(0, len(X64), bs):
            xb = X64[i0 : i0 + bs]
            H = C0[None, :, :] - np.tensordot(xb, A_ops, axes=([1], [0]))
            w = np.linalg.eigvalsh(H)
            e0_row[i0 : i0 + bs] = w[:, 0] + 0.5 * n2_row[i0 : i0 + bs]
            de_row[i0 : i0 + bs] = w[:, 1] - w[:, 0]

        e0_d = np.array([float(e0_row[ix].mean()) for ix in day_slices], dtype=np.float64)
        de_d = np.array([float(de_row[ix].mean()) for ix in day_slices], dtype=np.float64)

        res_de2 = de_d - A_lin @ np.linalg.lstsq(A_lin, de_d, rcond=None)[0]
        res_dep = de_d - A_poly @ np.linalg.lstsq(A_poly, de_d, rcond=None)[0]
        res_e02 = e0_d - A_lin @ np.linalg.lstsq(A_lin, e0_d, rcond=None)[0]
        r2_de_n2.append(float(1.0 - np.var(res_de2) / max(np.var(de_d), 1e-18)))
        r2_de_poly.append(float(1.0 - np.var(res_dep) / max(np.var(de_d), 1e-18)))
        r2_e0_n2.append(float(1.0 - np.var(res_e02) / max(np.var(e0_d), 1e-18)))

    return {
        "n_draws": int(n_draws),
        "n_days": int(n_days),
        "hilbert_dim": int(d),
        "kappa_nc_gue_mean": round(float(np.mean(kappas)), 6),
        "kappa_nc_gue_std": round(float(np.std(kappas, ddof=1)) if n_draws > 1 else 0.0, 6),
        "r2_dE_on_norm2_daily_mean": round(float(np.mean(r2_de_n2)), 6),
        "r2_dE_on_norm2_daily_std": round(float(np.std(r2_de_n2, ddof=1)) if n_draws > 1 else 0.0, 6),
        "r2_dE_on_norm_poly_daily_mean": round(float(np.mean(r2_de_poly)), 6),
        "r2_E0_on_norm2_daily_mean": round(float(np.mean(r2_e0_n2)), 6),
    }


def _fold_causal_norm_residual(
    score_day: np.ndarray,
    n1_day: np.ndarray,
    n2_day: np.ndarray,
    fold_day: np.ndarray,
    train_day_triples: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Fold-causal quadratic projection residual s_t^{perp} = s_t - [1, ||x||_2, ||x||_2^2] beta_{tr, f(t)}."""
    out = np.full(len(score_day), np.nan, dtype=np.float64)
    A_all = np.column_stack([np.ones_like(n1_day), n1_day, n2_day])
    for f in range(N_FOLDS):
        m = fold_day == f
        if not np.any(m):
            continue
        if train_day_triples is not None and f in train_day_triples:
            s_tr, n1_tr, n2_tr = train_day_triples[f]
            A_tr = np.column_stack([np.ones_like(n1_tr), n1_tr, n2_tr])
            beta, *_ = np.linalg.lstsq(A_tr, s_tr, rcond=None)
            out[m] = score_day[m] - A_all[m] @ beta
        elif f > 0:
            prev = fold_day < f
            beta, *_ = np.linalg.lstsq(A_all[prev], score_day[prev], rcond=None)
            out[m] = score_day[m] - A_all[m] @ beta
        else:
            idx0 = np.flatnonzero(m)
            for j, pos in enumerate(idx0):
                if j < 10:
                    out[pos] = np.nan
                    continue
                hist = idx0[:j]
                Ah = A_all[hist]
                yh = score_day[hist]
                reg = np.diag([0.0, 1e-4, 1e-4])
                beta = np.linalg.solve(Ah.T @ Ah + reg, Ah.T @ yh)
                out[pos] = score_day[pos] - float(A_all[pos] @ beta)
    return out


def evaluate_spectral_stress(
    spectral_by_model: dict[str, dict[str, np.ndarray]],
    X_test_concat: np.ndarray,
    di_test_concat: np.ndarray,
    fold_test_concat: np.ndarray,
    modality: str = "market_news",
    train_spectral_by_model: dict[str, dict[int, dict[str, np.ndarray]]] | None = None,
    gue_d: int = 8,
    gue_n_draws: int = 10,
    ood2022_preds: dict[str, np.ndarray] | None = None,
    n_boot: int = 1000,
    seed: int = 202601,
) -> dict[str, Any]:
    """Norm-controlled spectral-stress audit over 759 out-of-sample sessions (Sec. 4.2, Table 3)
    and February-March 2022 shock excursion bounds (Sec. 4.3, Table 4B).
    """
    bundle, panel = ensure_panels()
    X_te = np.asarray(X_test_concat, dtype=np.float64)
    di = np.asarray(di_test_concat, dtype=np.int32)
    fold_arr = np.asarray(fold_test_concat, dtype=np.int32)
    uniq_days = np.unique(di)
    day_slices = [np.flatnonzero(di == u) for u in uniq_days]
    n_days = len(uniq_days)
    fold_day = np.array([int(fold_arr[ix[0]]) for ix in day_slices], dtype=np.int32)

    vol5 = bundle["ood_imoex_roll_vol_5d_by_cal"].astype(np.float64)
    stress_contemp = np.zeros(n_days, dtype=bool)
    stress_lead5 = np.zeros(n_days, dtype=bool)
    for f in range(N_FOLDS):
        m = fold_day == f
        t0 = int(uniq_days[m].min())
        thr = float(np.nanpercentile(vol5[:t0], 90))
        stress_contemp[m] = vol5[uniq_days[m]] > thr
        t_lead = np.minimum(uniq_days[m] + 5, len(vol5) - 1)
        stress_lead5[m] = vol5[t_lead] > thr

    n2_row = np.sum(X_te ** 2, axis=1)
    n1_row = np.sqrt(n2_row)
    n2_day = np.array([float(n2_row[ix].mean()) for ix in day_slices], dtype=np.float64)
    n1_day = np.array([float(n1_row[ix].mean()) for ix in day_slices], dtype=np.float64)

    mah_day = np.full(n_days, np.nan, dtype=np.float64)
    pca_day = np.full(n_days, np.nan, dtype=np.float64)
    garch_day = np.full(n_days, np.nan, dtype=np.float64)
    im_ret_pct = panel["imoex_ret_1d"] * 100.0

    for f in range(N_FOLDS):
        fd = get_fold_data(f, modality=modality, target="y_h5_cs", horizon=5)
        X_tr = fd["X_tr"].astype(np.float64)
        mu = X_tr.mean(axis=0)
        C = np.cov(X_tr, rowvar=False) + 1e-5 * np.eye(X_tr.shape[1])
        Ci = np.linalg.inv(C)
        pca = PCA(n_components=4, random_state=DATA_SPLIT_SEED).fit(X_tr)

        f_days_idx = np.flatnonzero(fold_day == f)
        for j in f_days_idx:
            xx = X_te[day_slices[j]]
            dd = xx - mu
            mah_day[j] = float(np.mean(np.einsum("ij,jk,ik->i", dd, Ci, dd)))
            rec = pca.inverse_transform(pca.transform(xx))
            pca_day[j] = float(np.mean(np.sum((xx - rec) ** 2, axis=1)))

        t0 = int(uniq_days[f_days_idx].min())
        t1 = int(uniq_days[f_days_idx].max())
        omega, alpha, beta, h = _fit_causal_garch11(im_ret_pct[1:t0])
        var_path: dict[int, float] = {}
        for t in range(t0, t1 + 1):
            r_prev = float(im_ret_pct[t - 1]) if np.isfinite(im_ret_pct[t - 1]) else 0.0
            h = max(1e-8, omega + alpha * (r_prev ** 2) + beta * h)
            var_path[t] = math.sqrt(h)
        for j in f_days_idx:
            garch_day[j] = var_path[int(uniq_days[j])]

    def _score_summary(sc: np.ndarray, tag_seed: int) -> dict[str, Any]:
        return {
            "auroc_contemp": round(_auroc_rank(sc, stress_contemp), 5),
            "auroc_contemp_ci": _auroc_boot_ci(sc, stress_contemp, n_boot=n_boot, seed=tag_seed),
            "auroc_lead5": round(_auroc_rank(sc, stress_lead5), 5),
            "auroc_lead5_ci": _auroc_boot_ci(sc, stress_lead5, n_boot=n_boot, seed=tag_seed + 17),
        }

    baselines_out = {
        "norm2": _score_summary(n2_day, seed + 1),
        "norm": _score_summary(n1_day, seed + 2),
        "mahalanobis": _score_summary(mah_day, seed + 3),
        "pca4_recon": _score_summary(pca_day, seed + 4),
        "garch11": _score_summary(garch_day, seed + 5),
    }

    def _run_granger(score_d: np.ndarray, ctrl_mat: np.ndarray | None = None) -> dict[str, float]:
        Y_f, XR_f, XU_f = [], [], []
        Y_r, XR_r, XU_r = [], [], []
        for f in range(N_FOLDS):
            idx_f = np.flatnonzero(fold_day == f)
            for k in range(5, len(idx_f)):
                cur = idx_f[k]
                lags = idx_f[k - 5 : k][::-1]
                cal_cur = int(uniq_days[cur])
                cal_lags = uniq_days[lags]
                v_cur = float(vol5[cal_cur])
                v_lags = vol5[cal_lags]
                s_cur = float(score_d[cur])
                s_lags = score_d[lags]
                if not (np.isfinite(v_cur) and np.all(np.isfinite(v_lags)) and np.isfinite(s_cur) and np.all(np.isfinite(s_lags))):
                    continue
                xr_row = list(v_lags)
                if ctrl_mat is not None:
                    c_lags = ctrl_mat[lags].reshape(-1)
                    if not np.all(np.isfinite(c_lags)):
                        continue
                    xr_row.extend(c_lags.tolist())
                Y_f.append(v_cur)
                XR_f.append(xr_row)
                XU_f.append(s_lags.tolist())
                Y_r.append(s_cur)
                XR_r.append(s_lags.tolist())
                XU_r.append(v_lags.tolist())
        if len(Y_f) < 30:
            return {"fwd_wald": float("nan"), "fwd_pval": float("nan"), "rev_wald": float("nan"), "rev_pval": float("nan")}
        wf, pf = _wald_hac_granger(np.array(Y_f), np.array(XR_f), np.array(XU_f), l_hac=5)
        wr, pr = _wald_hac_granger(np.array(Y_r), np.array(XR_r), np.array(XU_r), l_hac=5)
        return {
            "fwd_wald": round(wf, 4), "fwd_pval": round(pf, 6),
            "rev_wald": round(wr, 4), "rev_pval": round(pr, 6),
            "n_obs": int(len(Y_f)),
        }

    def _to_daily(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if len(a) == n_days:
            return a
        return np.array([float(np.nanmean(a[ix])) for ix in day_slices], dtype=np.float64)

    A_lin = np.column_stack([np.ones(n_days), n2_day])
    A_poly = np.column_stack([np.ones(n_days), n1_day, n2_day])
    ctrl_poly = np.column_stack([n1_day, n2_day])

    models_out: dict[str, Any] = {}
    daily_series_export: dict[str, np.ndarray] = {
        "cal_idx": uniq_days,
        "vol5": vol5[uniq_days],
        "stress_contemp": stress_contemp.astype(np.int32),
        "stress_lead5": stress_lead5.astype(np.int32),
        "norm2": n2_day,
        "norm": n1_day,
        "mahalanobis": mah_day,
        "pca4_recon": pca_day,
        "garch11": garch_day,
    }

    for m_idx, (m_name, spec) in enumerate(spectral_by_model.items()):
        e0_d = _to_daily(spec["E0"])
        de_d = _to_daily(spec["dE"])
        e0_sh_d = _to_daily(spec["E0_shifted"]) if "E0_shifted" in spec else (e0_d - 0.5 * n2_day)
        de_tilde_d = _to_daily(spec["dE_tilde"]) if "dE_tilde" in spec else (de_d / np.sqrt(1.0 + n2_day))

        tr_triples_e0 = None
        tr_triples_de = None
        if train_spectral_by_model is not None and m_name in train_spectral_by_model:
            tr_triples_e0, tr_triples_de = {}, {}
            for f, tdict in train_spectral_by_model[m_name].items():
                tr_triples_e0[f] = (tdict["E0"], tdict["norm"], tdict["norm2"])
                tr_triples_de[f] = (tdict["dE"], tdict["norm"], tdict["norm2"])

        e0_perp_d = _fold_causal_norm_residual(e0_d, n1_day, n2_day, fold_day, tr_triples_e0)
        de_perp_d = _fold_causal_norm_residual(de_d, n1_day, n2_day, fold_day, tr_triples_de)

        r2_e0_n2 = float(1.0 - np.var(e0_d - A_lin @ np.linalg.lstsq(A_lin, e0_d, rcond=None)[0]) / max(np.var(e0_d), 1e-18))
        r2_de_n2 = float(1.0 - np.var(de_d - A_lin @ np.linalg.lstsq(A_lin, de_d, rcond=None)[0]) / max(np.var(de_d), 1e-18))
        r2_de_poly = float(1.0 - np.var(de_d - A_poly @ np.linalg.lstsq(A_poly, de_d, rcond=None)[0]) / max(np.var(de_d), 1e-18))

        s_base = seed + 100 * (m_idx + 1)
        m_res: dict[str, Any] = {
            "E0": _score_summary(e0_d, s_base + 1),
            "E0_shifted": _score_summary(e0_sh_d, s_base + 2),
            "E0_perp_causal": _score_summary(e0_perp_d, s_base + 3),
            "dE": _score_summary(de_d, s_base + 4),
            "dE_tilde": _score_summary(de_tilde_d, s_base + 5),
            "dE_perp_causal": _score_summary(de_perp_d, s_base + 6),
            "r2_E0_on_norm2": round(r2_e0_n2, 6),
            "r2_dE_on_norm2": round(r2_de_n2, 6),
            "r2_dE_on_norm_poly": round(r2_de_poly, 6),
            "granger_dE_raw": _run_granger(de_d, ctrl_mat=None),
            "granger_dE_norm_controlled": _run_granger(de_d, ctrl_mat=ctrl_poly),
            "granger_dE_perp_norm_controlled": _run_granger(de_perp_d, ctrl_mat=ctrl_poly),
            "granger_E0_shifted_norm_controlled": _run_granger(e0_sh_d, ctrl_mat=ctrl_poly),
        }

        for opt_k, off in (("E0_sphere", 7), ("dE_sphere", 8), ("fubini_study_step", 9)):
            if opt_k in spec and spec[opt_k] is not None:
                opt_d = _to_daily(spec[opt_k])
                m_res[opt_k] = _score_summary(opt_d, s_base + off)
                daily_series_export[f"{m_name}_{opt_k}"] = opt_d

        models_out[m_name] = m_res
        daily_series_export[f"{m_name}_E0"] = e0_d
        daily_series_export[f"{m_name}_E0_shifted"] = e0_sh_d
        daily_series_export[f"{m_name}_E0_perp"] = e0_perp_d
        daily_series_export[f"{m_name}_dE"] = de_d
        daily_series_export[f"{m_name}_dE_tilde"] = de_tilde_d
        daily_series_export[f"{m_name}_dE_perp"] = de_perp_d

    gue_out = _daily_gue_null_759(X_te, day_slices, d=gue_d, n_draws=gue_n_draws, seed=1000)

    ood_out: dict[str, Any] = {}
    if ood2022_preds is not None:
        ood_d = get_ood2022_data(modality, winsor=False)
        y_ev = ood_d["y_ev"].astype(np.float64)
        reg = ood_d["regime_ev"]
        tr_mu_ood = float(ood_d["tr_mean"])
        ok_y = np.isfinite(y_ev)
        for m_name, p_arr in ood2022_preds.items():
            p = np.asarray(p_arr, dtype=np.float64)
            if p.ndim == 2:
                p = p.mean(axis=0)
            per_reg: dict[str, Any] = {}
            for lab, m_reg in (
                ("all", ok_y),
                ("normal", ok_y & (reg == 0)),
                ("acute", ok_y & (reg == 1)),
                ("post_reopen", ok_y & (reg == 2)),
            ):
                if int(m_reg.sum()) < 20:
                    continue
                e = y_ev[m_reg] - p[m_reg]
                eb = y_ev[m_reg] - tr_mu_ood
                sse = float(np.dot(e, e))
                sse_b = float(np.dot(eb, eb))
                per_reg[lab] = {
                    "r2_vs_precrisis_train_mean": round(1.0 - sse / max(sse_b, 1e-18), 6),
                    "rmse": round(math.sqrt(sse / int(m_reg.sum())), 6),
                    "max_abs_pred": round(float(np.max(np.abs(p[m_reg]))), 5),
                    "n_rows": int(m_reg.sum()),
                }
            ood_out[m_name] = per_reg

    return {
        "modality": modality,
        "n_days": int(n_days),
        "n_stress_contemp": int(stress_contemp.sum()),
        "n_stress_lead5": int(stress_lead5.sum()),
        "baselines": baselines_out,
        "models": models_out,
        "gue_null_759d": gue_out,
        "ood2022_bounds": ood_out,
        "daily_series": daily_series_export,
    }


def survivorship_stress_bound(
    preds: np.ndarray,
    y_true: np.ndarray,
    di: np.ndarray,
    si: np.ndarray,
    k_grid: tuple[int, ...] = (3, 5, 7),
    drop_range: tuple[float, float] = (-0.90, -0.50),
    n_mc: int = 50,
    l_hac: int = L_HAC_DEFAULT,
    seed: int = 202601,
) -> dict[str, Any]:
    """Monte Carlo distressed-exit stress bounds (k in {3, 5, 7} terminal crashes y ~ U[-0.90, -0.50]
    per fold, 50 draws) and continuous pre-2020 listing restriction (Supplementary Material S3, Table S2).
    """
    _, panel = ensure_panels()
    p = np.asarray(preds, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=0)
    yt = np.asarray(y_true, dtype=np.float64)
    d = np.asarray(di, dtype=np.int32)
    s = np.asarray(si, dtype=np.int32)

    ok = np.isfinite(p) & np.isfinite(yt)
    p, yt, d, s = p[ok], yt[ok], d[ok], s[ok]
    uniq_days = np.unique(d)
    day_slices = [np.flatnonzero(d == u) for u in uniq_days]

    base_ic, _, _ = _daily_spearman_and_mse(p, yt, day_slices)
    base_nw = nw_mean_stats(base_ic, l_hac=l_hac)

    pre_mask = panel["pre2020_continuous_mask"][s]
    p_pre, y_pre, d_pre = p[pre_mask], yt[pre_mask], d[pre_mask]
    ud_pre = np.unique(d_pre)
    ds_pre = [np.flatnonzero(d_pre == u) for u in ud_pre]
    ic_pre, _, _ = _daily_spearman_and_mse(p_pre, y_pre, ds_pre)
    pre_nw = nw_mean_stats(ic_pre, l_hac=l_hac)

    ad = panel["active_dates"]
    fold_rows: list[np.ndarray] = []
    for s_iso, e_iso, _ in FOLD_TEST_WINDOWS:
        ix_cal = np.flatnonzero((ad >= s_iso) & (ad <= e_iso))
        fold_rows.append(np.flatnonzero((d >= int(ix_cal.min())) & (d <= int(ix_cal.max()))))

    rng = np.random.default_rng(seed)
    exit_stress: dict[str, dict[str, float]] = {}
    for k_ex in k_grid:
        mc_ics = []
        for _ in range(n_mc):
            y_pert = yt.copy()
            for f_ix in fold_rows:
                if len(f_ix) == 0:
                    continue
                f_secs = np.unique(s[f_ix])
                chosen = rng.choice(f_secs, size=min(int(k_ex), len(f_secs)), replace=False)
                for sec in chosen:
                    sec_rows = f_ix[s[f_ix] == sec]
                    if len(sec_rows) == 0:
                        continue
                    pos = int(rng.integers(0, len(sec_rows)))
                    crash_rows = sec_rows[pos : min(len(sec_rows), pos + 5)]
                    y_pert[crash_rows] = rng.uniform(drop_range[0], drop_range[1])
            ic_m, _, _ = _daily_spearman_and_mse(p, y_pert, day_slices)
            mc_ics.append(float(np.nanmean(ic_m)))
        arr = np.asarray(mc_ics, dtype=np.float64)
        exit_stress[f"k_{k_ex}_exits_per_fold"] = {
            "rank_ic_mean": round(float(arr.mean()), 6),
            "rank_ic_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
            "rank_ic_p05": round(float(np.percentile(arr, 5.0)), 6),
            "delta_vs_base": round(float(arr.mean() - base_nw["mean"]), 6),
        }

    return {
        "baseline_rank_ic": round(base_nw["mean"], 6),
        "baseline_n_rows": int(len(yt)),
        "pre2020_continuous": {
            "rank_ic": round(pre_nw["mean"], 6),
            "nw_se": round(pre_nw["nw_se"], 6),
            "nw_pval": round(pre_nw["nw_pval"], 6),
            "n_rows": int(len(y_pre)),
            "row_retention_pct": round(100.0 * len(y_pre) / max(len(yt), 1), 2),
        },
        "distressed_exit_stress": exit_stress,
    }
