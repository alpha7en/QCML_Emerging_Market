"""Reproduce the empirical benchmark suites from raw MOEX panels and derived news aggregates.

Executes the six experimental suites reported in Section 4 and Supplementary Material S1-S3:
  - `synthetic_controls`:      Synthetic d=2 Bloch-sphere and d=3 SU(3) quadrupole positive controls
                               plus purged walk-forward calendar verification (Supplementary S1, Sec. 3.2).
  - `classical_controls`:      Linear factors, Ridge, Random Fourier Features, boosted trees, MLPs,
                               and iso-parameter FiLM-MLP across MKT (p=12) and M+N (p=18) (Sec. 4.1, 4.3).
  - `vqc_controls`:            Three-qubit data-reuploading statevector VQCs under affine (L=9) and
                               cyclic (L=3) encodings with full-Hermitian vs local Pauli-ZX readouts
                               and CNOT vs product-state ablations (Sec. 2.1, Sec. 4.1, 4.3).
  - `qcml_benchmark`:          Matrix-form QCML across d in {2, 4, 8, 16}, real-symmetric d=11,
                               commuting diagonal d=64, energy-anchoring ablations (lambda_E = 0),
                               date-wise news-permutation placebos, and unified Hansen MCS (Sec. 4.1, 4.3).
  - `spectral_stress`:         Norm-controlled spectral volatility audit (E0, Delta E, E0^perp,
                               Delta E^perp), bivariate VAR(5) Granger tests, 759-day GUE null, and
                               February-March 2022 out-of-distribution crisis stress (Sec. 4.2, 4.3).
  - `multioutput_robustness`:  Shared-state q=4 multi-task estimation (Supplementary S2) and horizon h=15,
                               rolling window W_train=126, liquidity sub-tier, weekly cohort, and
                               delisting stress diagnostics (Supplementary S3).

Requires `data/raw/moex_equities.parquet` (obtained via `python -I experiments/fetch_moex.py`).
All paper tables and figures can be generated directly from the shipped `data/runs/` directory
without re-running this script.
"""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import csv
import datetime as dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcml_moex import (
    DATA_SPLIT_SEED,
    L_HAC_DEFAULT,
    N_FOLDS,
    P_MARKET,
    SEEDS10,
    StatevectorVQC,
    ensure_panels,
    evaluate_predictions,
    evaluate_spectral_stress,
    extract_spectral_observables,
    fit_predict_classical,
    fit_predict_qcml,
    fit_predict_vqc,
    get_fold_data,
    get_ood2022_data,
    get_pooled_oos_meta,
    hansen_mcs_05,
    holm_bonferroni,
    paired_tests,
    permute_news_by_date,
    run_synthetic_controls,
    survivorship_stress_bound,
)

ALL_SUITES = (
    "synthetic_controls",
    "classical_controls",
    "vqc_controls",
    "qcml_benchmark",
    "spectral_stress",
    "multioutput_robustness",
)

RUNS_DIR = REPO_ROOT / "data" / "runs"
PREDS_CACHE_DIR = REPO_ROOT / "data" / "cache" / "preds"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def _device_label() -> str:
    arch = platform.machine().lower()
    sys_name = platform.system().lower()
    cpus = os.cpu_count() or 1
    if sys_name == "darwin" and "arm" in arch:
        return f"cpu_apple_silicon_{cpus}core"
    return f"cpu_{sys_name}_{arch}_{cpus}core"


def _write_suite_summary(
    suite_dir: Path,
    suite: str,
    wall_time_s: float,
    backend: dict[str, Any],
    seeds: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    summary = _json_safe(
        {
            "suite": suite,
            "finished_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_time_s": round(float(wall_time_s), 5),
            "backend": backend,
            "seeds": seeds,
            "metrics": metrics,
        }
    )
    _write_json(suite_dir / "summary.json", summary)
    return summary


def _compact_eval(ev: dict[str, Any]) -> dict[str, Any]:
    skip_keys = {"daily_ic", "daily_mse", "daily_ic_single_mean", "daily_mse_single_mean", "uniq_days"}
    out: dict[str, Any] = {}
    for k, v in ev.items():
        if k in skip_keys:
            continue
        if k == "economics" and isinstance(v, dict):
            out[k] = {
                ek: ev_v
                for ek, ev_v in v.items()
                if ek not in {"weekly_cohort_lo_net15", "weekly_cohort_ls_net15", "weekly_cohort_dates"}
            }
        else:
            out[k] = v
    return _json_safe(out)


def _save_pred_cache(preds_dir: Path, name: str, preds_seeds: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"preds_seeds": np.asarray(preds_seeds, dtype=np.float32)}
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(preds_dir / f"{name}.npz", **payload)


def _load_pred_cache(preds_dir: Path, name: str) -> dict[str, np.ndarray] | None:
    path = preds_dir / f"{name}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _worker_classical_job(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        model_name,
        f,
        modality,
        target,
        horizon,
        train_window,
        post2022_only,
        shuffle_news,
        seed,
        single_thread,
    ) = args
    if single_thread:
        torch.set_num_threads(1)
    ensure_panels()
    fd = get_fold_data(
        f,
        modality=modality,
        target=target,
        horizon=horizon,
        train_window=train_window,
        post2022_only=post2022_only,
    )
    if shuffle_news:
        fd = dict(fd)
        fd["X_tr"] = permute_news_by_date(fd["X_tr"], fd["di_tr"], p_mkt=P_MARKET, seed=seed * 10 + f)
        fd["X_va"] = permute_news_by_date(fd["X_va"], fd["di_va"], p_mkt=P_MARKET, seed=seed * 10 + f + 100)
        fd["X_te"] = permute_news_by_date(fd["X_te"], fd["di_te"], p_mkt=P_MARKET, seed=seed * 10 + f + 200)
    res = fit_predict_classical(model_name, fd, seed=seed)
    if res.get("skipped"):
        return {"f": f, "seed": seed, "skipped": True, "reason": res.get("reason", "unavailable")}
    val_mse = (
        float(np.mean((fd["y_va"].astype(np.float64) - res["pred_va"]) ** 2))
        if len(res["pred_va"]) > 0
        else float("nan")
    )
    return {
        "f": f,
        "seed": seed,
        "skipped": False,
        "pred_te": res["pred_te"],
        "val_mse": val_mse,
        "n_params": int(res["n_params"]),
    }


def _worker_vqc_job(args: tuple[Any, ...]) -> dict[str, Any]:
    encoding, n_qubits, n_layers, entangle, readout, f, modality, seed, single_thread = args
    if single_thread:
        torch.set_num_threads(1)
    ensure_panels()
    fd = get_fold_data(f, modality=modality, target="y_h5_cs", horizon=5)
    res = fit_predict_vqc(
        encoding=encoding,
        n_qubits=n_qubits,
        n_layers=n_layers,
        entangle=entangle,
        readout=readout,
        fold_dict=fd,
        seed=seed,
    )
    val_mse = (
        float(np.mean((fd["y_va"].astype(np.float64) - res["pred_va"]) ** 2))
        if len(res["pred_va"]) > 0
        else float("nan")
    )
    return {
        "f": f,
        "seed": seed,
        "pred_te": res["pred_te"],
        "val_mse": val_mse,
        "n_qubits": int(res["n_qubits"]),
        "depth": int(res["depth"]),
        "n_2q_gates": int(res["n_2q_gates"]),
        "n_params": int(res["n_params"]),
        "q_mw_mean": float(res["q_mw_mean"]),
    }


def _worker_qcml_job(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        mode,
        d,
        lambda_e,
        beta,
        multi_output,
        candelori_w,
        f,
        modality,
        target,
        horizon,
        train_window,
        post2022_only,
        shuffle_news,
        seed,
        return_spectral,
        single_thread,
    ) = args
    if single_thread:
        torch.set_num_threads(1)
    ensure_panels()
    fd = get_fold_data(
        f,
        modality=modality,
        target=target,
        horizon=horizon,
        train_window=train_window,
        post2022_only=post2022_only,
    )
    if shuffle_news:
        fd = dict(fd)
        fd["X_tr"] = permute_news_by_date(fd["X_tr"], fd["di_tr"], p_mkt=P_MARKET, seed=seed * 10 + f)
        fd["X_va"] = permute_news_by_date(fd["X_va"], fd["di_va"], p_mkt=P_MARKET, seed=seed * 10 + f + 100)
        fd["X_te"] = permute_news_by_date(fd["X_te"], fd["di_te"], p_mkt=P_MARKET, seed=seed * 10 + f + 200)
    res = fit_predict_qcml(
        mode=mode,
        d=d,
        fold_dict=fd,
        seed=seed,
        lambda_e=lambda_e,
        beta=beta,
        multi_output=multi_output,
        candelori_w=candelori_w,
    )
    val_mse = (
        float(np.mean((fd["y_va"].astype(np.float64) - res["pred_va"]) ** 2))
        if (not (candelori_w is not None or lambda_e >= 0.9999)) and len(res["pred_va"]) > 0
        else float(res["best_val_loss"])
    )
    out: dict[str, Any] = {
        "f": f,
        "seed": seed,
        "pred_te": res["pred_te"],
        "val_mse": val_mse,
        "n_params": int(res["n_params"]),
        "n_eff_params": int(res["n_eff_params"]),
        "kappa_nc": float(res["kappa_nc"]),
        "kappa_nc_mkt_news": float(res["kappa_nc_mkt_news"]),
        "norm_ratio_news_mkt": float(res["norm_ratio_news_mkt"]),
        "hermiticity_err": float(res["hermiticity_err"]),
    }
    if return_spectral:
        st = res["model_state"]
        obs_te = extract_spectral_observables(st, fd["X_te"], date_idx=fd["di_te"])
        x_te_64 = fd["X_te"].astype(np.float64)
        x_te_sph = x_te_64 / np.sqrt(np.sum(x_te_64**2, axis=1, keepdims=True) + 1e-8)
        obs_sph = extract_spectral_observables(st, x_te_sph, date_idx=None)

        # Training-fold daily averages used exclusively to fit causal quadratic norm regressions (Sec. 3.3, Sec. 4.2).
        obs_tr = extract_spectral_observables(st, fd["X_tr"], date_idx=None)
        di_tr = fd["di_tr"]
        u_tr = np.unique(di_tr)
        n2_tr = np.sum(fd["X_tr"].astype(np.float64) ** 2, axis=1)
        n1_tr = np.sqrt(n2_tr)
        out["spectral_te"] = {
            "E0": obs_te["E0"],
            "E0_shifted": obs_te["E0_shifted"],
            "dE": obs_te["dE"],
            "dE_tilde": obs_te["dE_tilde"],
            "fubini_study_step": obs_te["fubini_study_step"],
            "E0_sphere": obs_sph["E0"],
            "dE_sphere": obs_sph["dE"],
        }
        out["spectral_tr_daily"] = {
            "E0": np.array([float(obs_tr["E0"][di_tr == u].mean()) for u in u_tr], dtype=np.float64),
            "dE": np.array([float(obs_tr["dE"][di_tr == u].mean()) for u in u_tr], dtype=np.float64),
            "norm": np.array([float(n1_tr[di_tr == u].mean()) for u in u_tr], dtype=np.float64),
            "norm2": np.array([float(n2_tr[di_tr == u].mean()) for u in u_tr], dtype=np.float64),
        }
    return out


def _worker_qcml_t4_single_job(args: tuple[Any, ...]) -> dict[str, Any]:
    col_j, f, seed, single_thread = args
    if single_thread:
        torch.set_num_threads(1)
    ensure_panels()
    fd = get_fold_data(f, "market_news", "Y_state", 5)
    fd_1 = dict(fd)
    fd_1["y_tr"] = fd["y_tr"][:, col_j]
    fd_1["y_va"] = fd["y_va"][:, col_j]
    fd_1["y_te"] = fd["y_te"][:, col_j]
    r1 = fit_predict_qcml("hermitian", 8, fd_1, seed=seed, lambda_e=0.05, multi_output=False)
    return {
        "col_j": int(col_j),
        "f": int(f),
        "seed": int(seed),
        "pred_te": r1["pred_te"],
        "n_params": int(r1["n_params"]),
    }


_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_POOL_SIZE: int = 0


def _shutdown_pool() -> None:
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=False)
        _POOL = None
        _POOL_SIZE = 0


atexit.register(_shutdown_pool)


def _run_jobs(worker_fn: Any, jobs: list[tuple[Any, ...]], n_jobs: int) -> list[dict[str, Any]]:
    global _POOL, _POOL_SIZE
    if n_jobs <= 1 or len(jobs) <= 1:
        return [worker_fn(j) for j in jobs]
    if _POOL is None or _POOL_SIZE != n_jobs:
        _shutdown_pool()
        _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)
        _POOL_SIZE = n_jobs
    return list(_POOL.map(worker_fn, jobs))


def _assemble_seed_fold_preds(
    results: list[dict[str, Any]], seeds: list[int], n_folds: int = N_FOLDS
) -> tuple[np.ndarray, dict[int, float]]:
    by_sf: dict[tuple[int, int], np.ndarray] = {}
    val_by_f: dict[int, list[float]] = {f: [] for f in range(n_folds)}
    for r in results:
        by_sf[(int(r["seed"]), int(r["f"]))] = np.asarray(r["pred_te"], dtype=np.float64)
        if np.isfinite(r.get("val_mse", float("nan"))):
            val_by_f[int(r["f"])].append(float(r["val_mse"]))
    seed_rows = [np.concatenate([by_sf[(sd, f)] for f in range(n_folds)], axis=0) for sd in seeds]
    preds = np.stack(seed_rows, axis=0)
    val_mean = {f: float(np.mean(v)) if v else float("inf") for f, v in val_by_f.items()}
    return preds, val_mean


def _export_daily_csv(
    path: Path, dates_iso: np.ndarray, uniq_days: np.ndarray, series_map: dict[str, np.ndarray]
) -> None:
    cols = ["cal_idx", "date"] + sorted(series_map.keys())
    rows: list[dict[str, Any]] = []
    for k, cal_i in enumerate(uniq_days):
        row: dict[str, Any] = {"cal_idx": int(cal_i), "date": str(dates_iso[int(cal_i)])}
        for name, arr in series_map.items():
            v = float(arr[k]) if k < len(arr) else float("nan")
            row[name] = "" if not np.isfinite(v) else f"{v:.6f}"
        rows.append(row)
    _write_csv(path, cols, rows)


def _export_equity_curves_csv(path: Path, dates_iso: np.ndarray, econ_map: dict[str, dict[str, Any]]) -> None:
    """Weekly staggered 1/5-tranche net-15 bps active long-only and dollar-neutral return series (Sec. 3.3)."""
    if not econ_map:
        return
    first = next(iter(econ_map.values()))
    cohorts_d = first["weekly_cohort_dates"]
    min_w = min(len(c) for c in cohorts_d)
    cols = ["week_idx", "cohort0_date"]
    for m in sorted(econ_map.keys()):
        cols.extend([f"{m}_lo_act_net15", f"{m}_ls_net15"])
    rows: list[dict[str, Any]] = []
    for w in range(min_w):
        cal_i = int(cohorts_d[0][w])
        row: dict[str, Any] = {"week_idx": w, "cohort0_date": str(dates_iso[cal_i])}
        for m in sorted(econ_map.keys()):
            lo_tr = float(np.mean([c[w] for c in econ_map[m]["weekly_cohort_lo_net15"]]))
            ls_tr = float(np.mean([c[w] for c in econ_map[m]["weekly_cohort_ls_net15"]]))
            row[f"{m}_lo_act_net15"] = f"{lo_tr:.6f}"
            row[f"{m}_ls_net15"] = f"{ls_tr:.6f}"
        rows.append(row)
    _write_csv(path, cols, rows)


def run_suite_synthetic_controls(out_dir: Path, device_label: str) -> dict[str, Any]:
    """Supplementary S1 synthetic positive controls (d=2 Bloch sphere and d=3 SU(3) quadrupole) and
    purged walk-forward boundary verification (Sec. 3.2).
    """
    t0 = time.perf_counter()
    bundle, panel = ensure_panels()
    ad = panel["active_dates"]
    syn = run_synthetic_controls()

    schedule_h5, schedule_h15 = [], []
    for f in range(N_FOLDS):
        fd5 = get_fold_data(f, "market_news", "y_h5_cs", 5)
        fd15 = get_fold_data(f, "market_news", "y_h15_cs", 15)
        schedule_h5.append(
            {
                "fold": f,
                "train_range": [str(ad[int(fd5["di_tr"].min())]), str(ad[int(fd5["di_tr"].max())])],
                "val_range": [str(ad[int(fd5["di_va"].min())]), str(ad[int(fd5["di_va"].max())])],
                "test_range": [str(ad[int(fd5["di_te"].min())]), str(ad[int(fd5["di_te"].max())])],
                "gap_tr_va": int(fd5["di_va"].min() - fd5["di_tr"].max()),
                "gap_va_te": int(fd5["di_te"].min() - fd5["di_va"].max()),
                "n_tr": int(len(fd5["y_tr"])),
                "n_va": int(len(fd5["y_va"])),
                "n_te": int(len(fd5["y_te"])),
            }
        )
        schedule_h15.append(
            {
                "fold": f,
                "gap_tr_va": int(fd15["di_va"].min() - fd15["di_tr"].max()),
                "gap_va_te": int(fd15["di_te"].min() - fd15["di_va"].max()),
                "n_tr": int(len(fd15["y_tr"])),
                "n_va": int(len(fd15["y_va"])),
                "n_te": int(len(fd15["y_te"])),
            }
        )

    pooled = get_pooled_oos_meta("market_news", "y_h5_cs", 5)
    const_check = evaluate_predictions(
        pooled["tr_mean_per_row"][None, :],
        pooled["y_te"],
        pooled["di_te"],
        pooled["si_te"],
        pooled["tr_mean_per_row"],
        ortho_factors=pooled["ortho_te"],
        n_boot=50,
        compute_econ=False,
    )

    _write_json(out_dir / "synthetic_pos_control.json", syn)
    _write_json(
        out_dir / "fold_schedule.json",
        {
            "n_oos_rows_h5": int(len(pooled["y_te"])),
            "n_oos_days_h5": int(len(np.unique(pooled["di_te"]))),
            "constant_tr_mean_oos_r2": float(const_check["oos_r2"]),
            "h5_folds": schedule_h5,
            "h15_folds": schedule_h15,
        },
    )

    n_liq_day = ((panel["close"] > 0) & (panel["value_rub"] > 0)).sum(axis=1)
    n_exdiv_day = panel["exdiv_sens_mask"].sum(axis=1)
    imoex_ret_1d = bundle.get("ood_imoex_ret_1d_by_cal")
    imoex_vol_5d = bundle.get("ood_imoex_roll_vol_5d_by_cal")
    regime_rows: list[dict[str, Any]] = []
    for idx in range(len(ad)):
        r1 = float(imoex_ret_1d[idx]) if imoex_ret_1d is not None else float("nan")
        v5 = float(imoex_vol_5d[idx]) if imoex_vol_5d is not None else float("nan")
        regime_rows.append(
            {
                "active_cal_idx": int(idx),
                "tradedate": str(ad[idx]),
                "imoex_close": round(float(panel["imoex"][idx]), 4),
                "n_liquid_tickers": int(n_liq_day[idx]),
                "cbr_key_rate_pct": round(float(panel["cbr_key_rate_pct"][idx]), 4),
                "short_ban": int(bool(panel["short_ban_mask"][idx])),
                "is_quarantine_h5": int(464 <= idx <= 476),
                "is_halt_adjacent": int(idx in (470, 471, 472)),
                "n_exdiv_flagged": int(n_exdiv_day[idx]),
                "imoex_ret_1d": round(r1, 6) if math.isfinite(r1) else None,
                "imoex_roll_vol_5d": round(v5, 6) if math.isfinite(v5) else None,
            }
        )
    _write_csv(
        out_dir / "daily_regime_series.csv",
        [
            "active_cal_idx",
            "tradedate",
            "imoex_close",
            "n_liquid_tickers",
            "cbr_key_rate_pct",
            "short_ban",
            "is_quarantine_h5",
            "is_halt_adjacent",
            "n_exdiv_flagged",
            "imoex_ret_1d",
            "imoex_roll_vol_5d",
        ],
        regime_rows,
    )

    metrics = {
        "n_oos_rows_h5": int(len(pooled["y_te"])),
        "constant_tr_mean_oos_r2": float(const_check["oos_r2"]),
        "b0a": syn["b0a"],
        "b0b": syn["b0b"],
    }
    return _write_suite_summary(
        out_dir,
        suite="synthetic_controls",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "pytorch+scikit-learn",
            "device": device_label,
            "diff_method": "tikhonov_eigh+daleckii_krein",
            "shots": None,
            "noise_model": "ideal",
            "n_qubits": 0,
            "depth": 0,
            "n_2q_gates": 0,
            "n_params": 36,
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics=metrics,
    )


def run_suite_classical_controls(
    out_dir: Path, preds_dir: Path, n_jobs: int, device_label: str
) -> dict[str, Any]:
    """Tier-1 linear factors and Tier-2 nonlinear classical baselines across MKT and M+N (Sec. 4.1)."""
    t0 = time.perf_counter()
    _, panel = ensure_panels()
    ad = panel["active_dates"]
    seeds_multi = list(SEEDS10)
    seeds_shuf = SEEDS10[:5]
    single_thread = n_jobs > 1

    det_models = ["zero", "mom5", "lowvol", "mom21", "mom21_lowvol", "ridge", "hgbdt", "lgbm"]
    stoch_models = ["rff_krr", "mlp_relu", "mlp_tanh", "film_mlp_iso"]
    all_models = det_models + stoch_models

    models_summary: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    cost_sweep_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    fold_metrics_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    mcs_out: dict[str, Any] = {}
    best_val_classical: dict[str, str] = {}
    daily_ic_csv_map: dict[str, np.ndarray] = {}
    daily_mse_csv_map: dict[str, np.ndarray] = {}
    equity_map: dict[str, dict[str, Any]] = {}
    uniq_days_ref: np.ndarray | None = None
    skipped_models: dict[str, str] = {}
    placebo_out: dict[str, Any] = {}

    for mod in ("market_only", "market_news"):
        pooled = get_pooled_oos_meta(mod, "y_h5_cs", 5)
        uniq_days_ref = np.unique(pooled["di_te"])
        mod_tag = "mkt" if mod == "market_only" else "mn"
        loss_ic_map: dict[str, np.ndarray] = {}
        loss_mse_map: dict[str, np.ndarray] = {}
        val_mses_learned: dict[str, float] = {}

        for m_name in all_models:
            m_seeds = [101] if m_name in det_models else seeds_multi
            jobs = [
                (m_name, f, mod, "y_h5_cs", 5, None, False, False, sd, single_thread)
                for sd in m_seeds
                for f in range(N_FOLDS)
            ]
            res_list = _run_jobs(_worker_classical_job, jobs, n_jobs)
            if any(r.get("skipped") for r in res_list):
                skipped_models[f"{mod}_{m_name}"] = str(res_list[0].get("reason", "skipped"))
                continue

            preds, val_by_f = _assemble_seed_fold_preds(res_list, m_seeds)
            n_par = int(res_list[-1]["n_params"])
            compute_econ = m_name != "zero"
            ev = evaluate_predictions(
                preds,
                pooled["y_te"],
                pooled["di_te"],
                pooled["si_te"],
                pooled["tr_mean_per_row"],
                ortho_factors=pooled["ortho_te"],
                k_trials_dsr=40,
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
                horizon=5,
                exec_lag=1,
                compute_econ=compute_econ,
            )
            ev["n_params"] = n_par
            ev["mean_val_mse"] = round(float(np.mean(list(val_by_f.values()))), 8)
            _save_pred_cache(
                preds_dir,
                f"classical_{mod}_{m_name}",
                preds,
                {"daily_ic": ev["daily_ic"], "daily_mse": ev["daily_mse"]},
            )

            models_summary[mod][m_name] = _compact_eval(ev)
            fold_metrics_out[mod][m_name] = {
                "single_seed_fold_ics": ev["single_seed"]["fold_rank_ics"],
                "ens_fold_ics": ev["ensemble"]["fold_rank_ics"],
                "val_mse_by_fold": {str(k): round(v, 8) for k, v in val_by_f.items()},
            }
            if compute_econ and "economics" in ev:
                cost_sweep_out[mod][m_name] = _compact_eval({"economics": ev["economics"]})["economics"]
                if m_name in ("mom21_lowvol", "ridge", "rff_krr", "hgbdt", "mlp_tanh", "film_mlp_iso"):
                    equity_map[f"{mod_tag}_{m_name}"] = ev["economics"]

            daily_ic_csv_map[f"{mod_tag}_{m_name}"] = ev["daily_ic"]
            daily_mse_csv_map[f"{mod_tag}_{m_name}"] = ev["daily_mse"]
            loss_ic_map[m_name] = -ev["daily_ic"]
            if m_name not in ("zero", "mom5", "lowvol", "mom21", "mom21_lowvol"):
                loss_mse_map[m_name] = ev["daily_mse"]
                val_mses_learned[m_name] = float(np.mean(list(val_by_f.values())))

        best_val_classical[mod] = min(val_mses_learned, key=val_mses_learned.get)
        mcs_ic = hansen_mcs_05(loss_ic_map, n_boot=1000, block=10)
        mcs_mse = hansen_mcs_05(loss_mse_map, n_boot=1000, block=10)
        mcs_out[mod] = {"mcs_ic": mcs_ic, "mcs_mse": mcs_mse, "best_val_classical": best_val_classical[mod]}
        for m_name in models_summary[mod]:
            if m_name in mcs_ic:
                models_summary[mod][m_name]["in_mcs_05_ic"] = int(mcs_ic[m_name]["in_mcs_05"])
                models_summary[mod][m_name]["mcs_pval_ic"] = mcs_ic[m_name]["mcs_pval"]
            if m_name in mcs_mse:
                models_summary[mod][m_name]["in_mcs_05_mse"] = int(mcs_mse[m_name]["in_mcs_05"])
                models_summary[mod][m_name]["mcs_pval_mse"] = mcs_mse[m_name]["mcs_pval"]

    pooled_mn = get_pooled_oos_meta("market_news", "y_h5_cs", 5)
    for p_model, p_seeds in [("rff_krr", seeds_shuf), ("hgbdt", [101])]:
        jobs = [
            (p_model, f, "market_news", "y_h5_cs", 5, None, False, True, sd, single_thread)
            for sd in p_seeds
            for f in range(N_FOLDS)
        ]
        r_sh = _run_jobs(_worker_classical_job, jobs, n_jobs)
        p_sh, _ = _assemble_seed_fold_preds(r_sh, p_seeds)
        ev_sh = evaluate_predictions(
            p_sh,
            pooled_mn["y_te"],
            pooled_mn["di_te"],
            pooled_mn["si_te"],
            pooled_mn["tr_mean_per_row"],
            ortho_factors=pooled_mn["ortho_te"],
            n_boot=2000,
            l_hac=L_HAC_DEFAULT,
            compute_econ=False,
        )
        pt_real_vs_shuf = paired_tests(
            daily_ic_csv_map[f"mn_{p_model}"], ev_sh["daily_ic"], n_boot=2000, l_hac=L_HAC_DEFAULT
        )
        pt_real_vs_mkt = paired_tests(
            daily_ic_csv_map[f"mn_{p_model}"], daily_ic_csv_map[f"mkt_{p_model}"], n_boot=2000, l_hac=L_HAC_DEFAULT
        )
        placebo_out[p_model] = {
            "shuffled_eval": _compact_eval(ev_sh),
            "real_minus_shuffled_ic": _json_safe(pt_real_vs_shuf),
            "real_mn_minus_mkt_ic": _json_safe(pt_real_vs_mkt),
        }

    assert uniq_days_ref is not None
    _export_daily_csv(out_dir / "daily_rank_ic.csv", ad, uniq_days_ref, daily_ic_csv_map)
    _export_daily_csv(out_dir / "daily_mse.csv", ad, uniq_days_ref, daily_mse_csv_map)
    _export_equity_curves_csv(out_dir / "equity_curves.csv", ad, equity_map)
    _write_json(out_dir / "cost_sweep.json", cost_sweep_out)
    _write_json(out_dir / "fold_metrics.json", fold_metrics_out)
    _write_json(out_dir / "mcs_and_paired.json", {"mcs": mcs_out, "news_placebo": placebo_out})

    best_c_mn = best_val_classical["market_news"]
    ref_ev = models_summary["market_news"][best_c_mn]
    return _write_suite_summary(
        out_dir,
        suite="classical_controls",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "scikit-learn+pytorch+lightgbm",
            "device": device_label,
            "diff_method": "analytic+autograd",
            "shots": None,
            "noise_model": "none",
            "n_qubits": 0,
            "depth": 0,
            "n_2q_gates": 0,
            "n_params": 1207,
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics={
            "best_val_classical_market_only": best_val_classical["market_only"],
            "best_val_classical_market_news": best_c_mn,
            "oos_rank_ic_mean": ref_ev["oos_rank_ic_single_mean"],
            "oos_rank_ic_ens": ref_ev["oos_rank_ic_ens"],
            "oos_rank_ic_nw_pval": ref_ev["oos_rank_ic_nw_pval"],
            "oos_r2": ref_ev["oos_r2"],
            "skipped_models": skipped_models,
            "models": models_summary,
            "news_placebo": placebo_out,
        },
    )


def run_suite_vqc_controls(
    out_dir: Path, preds_dir: Path, n_jobs: int, device_label: str
) -> dict[str, Any]:
    """Tier-3 three-qubit variational circuit controls and entanglement ablations (Sec. 2.1, Sec. 4.1)."""
    t0 = time.perf_counter()
    _, panel = ensure_panels()
    ad = panel["active_dates"]
    seeds_full = list(SEEDS10)
    seeds_zx = SEEDS10[:5]
    single_thread = n_jobs > 1

    probe_vqc = StatevectorVQC(
        in_features=18, n_qubits=3, n_layers=9, encoding="affine", entangle=True, readout="full_hermitian", seed=101
    )
    pspec = probe_vqc.circuit_spec()

    vqc_specs = [
        ("vqc_n3_affine_cnot_fullherm", "affine", 3, 9, True, "full_hermitian", seeds_full),
        ("vqc_n3_affine_nocnot_fullherm", "affine", 3, 9, False, "full_hermitian", seeds_full),
        ("vqc_n3_cyclic_cnot_fullherm", "cyclic_packed", 3, 3, True, "full_hermitian", seeds_full),
        ("vqc_n3_cyclic_nocnot_fullherm", "cyclic_packed", 3, 3, False, "full_hermitian", seeds_full),
        ("vqc_n3_affine_cnot_localzx", "affine", 3, 9, True, "local_zx", seeds_zx),
        ("vqc_n3_affine_nocnot_localzx", "affine", 3, 9, False, "local_zx", seeds_zx),
    ]

    models_summary: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    cost_sweep_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    fold_metrics_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    circuit_specs_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    ent_tests: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    daily_ic_csv_map: dict[str, np.ndarray] = {}
    daily_mse_csv_map: dict[str, np.ndarray] = {}
    uniq_days_ref: np.ndarray | None = None

    for mod in ("market_only", "market_news"):
        pooled = get_pooled_oos_meta(mod, "y_h5_cs", 5)
        uniq_days_ref = np.unique(pooled["di_te"])
        mod_tag = "mkt" if mod == "market_only" else "mn"

        for v_name, enc, nq, nl, ent, ro, v_seeds in vqc_specs:
            jobs = [
                (enc, nq, nl, ent, ro, f, mod, sd, single_thread)
                for sd in v_seeds
                for f in range(N_FOLDS)
            ]
            res_list = _run_jobs(_worker_vqc_job, jobs, n_jobs)
            preds, val_by_f = _assemble_seed_fold_preds(res_list, v_seeds)
            ev = evaluate_predictions(
                preds,
                pooled["y_te"],
                pooled["di_te"],
                pooled["si_te"],
                pooled["tr_mean_per_row"],
                ortho_factors=pooled["ortho_te"],
                k_trials_dsr=40,
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
                horizon=5,
                exec_lag=1,
                compute_econ=True,
            )
            c_spec = {
                "n_qubits": int(res_list[0]["n_qubits"]),
                "depth": int(res_list[0]["depth"]),
                "n_2q_gates": int(res_list[0]["n_2q_gates"]),
                "n_params": int(res_list[0]["n_params"]),
                "q_mw_mean": round(float(np.mean([r["q_mw_mean"] for r in res_list])), 6),
                "encoding": enc,
                "entangle": ent,
                "readout": ro,
            }
            ev.update(c_spec)
            ev["mean_val_mse"] = round(float(np.mean(list(val_by_f.values()))), 8)
            _save_pred_cache(
                preds_dir,
                f"vqc_{mod}_{v_name}",
                preds,
                {"daily_ic": ev["daily_ic"], "daily_mse": ev["daily_mse"]},
            )

            models_summary[mod][v_name] = _compact_eval(ev)
            circuit_specs_out[mod][v_name] = c_spec
            cost_sweep_out[mod][v_name] = _compact_eval({"economics": ev["economics"]})["economics"]
            fold_metrics_out[mod][v_name] = {
                "single_seed_fold_ics": ev["single_seed"]["fold_rank_ics"],
                "ens_fold_ics": ev["ensemble"]["fold_rank_ics"],
            }
            daily_ic_csv_map[f"{mod_tag}_{v_name}"] = ev["daily_ic"]
            daily_mse_csv_map[f"{mod_tag}_{v_name}"] = ev["daily_mse"]

        for pair_label, k_cnot, k_no in [
            ("affine_fullherm", "vqc_n3_affine_cnot_fullherm", "vqc_n3_affine_nocnot_fullherm"),
            ("cyclic_fullherm", "vqc_n3_cyclic_cnot_fullherm", "vqc_n3_cyclic_nocnot_fullherm"),
            ("affine_localzx", "vqc_n3_affine_cnot_localzx", "vqc_n3_affine_nocnot_localzx"),
        ]:
            pt_ic = paired_tests(
                daily_ic_csv_map[f"{mod_tag}_{k_cnot}"],
                daily_ic_csv_map[f"{mod_tag}_{k_no}"],
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
            )
            pt_mse = paired_tests(
                daily_mse_csv_map[f"{mod_tag}_{k_no}"],
                daily_mse_csv_map[f"{mod_tag}_{k_cnot}"],
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
            )
            ent_tests[mod][pair_label] = {
                "single_seed_delta_ic": round(
                    models_summary[mod][k_cnot]["oos_rank_ic_single_mean"]
                    - models_summary[mod][k_no]["oos_rank_ic_single_mean"],
                    6,
                ),
                "ens_paired_ic": _json_safe(pt_ic),
                "ens_paired_mse_gain": _json_safe(pt_mse),
            }

    assert uniq_days_ref is not None
    _export_daily_csv(out_dir / "daily_rank_ic.csv", ad, uniq_days_ref, daily_ic_csv_map)
    _export_daily_csv(out_dir / "daily_mse.csv", ad, uniq_days_ref, daily_mse_csv_map)
    _write_json(out_dir / "cost_sweep.json", cost_sweep_out)
    _write_json(out_dir / "fold_metrics.json", fold_metrics_out)
    _write_json(out_dir / "vqc_entanglement_tests.json", ent_tests)
    _write_json(out_dir / "circuit_specs.json", circuit_specs_out)

    prim_mn = models_summary["market_news"]["vqc_n3_affine_cnot_fullherm"]
    return _write_suite_summary(
        out_dir,
        suite="vqc_controls",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "pytorch_statevector",
            "device": device_label,
            "diff_method": "exact_autograd",
            "shots": None,
            "noise_model": "ideal",
            "n_qubits": int(pspec["n_qubits"]),
            "depth": int(pspec["depth"]),
            "n_2q_gates": int(pspec["n_2q_gates"]),
            "n_params": int(pspec["n_params"]),
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics={
            "primary_variant": "vqc_n3_affine_cnot_fullherm",
            "oos_rank_ic_mean": prim_mn["oos_rank_ic_single_mean"],
            "oos_rank_ic_ens": prim_mn["oos_rank_ic_ens"],
            "oos_rank_ic_nw_pval": prim_mn["oos_rank_ic_nw_pval"],
            "oos_r2": prim_mn["oos_r2"],
            "n_params": prim_mn["n_params"],
            "depth": prim_mn["depth"],
            "n_2q_gates": prim_mn["n_2q_gates"],
            "q_mw_mean": prim_mn["q_mw_mean"],
            "entanglement_tests": ent_tests,
            "models": models_summary,
        },
    )


def run_suite_qcml_benchmark(
    out_dir: Path, preds_dir: Path, n_jobs: int, device_label: str
) -> dict[str, Any]:
    """Tier-4 matrix-form QCML dimension ladder, iso-parameter commuting/real-symmetric swaps,
    energy-anchoring ablations, news-permutation placebos, and unified Hansen MCS (Sec. 4).
    """
    t0 = time.perf_counter()
    _, panel = ensure_panels()
    ad = panel["active_dates"]
    seeds10 = list(SEEDS10)
    seeds5 = SEEDS10[:5]
    diag_betas = [0.05, 0.25, 1.0, 2.0, 4.0, 8.0, 16.0]
    single_thread = n_jobs > 1

    qcml_core_specs: list[tuple[str, str, int, float, float | None, list[int]]] = [
        ("qcml_herm_d2", "hermitian", 2, 0.05, None, seeds10),
        ("qcml_herm_d4", "hermitian", 4, 0.05, None, seeds10),
        ("qcml_herm_d8", "hermitian", 8, 0.05, None, seeds10),
        ("qcml_herm_d16", "hermitian", 16, 0.05, None, seeds10),
        ("qcml_realsym_d11", "real_sym", 11, 0.05, None, seeds10),
        ("qcml_herm_d8_le0", "hermitian", 8, 0.0, None, seeds10),
        ("qcml_realsym_d11_le0", "real_sym", 11, 0.0, None, seeds5),
    ]
    for b_val in diag_betas:
        qcml_core_specs.append((f"qcml_diag_d64_b{b_val:g}", "commuting_diag", 64, 0.05, b_val, seeds10))
    qcml_core_specs.append(("qcml_diag_d64_b2_le0", "commuting_diag", 64, 0.0, 2.0, seeds5))

    models_summary: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    cost_sweep_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    fold_metrics_out: dict[str, dict[str, Any]] = {"market_only": {}, "market_news": {}}
    opswap_grid: dict[str, Any] = {"market_only": {}, "market_news": {}}
    paired_out: dict[str, Any] = {"market_only": {}, "market_news": {}, "cross_modality": {}}
    daily_ic_csv_map: dict[str, np.ndarray] = {}
    daily_mse_csv_map: dict[str, np.ndarray] = {}
    equity_map: dict[str, dict[str, Any]] = {}
    uniq_days_ref: np.ndarray | None = None

    for mod in ("market_only", "market_news"):
        pooled = get_pooled_oos_meta(mod, "y_h5_cs", 5)
        uniq_days_ref = np.unique(pooled["di_te"])
        mod_tag = "mkt" if mod == "market_only" else "mn"

        fold_preds_store: dict[str, dict[tuple[int, int], np.ndarray]] = {}
        fold_val_store: dict[str, dict[int, float]] = {}

        for m_name, mode, dim, le, beta, m_seeds in qcml_core_specs:
            jobs = [
                (
                    mode,
                    dim,
                    le,
                    beta,
                    False,
                    None,
                    f,
                    mod,
                    "y_h5_cs",
                    5,
                    None,
                    False,
                    False,
                    sd,
                    False,
                    single_thread,
                )
                for sd in m_seeds
                for f in range(N_FOLDS)
            ]
            res_list = _run_jobs(_worker_qcml_job, jobs, n_jobs)
            preds, val_by_f = _assemble_seed_fold_preds(res_list, m_seeds)
            fold_preds_store[m_name] = {
                (int(r["seed"]), int(r["f"])): np.asarray(r["pred_te"], dtype=np.float64) for r in res_list
            }
            fold_val_store[m_name] = val_by_f

            ev = evaluate_predictions(
                preds,
                pooled["y_te"],
                pooled["di_te"],
                pooled["si_te"],
                pooled["tr_mean_per_row"],
                ortho_factors=pooled["ortho_te"],
                k_trials_dsr=40,
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
                horizon=5,
                exec_lag=1,
                compute_econ=True,
            )
            ev["n_params"] = int(res_list[0]["n_params"])
            ev["n_eff_params"] = int(res_list[0]["n_eff_params"])
            ev["kappa_nc_mean"] = round(float(np.mean([r["kappa_nc"] for r in res_list])), 6)
            ev["kappa_nc_mkt_news_mean"] = round(float(np.mean([r["kappa_nc_mkt_news"] for r in res_list])), 6)
            ev["norm_ratio_news_mkt_mean"] = round(float(np.mean([r["norm_ratio_news_mkt"] for r in res_list])), 6)
            ev["hermiticity_err_max"] = float(np.max([r["hermiticity_err"] for r in res_list]))
            ev["mean_val_mse"] = round(float(np.mean(list(val_by_f.values()))), 8)

            _save_pred_cache(
                preds_dir,
                f"qcml_{mod}_{m_name}",
                preds,
                {"daily_ic": ev["daily_ic"], "daily_mse": ev["daily_mse"]},
            )
            models_summary[mod][m_name] = _compact_eval(ev)
            cost_sweep_out[mod][m_name] = _compact_eval({"economics": ev["economics"]})["economics"]
            fold_metrics_out[mod][m_name] = {
                "single_seed_fold_ics": ev["single_seed"]["fold_rank_ics"],
                "ens_fold_ics": ev["ensemble"]["fold_rank_ics"],
                "val_mse_by_fold": {str(k): round(v, 8) for k, v in val_by_f.items()},
            }
            daily_ic_csv_map[f"{mod_tag}_{m_name}"] = ev["daily_ic"]
            daily_mse_csv_map[f"{mod_tag}_{m_name}"] = ev["daily_mse"]
            if m_name in ("qcml_herm_d8", "qcml_realsym_d11", "qcml_herm_d8_le0"):
                equity_map[f"{mod_tag}_{m_name}"] = ev["economics"]

        for sel_name, cand_names in [
            ("qcml_herm_dstar", ["qcml_herm_d2", "qcml_herm_d4", "qcml_herm_d8", "qcml_herm_d16"]),
            ("qcml_diag_d64_bstar", [f"qcml_diag_d64_b{b:g}" for b in diag_betas]),
        ]:
            chosen_per_fold = [min(cand_names, key=lambda cn: fold_val_store[cn][f]) for f in range(N_FOLDS)]
            seed_rows = [
                np.concatenate([fold_preds_store[chosen_per_fold[f]][(sd, f)] for f in range(N_FOLDS)], axis=0)
                for sd in seeds10
            ]
            preds_sel = np.stack(seed_rows, axis=0)
            ev_sel = evaluate_predictions(
                preds_sel,
                pooled["y_te"],
                pooled["di_te"],
                pooled["si_te"],
                pooled["tr_mean_per_row"],
                ortho_factors=pooled["ortho_te"],
                k_trials_dsr=40,
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
                horizon=5,
                exec_lag=1,
                compute_econ=True,
            )
            ev_sel["chosen_per_fold"] = chosen_per_fold
            ev_sel["n_params"] = int(models_summary[mod][chosen_per_fold[-1]]["n_params"])
            _save_pred_cache(
                preds_dir,
                f"qcml_{mod}_{sel_name}",
                preds_sel,
                {"daily_ic": ev_sel["daily_ic"], "daily_mse": ev_sel["daily_mse"]},
            )
            models_summary[mod][sel_name] = _compact_eval(ev_sel)
            cost_sweep_out[mod][sel_name] = _compact_eval({"economics": ev_sel["economics"]})["economics"]
            daily_ic_csv_map[f"{mod_tag}_{sel_name}"] = ev_sel["daily_ic"]
            daily_mse_csv_map[f"{mod_tag}_{sel_name}"] = ev_sel["daily_mse"]
            equity_map[f"{mod_tag}_{sel_name}"] = ev_sel["economics"]
            opswap_grid[mod][f"{sel_name}_chosen"] = chosen_per_fold

        c_mlv = _load_pred_cache(preds_dir, f"classical_{mod}_mom21_lowvol")
        c_h8 = _load_pred_cache(preds_dir, f"qcml_{mod}_qcml_herm_d8")
        if c_mlv is not None and c_h8 is not None:
            p_mlv = c_mlv["preds_seeds"][0].astype(np.float64)
            p_h8 = c_h8["preds_seeds"].mean(axis=0).astype(np.float64)
            di_te = pooled["di_te"]
            p_blend = np.empty_like(p_h8)
            for u in np.unique(di_te):
                m = di_te == u
                z1 = (p_mlv[m] - p_mlv[m].mean()) / (p_mlv[m].std() + 1e-12)
                z2 = (p_h8[m] - p_h8[m].mean()) / (p_h8[m].std() + 1e-12)
                p_blend[m] = 0.5 * (z1 + z2)
            ev_bl = evaluate_predictions(
                p_blend[None, :],
                pooled["y_te"],
                pooled["di_te"],
                pooled["si_te"],
                pooled["tr_mean_per_row"],
                ortho_factors=pooled["ortho_te"],
                k_trials_dsr=40,
                n_boot=2000,
                l_hac=L_HAC_DEFAULT,
                horizon=5,
                exec_lag=1,
                compute_econ=True,
            )
            models_summary[mod]["blend_50_50_momlv_herm_d8"] = _compact_eval(ev_bl)
            daily_ic_csv_map[f"{mod_tag}_blend_50_50_momlv_herm_d8"] = ev_bl["daily_ic"]

        if mod == "market_news":
            for sh_name, mode, dim, beta in [
                ("qcml_herm_d8_newsshuf", "hermitian", 8, None),
                ("qcml_diag_d64_b2_newsshuf", "commuting_diag", 64, 2.0),
            ]:
                jobs = [
                    (
                        mode,
                        dim,
                        0.05,
                        beta,
                        False,
                        None,
                        f,
                        "market_news",
                        "y_h5_cs",
                        5,
                        None,
                        False,
                        True,
                        sd,
                        False,
                        single_thread,
                    )
                    for sd in seeds5
                    for f in range(N_FOLDS)
                ]
                r_sh = _run_jobs(_worker_qcml_job, jobs, n_jobs)
                p_sh, _ = _assemble_seed_fold_preds(r_sh, seeds5)
                ev_sh = evaluate_predictions(
                    p_sh,
                    pooled["y_te"],
                    pooled["di_te"],
                    pooled["si_te"],
                    pooled["tr_mean_per_row"],
                    ortho_factors=pooled["ortho_te"],
                    n_boot=2000,
                    l_hac=L_HAC_DEFAULT,
                    compute_econ=False,
                )
                ev_sh["kappa_nc_mean"] = round(float(np.mean([r["kappa_nc"] for r in r_sh])), 6)
                ev_sh["norm_ratio_news_mkt_mean"] = round(float(np.mean([r["norm_ratio_news_mkt"] for r in r_sh])), 6)
                _save_pred_cache(
                    preds_dir,
                    f"qcml_market_news_{sh_name}",
                    p_sh,
                    {"daily_ic": ev_sh["daily_ic"], "daily_mse": ev_sh["daily_mse"]},
                )
                models_summary[mod][sh_name] = _compact_eval(ev_sh)
                daily_ic_csv_map[f"mn_{sh_name}"] = ev_sh["daily_ic"]
                daily_mse_csv_map[f"mn_{sh_name}"] = ev_sh["daily_mse"]

        contrasts = [
            ("herm_d8_vs_diag_bstar", "qcml_herm_d8", "qcml_diag_d64_bstar"),
            ("herm_d8_vs_realsym_d11", "qcml_herm_d8", "qcml_realsym_d11"),
            ("herm_d8_vs_le0", "qcml_herm_d8", "qcml_herm_d8_le0"),
            ("realsym_d11_vs_le0", "qcml_realsym_d11", "qcml_realsym_d11_le0"),
        ]
        if mod == "market_news":
            contrasts.append(("herm_d8_vs_newsshuf", "qcml_herm_d8", "qcml_herm_d8_newsshuf"))
            contrasts.append(
                (
                    "diag_b2_vs_newsshuf",
                    "qcml_diag_d64_b2" if "qcml_diag_d64_b2" in models_summary[mod] else "qcml_diag_d64_bstar",
                    "qcml_diag_d64_b2_newsshuf",
                )
            )

        p_list_names, p_list_raw = [], []
        for label, a_k, b_k in contrasts:
            if f"{mod_tag}_{a_k}" in daily_ic_csv_map and f"{mod_tag}_{b_k}" in daily_ic_csv_map:
                pt_ic = paired_tests(
                    daily_ic_csv_map[f"{mod_tag}_{a_k}"],
                    daily_ic_csv_map[f"{mod_tag}_{b_k}"],
                    n_boot=2000,
                    l_hac=L_HAC_DEFAULT,
                )
                pt_mse = paired_tests(
                    daily_mse_csv_map[f"{mod_tag}_{b_k}"],
                    daily_mse_csv_map[f"{mod_tag}_{a_k}"],
                    n_boot=2000,
                    l_hac=L_HAC_DEFAULT,
                )
                paired_out[mod][label] = {"ic": _json_safe(pt_ic), "mse_gain": _json_safe(pt_mse)}
                p_list_names.append(label)
                p_list_raw.append(pt_ic["nw_pval"])

        mcs_ic_all: dict[str, np.ndarray] = {
            k: -daily_ic_csv_map[f"{mod_tag}_{k}"]
            for k in (
                "qcml_herm_d2",
                "qcml_herm_d4",
                "qcml_herm_d8",
                "qcml_herm_d16",
                "qcml_herm_dstar",
                "qcml_realsym_d11",
                "qcml_diag_d64_bstar",
            )
        }
        mcs_mse_all: dict[str, np.ndarray] = {k: daily_mse_csv_map[f"{mod_tag}_{k}"] for k in mcs_ic_all}
        for ext_pfx, ext_models in [
            ("classical", ["mom21_lowvol", "ridge", "rff_krr", "hgbdt", "lgbm", "mlp_tanh", "film_mlp_iso"]),
            (
                "vqc",
                [
                    "vqc_n3_affine_cnot_fullherm",
                    "vqc_n3_affine_nocnot_fullherm",
                    "vqc_n3_cyclic_cnot_fullherm",
                    "vqc_n3_cyclic_nocnot_fullherm",
                ],
            ),
        ]:
            for em in ext_models:
                cached = _load_pred_cache(preds_dir, f"{ext_pfx}_{mod}_{em}")
                if cached is None or "daily_ic" not in cached:
                    continue
                mcs_ic_all[em] = -cached["daily_ic"]
                if em != "mom21_lowvol":
                    mcs_mse_all[em] = cached["daily_mse"]
                if em in (
                    "mom21_lowvol",
                    "hgbdt",
                    "mlp_tanh",
                    "rff_krr",
                    "film_mlp_iso",
                    "vqc_n3_affine_cnot_fullherm",
                ):
                    pt_ic = paired_tests(
                        daily_ic_csv_map[f"{mod_tag}_qcml_herm_d8"],
                        cached["daily_ic"],
                        n_boot=2000,
                        l_hac=L_HAC_DEFAULT,
                    )
                    pt_mse = paired_tests(
                        cached["daily_mse"],
                        daily_mse_csv_map[f"{mod_tag}_qcml_herm_d8"],
                        n_boot=2000,
                        l_hac=L_HAC_DEFAULT,
                    )
                    lbl = f"herm_d8_vs_{em}"
                    paired_out[mod][lbl] = {"ic": _json_safe(pt_ic), "mse_gain": _json_safe(pt_mse)}
                    p_list_names.append(lbl)
                    p_list_raw.append(pt_ic["nw_pval"])

        if p_list_raw:
            adj = holm_bonferroni(p_list_raw)
            for lbl, p_adj in zip(p_list_names, adj):
                paired_out[mod][lbl]["ic"]["holm_pval"] = round(float(p_adj), 6)

        opswap_grid[mod]["mcs_unified_ic"] = hansen_mcs_05(mcs_ic_all, n_boot=1000, block=10)
        opswap_grid[mod]["mcs_unified_mse"] = hansen_mcs_05(mcs_mse_all, n_boot=1000, block=10)

    for m_k in ("qcml_herm_d8", "qcml_herm_d16", "qcml_realsym_d11", "qcml_diag_d64_bstar"):
        pt_nl = paired_tests(
            daily_ic_csv_map[f"mn_{m_k}"], daily_ic_csv_map[f"mkt_{m_k}"], n_boot=2000, l_hac=L_HAC_DEFAULT
        )
        paired_out["cross_modality"][f"news_lift_{m_k}"] = _json_safe(pt_nl)

    assert uniq_days_ref is not None
    _export_daily_csv(out_dir / "daily_rank_ic.csv", ad, uniq_days_ref, daily_ic_csv_map)
    _export_daily_csv(out_dir / "daily_mse.csv", ad, uniq_days_ref, daily_mse_csv_map)
    _export_equity_curves_csv(out_dir / "equity_curves.csv", ad, equity_map)
    _write_json(out_dir / "cost_sweep.json", cost_sweep_out)
    _write_json(out_dir / "fold_metrics.json", fold_metrics_out)
    _write_json(out_dir / "opswap_grid.json", opswap_grid)
    _write_json(out_dir / "paired_tests_all.json", paired_out)

    h8_mn = models_summary["market_news"]["qcml_herm_d8"]
    in_mcs = bool(opswap_grid["market_news"]["mcs_unified_ic"].get("qcml_herm_d8", {}).get("in_mcs_05", False))
    swap_p = paired_out["market_news"]["herm_d8_vs_diag_bstar"]["ic"]["nw_pval"]
    swap_d = paired_out["market_news"]["herm_d8_vs_diag_bstar"]["ic"]["diff_mean"]

    return _write_suite_summary(
        out_dir,
        suite="qcml_benchmark",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "pytorch_exact_eigh",
            "device": device_label,
            "diff_method": "tikhonov_eigh+daleckii_krein",
            "shots": None,
            "noise_model": "ideal",
            "n_qubits": 0,
            "depth": 0,
            "n_2q_gates": 0,
            "n_params": 1216,
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics={
            "primary_model": "qcml_herm_d8",
            "oos_rank_ic_mean": h8_mn["oos_rank_ic_single_mean"],
            "oos_rank_ic_seed_std": h8_mn["oos_rank_ic_seed_std"],
            "oos_rank_ic_ens": h8_mn["oos_rank_ic_ens"],
            "oos_rank_ic_nw_pval": h8_mn["oos_rank_ic_nw_pval"],
            "oos_r2": h8_mn["oos_r2"],
            "oos_r2_ens": h8_mn["oos_r2_ens"],
            "in_mcs_05_ic": int(in_mcs),
            "kappa_nc_mn_d8": h8_mn["kappa_nc_mean"],
            "opswap_d8_vs_diag_bstar_delta": swap_d,
            "opswap_d8_vs_diag_bstar_pval": swap_p,
            "news_lift_d8_delta": paired_out["cross_modality"]["news_lift_qcml_herm_d8"]["diff_mean"],
            "news_lift_d8_pval": paired_out["cross_modality"]["news_lift_qcml_herm_d8"]["nw_pval"],
            "models": models_summary,
            "paired_tests": paired_out,
            "mcs_summary": {
                m: {k: opswap_grid[m][k] for k in ("mcs_unified_ic", "mcs_unified_mse")}
                for m in ("market_only", "market_news")
            },
        },
    )


def _build_ood2022_fold_dict(modality: str, winsor: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pre-crisis training/validation split ending strictly before the February 2022 window Q_{OOD} (Sec. 4.3)."""
    ood = get_ood2022_data(modality, winsor=winsor)
    d_tr = ood["di_tr"]
    u_tr = np.unique(d_tr)
    val_days = u_tr[-42:]
    v_min = int(val_days.min())
    m_tr = d_tr <= (v_min - 6)
    m_va = np.isin(d_tr, val_days)
    fd = {
        "X_tr": ood["X_tr"][m_tr],
        "y_tr": ood["y_tr"][m_tr],
        "di_tr": d_tr[m_tr],
        "si_tr": ood["si_tr"][m_tr],
        "X_va": ood["X_tr"][m_va],
        "y_va": ood["y_tr"][m_va],
        "di_va": d_tr[m_va],
        "si_va": ood["si_tr"][m_va],
        "X_te": ood["X_ev"],
        "y_te": ood["y_ev"],
        "di_te": ood["di_ev"],
        "si_te": ood["si_ev"],
    }
    return fd, ood


def run_suite_spectral_stress(out_dir: Path, n_jobs: int, device_label: str) -> dict[str, Any]:
    """Norm-controlled spectral volatility audit, GUE random-matrix null, and 2022 OOD stress (Sec. 4.2, Sec. 4.3)."""
    t0 = time.perf_counter()
    _, panel = ensure_panels()
    ad = panel["active_dates"]
    spectral_seeds = [101, 102, 103]
    single_thread = n_jobs > 1

    spectral_regimes = [
        ("qcml_sup_le005", "hermitian", 8, 0.05, None, None),
        ("qcml_unsup_le100", "hermitian", 8, 1.0, None, None),
        ("qcml_candelori_w01", "hermitian", 8, 1.0, None, 0.1),
    ]

    b4_all: dict[str, Any] = {}
    ood_all: dict[str, Any] = {}
    granger_all: dict[str, Any] = {}
    daily_csv_series: dict[str, np.ndarray] = {}
    uniq_days_ref: np.ndarray | None = None

    for mod in ("market_only", "market_news"):
        mod_tag = "mkt" if mod == "market_only" else "mn"
        pooled = get_pooled_oos_meta(mod, "y_h5_cs", 5)
        uniq_days_ref = np.unique(pooled["di_te"])

        spectral_by_model: dict[str, dict[str, np.ndarray]] = {}
        train_spectral_by_model: dict[str, dict[int, dict[str, np.ndarray]]] = {}

        for r_name, mode, dim, le, beta, cw in spectral_regimes:
            jobs = [
                (
                    mode,
                    dim,
                    le,
                    beta,
                    False,
                    cw,
                    f,
                    mod,
                    "y_h5_cs",
                    5,
                    None,
                    False,
                    False,
                    sd,
                    True,
                    single_thread,
                )
                for sd in spectral_seeds
                for f in range(N_FOLDS)
            ]
            res_list = _run_jobs(_worker_qcml_job, jobs, n_jobs)
            by_sf = {(int(r["seed"]), int(r["f"])): r for r in res_list}
            conc_keys = ("E0", "E0_shifted", "dE", "dE_tilde", "fubini_study_step", "E0_sphere", "dE_sphere")
            spectral_by_model[r_name] = {
                ck: np.concatenate(
                    [np.mean([by_sf[(sd, f)]["spectral_te"][ck] for sd in spectral_seeds], axis=0) for f in range(N_FOLDS)],
                    axis=0,
                )
                for ck in conc_keys
            }
            train_spectral_by_model[r_name] = {
                f: {
                    tk: np.mean([by_sf[(sd, f)]["spectral_tr_daily"][tk] for sd in spectral_seeds], axis=0)
                    for tk in ("E0", "dE", "norm", "norm2")
                }
                for f in range(N_FOLDS)
            }

        fd_ood_raw, ood_raw = _build_ood2022_fold_dict(mod, winsor=False)
        ood_win = get_ood2022_data(mod, winsor=True)
        X_ev_raw = ood_raw["X_ev"]
        X_ev_win = ood_win["X_ev"]

        ood_preds_raw: dict[str, np.ndarray] = {}
        ood_preds_win: dict[str, np.ndarray] = {}
        ood_detectors: dict[str, np.ndarray] = {}

        for cm in ("ridge", "hgbdt", "mlp_relu", "mlp_tanh", "film_mlp_iso"):
            r_cm = fit_predict_classical(cm, fd_ood_raw, seed=101)
            if not r_cm.get("skipped"):
                ood_preds_raw[cm] = r_cm["ood_fn"](X_ev_raw)["pred"]
                ood_preds_win[cm] = r_cm["ood_fn"](X_ev_win)["pred"]

        for v_key, ent_flag in [
            ("vqc_n3_affine_cnot_fullherm", True),
            ("vqc_n3_affine_nocnot_fullherm", False),
        ]:
            r_vq = fit_predict_vqc("affine", 3, 9, ent_flag, "full_hermitian", fd_ood_raw, seed=101)
            o_raw = r_vq["ood_fn"](X_ev_raw)
            o_win = r_vq["ood_fn"](X_ev_win)
            ood_preds_raw[v_key] = o_raw["pred"]
            ood_preds_win[v_key] = o_win["pred"]
            ood_detectors[f"{v_key}_infidelity"] = o_raw["infidelity"]
            ood_detectors[f"{v_key}_rho_infidelity"] = o_raw["rho_infidelity"]

        for q_key, q_mode, q_d, q_le, q_b in [
            ("qcml_herm_d8", "hermitian", 8, 0.05, None),
            ("qcml_realsym_d11", "real_sym", 11, 0.05, None),
            ("qcml_diag_d64_b2", "commuting_diag", 64, 0.05, 2.0),
            ("qcml_unsup_d8_le100", "hermitian", 8, 1.0, None),
        ]:
            r_q = fit_predict_qcml(q_mode, q_d, fd_ood_raw, seed=101, lambda_e=q_le, beta=q_b)
            st = r_q["model_state"]
            obs_r = extract_spectral_observables(st, X_ev_raw)
            obs_w = extract_spectral_observables(st, X_ev_win)
            if q_le < 0.9999:
                ood_preds_raw[q_key] = obs_r["pred"]
                ood_preds_win[q_key] = obs_w["pred"]
            ood_detectors[f"{q_key}_E0"] = obs_r["E0"]
            ood_detectors[f"{q_key}_dE"] = obs_r["dE"]

        b4_res = evaluate_spectral_stress(
            spectral_by_model=spectral_by_model,
            X_test_concat=pooled["X_te"],
            di_test_concat=pooled["di_te"],
            fold_test_concat=pooled["fold_te"],
            modality=mod,
            train_spectral_by_model=train_spectral_by_model,
            gue_d=8,
            gue_n_draws=10,
            ood2022_preds=ood_preds_raw,
            n_boot=1000,
        )

        ood_win_bounds: dict[str, Any] = {}
        y_ev_o = ood_raw["y_ev"].astype(np.float64)
        reg_o = ood_raw["regime_ev"]
        tr_mu_o = float(ood_raw["tr_mean"])
        ok_y = np.isfinite(y_ev_o)
        for mk, p_w in ood_preds_win.items():
            per_r = {}
            for lab, mr in [
                ("normal", ok_y & (reg_o == 0)),
                ("acute", ok_y & (reg_o == 1)),
                ("post_reopen", ok_y & (reg_o == 2)),
            ]:
                if int(mr.sum()) < 20:
                    continue
                sse = float(np.sum((y_ev_o[mr] - p_w[mr]) ** 2))
                sse_b = float(np.sum((y_ev_o[mr] - tr_mu_o) ** 2))
                per_r[lab] = {
                    "r2_vs_precrisis_train_mean": round(1.0 - sse / max(sse_b, 1e-18), 6),
                    "rmse": round(math.sqrt(sse / int(mr.sum())), 6),
                    "max_abs_pred": round(float(np.max(np.abs(p_w[mr]))), 5),
                }
            ood_win_bounds[mk] = per_r

        d_ev_o = ood_raw["di_ev"]
        u_o = np.unique(d_ev_o)
        reg_d = np.array([int(reg_o[d_ev_o == d][0]) for d in u_o])
        m01 = (reg_d == 0) | (reg_d == 1)
        y01 = reg_d[m01] == 1
        ood_detectors["norm2"] = np.sum(X_ev_raw.astype(np.float64) ** 2, axis=1)
        ood_acute_auroc = {
            dk: round(float(roc_auc_score(y01, np.array([float(np.median(arr[d_ev_o == d])) for d in u_o])[m01])), 5)
            for dk, arr in ood_detectors.items()
        }

        for sk, sarr in b4_res["daily_series"].items():
            if sk != "cal_idx":
                daily_csv_series[f"{mod_tag}_{sk}"] = sarr

        if mod == "market_news":
            x_te_64 = pooled["X_te"].astype(np.float64)
            di_te = pooled["di_te"]
            day_slices = [np.flatnonzero(di_te == u) for u in uniq_days_ref]
            n2_row = np.sum(x_te_64**2, axis=1)
            scale_row = np.sqrt(1.0 + n2_row)
            n1_day = np.array([float(np.sqrt(n2_row[ix]).mean()) for ix in day_slices], dtype=np.float64)
            n_draws, d_g, p_g = 10, 8, x_te_64.shape[1]
            e0_draws = np.empty((n_draws, len(day_slices)), dtype=np.float64)
            de_draws = np.empty((n_draws, len(day_slices)), dtype=np.float64)
            det_draws = np.empty((n_draws, len(day_slices)), dtype=np.float64)
            for s_idx in range(n_draws):
                rng_g = np.random.default_rng(1000 + s_idx)
                re_p = rng_g.normal(scale=1.0 / math.sqrt(2.0 * d_g), size=(p_g, d_g, d_g))
                im_p = rng_g.normal(scale=1.0 / math.sqrt(2.0 * d_g), size=(p_g, d_g, d_g))
                a_ops = 0.5 * (re_p + np.swapaxes(re_p, -1, -2)) + 1j * 0.5 * (im_p - np.swapaxes(im_p, -1, -2))
                c0 = 0.5 * np.sum(a_ops @ a_ops, axis=0)
                e0_row = np.empty(len(x_te_64), dtype=np.float64)
                de_row = np.empty(len(x_te_64), dtype=np.float64)
                for i0 in range(0, len(x_te_64), 4096):
                    xb = x_te_64[i0 : i0 + 4096]
                    w = np.linalg.eigvalsh(c0[None, :, :] - np.tensordot(xb, a_ops, axes=([1], [0])))
                    e0_row[i0 : i0 + 4096] = w[:, 0] + 0.5 * n2_row[i0 : i0 + 4096]
                    de_row[i0 : i0 + 4096] = w[:, 1] - w[:, 0]
                det_row = de_row / scale_row
                e0_draws[s_idx] = [float(e0_row[ix].mean()) for ix in day_slices]
                de_draws[s_idx] = [float(de_row[ix].mean()) for ix in day_slices]
                det_draws[s_idx] = [float(det_row[ix].mean()) for ix in day_slices]
            stress_idx = np.flatnonzero(np.asarray(daily_csv_series["mn_stress_contemp"], dtype=np.int64) == 1)
            x_stress_dir = np.mean([x_te_64[day_slices[i]].mean(axis=0) for i in stress_idx], axis=0)
            x_hat = x_stress_dir / float(np.linalg.norm(x_stress_dir))
            np.savez_compressed(
                out_dir / "gue_daily_null.npz",
                n1_day=n1_day,
                e0_draws=e0_draws,
                de_draws=de_draws,
                det_draws=det_draws,
                x_hat=x_hat,
            )

        b4_all[mod] = _json_safe({k: v for k, v in b4_res.items() if k != "daily_series"})
        ood_all[mod] = _json_safe(
            {
                "raw_unclipped_bounds": b4_res["ood2022_bounds"],
                "winsorized_bounds": ood_win_bounds,
                "acute_13d_vs_normal_58d_auroc": ood_acute_auroc,
            }
        )
        granger_all[mod] = _json_safe(
            {
                rn: {
                    "granger_dE_raw": b4_res["models"][rn]["granger_dE_raw"],
                    "granger_dE_norm_controlled": b4_res["models"][rn]["granger_dE_norm_controlled"],
                    "granger_dE_perp_norm_controlled": b4_res["models"][rn]["granger_dE_perp_norm_controlled"],
                    "granger_E0_shifted_norm_controlled": b4_res["models"][rn]["granger_E0_shifted_norm_controlled"],
                }
                for rn in b4_res["models"]
            }
        )

    assert uniq_days_ref is not None
    _export_daily_csv(out_dir / "daily_regime_series.csv", ad, uniq_days_ref, daily_csv_series)
    _write_json(out_dir / "granger_var5.json", granger_all)
    _write_json(out_dir / "roc_and_spectral.json", b4_all)
    _write_json(out_dir / "ood_2022_bounds.json", ood_all)

    mn_sup = b4_all["market_news"]["models"]["qcml_sup_le005"]
    mn_uns = b4_all["market_news"]["models"]["qcml_unsup_le100"]
    mn_base = b4_all["market_news"]["baselines"]

    return _write_suite_summary(
        out_dir,
        suite="spectral_stress",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "pytorch_exact_eigh+scipy",
            "device": device_label,
            "diff_method": "tikhonov_eigh",
            "shots": None,
            "noise_model": "ideal",
            "n_qubits": 0,
            "depth": 0,
            "n_2q_gates": 0,
            "n_params": 1216,
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics={
            "mn_norm2_auroc_contemp": mn_base["norm2"]["auroc_contemp"],
            "mn_garch11_auroc_contemp": mn_base["garch11"]["auroc_contemp"],
            "mn_sup_E0_auroc_contemp": mn_sup["E0"]["auroc_contemp"],
            "mn_sup_E0_perp_auroc_contemp": mn_sup["E0_perp_causal"]["auroc_contemp"],
            "mn_sup_dE_auroc_contemp": mn_sup["dE"]["auroc_contemp"],
            "mn_sup_dE_perp_auroc_contemp": mn_sup["dE_perp_causal"]["auroc_contemp"],
            "mn_sup_granger_raw_pval": mn_sup["granger_dE_raw"]["fwd_pval"],
            "mn_sup_granger_normctrl_pval": mn_sup["granger_dE_norm_controlled"]["fwd_pval"],
            "mn_unsup_E0_auroc_contemp": mn_uns["E0"]["auroc_contemp"],
            "mn_unsup_dE_auroc_contemp": mn_uns["dE"]["auroc_contemp"],
            "mn_unsup_dE_perp_auroc_contemp": mn_uns["dE_perp_causal"]["auroc_contemp"],
            "mn_unsup_granger_raw_pval": mn_uns["granger_dE_raw"]["fwd_pval"],
            "mn_unsup_granger_normctrl_pval": mn_uns["granger_dE_norm_controlled"]["fwd_pval"],
            "mn_gue_759d_r2_dE_norm2": b4_all["market_news"]["gue_null_759d"]["r2_dE_on_norm2_daily_mean"],
            "b4": b4_all,
            "ood_2022": ood_all,
        },
    )


def run_suite_multioutput_robustness(
    out_dir: Path, preds_dir: Path, n_jobs: int, device_label: str
) -> dict[str, Any]:
    """Supplementary S2 multi-task joint-state benchmark (q=4 vs 4x q=1) and Supplementary S3 robustness grid."""
    t0 = time.perf_counter()
    _, panel = ensure_panels()
    seeds_b5 = SEEDS10[:5]
    single_thread = n_jobs > 1
    t4_names = ["y_h5_cs", "fwd_vol_5d", "fwd_dd_5d", "fwd_vol_shock_5d"]

    pooled_t4 = get_pooled_oos_meta("market_news", "Y_state", 5)
    t4_out: dict[str, Any] = {}

    for m_key, kind in [
        ("ridge_multi", "classic"),
        ("mlp_tanh_multi", "classic"),
        ("qcml_herm_d8_t4", "qcml"),
    ]:
        m_seeds = [101] if m_key == "ridge_multi" else seeds_b5
        if kind == "classic":
            jobs = [
                (m_key, f, "market_news", "Y_state", 5, None, False, False, sd, single_thread)
                for sd in m_seeds
                for f in range(N_FOLDS)
            ]
            r_list = _run_jobs(_worker_classical_job, jobs, n_jobs)
        else:
            jobs = [
                (
                    "hermitian",
                    8,
                    0.05,
                    None,
                    True,
                    None,
                    f,
                    "market_news",
                    "Y_state",
                    5,
                    None,
                    False,
                    False,
                    sd,
                    False,
                    single_thread,
                )
                for sd in m_seeds
                for f in range(N_FOLDS)
            ]
            r_list = _run_jobs(_worker_qcml_job, jobs, n_jobs)
        preds_q, _ = _assemble_seed_fold_preds(r_list, m_seeds)
        per_t: dict[str, Any] = {}
        for j, tn in enumerate(t4_names):
            ev_j = evaluate_predictions(
                preds_q[:, :, j],
                pooled_t4["y_te"][:, j],
                pooled_t4["di_te"],
                pooled_t4["si_te"],
                pooled_t4["tr_mean_per_row"][:, j],
                ortho_factors=pooled_t4["ortho_te"],
                n_boot=1000,
                l_hac=L_HAC_DEFAULT,
                compute_econ=False,
            )
            per_t[tn] = _compact_eval(ev_j)
        t4_out[m_key] = {
            "n_params": int(r_list[0]["n_params"]),
            "per_target": per_t,
            "mean_single_seed_rank_ic": round(
                float(np.mean([per_t[t]["oos_rank_ic_single_mean"] for t in t4_names])), 6
            ),
            "mean_single_seed_r2": round(float(np.mean([per_t[t]["oos_r2"] for t in t4_names])), 6),
        }

    jobs_4x1 = [
        (j, f, sd, single_thread)
        for j in range(len(t4_names))
        for sd in seeds_b5
        for f in range(N_FOLDS)
    ]
    res_4x1 = _run_jobs(_worker_qcml_t4_single_job, jobs_4x1, n_jobs)
    by_jsf = {(r["col_j"], r["seed"], r["f"]): r["pred_te"] for r in res_4x1}
    n_par_4x1 = 4 * int(res_4x1[0]["n_params"]) if res_4x1 else 0
    per_t_single: dict[str, Any] = {}
    for j, tn in enumerate(t4_names):
        sd_preds_j = [np.concatenate([by_jsf[(j, sd, f)] for f in range(N_FOLDS)], axis=0) for sd in seeds_b5]
        ev_j = evaluate_predictions(
            np.stack(sd_preds_j, axis=0),
            pooled_t4["y_te"][:, j],
            pooled_t4["di_te"],
            pooled_t4["si_te"],
            pooled_t4["tr_mean_per_row"][:, j],
            ortho_factors=pooled_t4["ortho_te"],
            n_boot=1000,
            l_hac=L_HAC_DEFAULT,
            compute_econ=False,
        )
        per_t_single[tn] = _compact_eval(ev_j)
    t4_out["qcml_herm_d8_4x_q1"] = {
        "n_params": int(n_par_4x1),
        "per_target": per_t_single,
        "mean_single_seed_rank_ic": round(
            float(np.mean([per_t_single[t]["oos_rank_ic_single_mean"] for t in t4_names])), 6
        ),
        "mean_single_seed_r2": round(float(np.mean([per_t_single[t]["oos_r2"] for t in t4_names])), 6),
    }

    rob_grid: dict[str, Any] = {}
    for r_label, tgt, hor, tw in [
        ("h15_purged_gap15", "y_h15_cs", 15, None),
        ("w126_rolling_train", "y_h5_cs", 5, 126),
    ]:
        pooled_r = get_pooled_oos_meta("market_news", tgt, hor, train_window=tw)
        l_hac_r = 23 if hor == 15 else L_HAC_DEFAULT
        rob_grid[r_label] = {}
        for m_key, kind in [
            ("ridge", "classic"),
            ("hgbdt", "classic"),
            ("mlp_tanh", "classic"),
            ("qcml_herm_d8", "qcml"),
        ]:
            m_seeds = [101] if m_key in ("ridge", "hgbdt") else seeds_b5
            if kind == "classic":
                jobs = [
                    (m_key, f, "market_news", tgt, hor, tw, False, False, sd, single_thread)
                    for sd in m_seeds
                    for f in range(N_FOLDS)
                ]
                r_list = _run_jobs(_worker_classical_job, jobs, n_jobs)
            else:
                jobs = [
                    (
                        "hermitian",
                        8,
                        0.05,
                        None,
                        False,
                        None,
                        f,
                        "market_news",
                        tgt,
                        hor,
                        tw,
                        False,
                        False,
                        sd,
                        False,
                        single_thread,
                    )
                    for sd in m_seeds
                    for f in range(N_FOLDS)
                ]
                r_list = _run_jobs(_worker_qcml_job, jobs, n_jobs)
            preds_r, _ = _assemble_seed_fold_preds(r_list, m_seeds)
            ev_r = evaluate_predictions(
                preds_r,
                pooled_r["y_te"],
                pooled_r["di_te"],
                pooled_r["si_te"],
                pooled_r["tr_mean_per_row"],
                ortho_factors=pooled_r["ortho_te"],
                n_boot=1000,
                l_hac=l_hac_r,
                horizon=hor,
                compute_econ=False,
            )
            rob_grid[r_label][m_key] = _compact_eval(ev_r)

    pooled_mn = get_pooled_oos_meta("market_news", "y_h5_cs", 5)
    di_mn = pooled_mn["di_te"]
    si_mn = pooled_mn["si_te"]
    y_mn = pooled_mn["y_te"]
    trm_mn = pooled_mn["tr_mean_per_row"]
    top30_mask_rows = panel["top30_mask"][di_mn, si_mn]
    exdiv_mask_rows = panel["exdiv_sens_mask"][di_mn, si_mn]

    liq_and_cohorts: dict[str, Any] = {}
    surv_out: dict[str, Any] = {}
    for m_key, pfx in [
        ("mom21_lowvol", "classical"),
        ("ridge", "classical"),
        ("rff_krr", "classical"),
        ("hgbdt", "classical"),
        ("mlp_tanh", "classical"),
        ("qcml_herm_d8", "qcml"),
        ("qcml_realsym_d11", "qcml"),
    ]:
        c = _load_pred_cache(preds_dir, f"{pfx}_market_news_{m_key}")
        if c is None:
            continue
        p_ens = c["preds_seeds"].mean(axis=0).astype(np.float64)
        ev_t30 = evaluate_predictions(
            p_ens[top30_mask_rows][None, :],
            y_mn[top30_mask_rows],
            di_mn[top30_mask_rows],
            si_mn[top30_mask_rows],
            trm_mn[top30_mask_rows],
            n_boot=50,
            compute_econ=False,
        )
        ev_3150 = evaluate_predictions(
            p_ens[~top30_mask_rows][None, :],
            y_mn[~top30_mask_rows],
            di_mn[~top30_mask_rows],
            si_mn[~top30_mask_rows],
            trm_mn[~top30_mask_rows],
            n_boot=50,
            compute_econ=False,
        )
        m_clean = ~exdiv_mask_rows
        ev_exdiv = evaluate_predictions(
            p_ens[m_clean][None, :],
            y_mn[m_clean],
            di_mn[m_clean],
            si_mn[m_clean],
            trm_mn[m_clean],
            n_boot=50,
            compute_econ=False,
        )
        u_days = np.unique(di_mn)
        coh_t = []
        for c0 in range(5):
            c_days = u_days[c0::5]
            mc = np.isin(di_mn, c_days)
            ev_c = evaluate_predictions(
                p_ens[mc][None, :], y_mn[mc], di_mn[mc], si_mn[mc], trm_mn[mc], l_hac=2, n_boot=50, compute_econ=False
            )
            coh_t.append(round(float(ev_c["ensemble"]["rank_ic_nw_t"]), 4))
        liq_and_cohorts[m_key] = {
            "top30_ens_ic": ev_t30["oos_rank_ic_ens"],
            "ranks_31_50_ens_ic": ev_3150["oos_rank_ic_ens"],
            "bluechip_exdiv_sens_ens_ic": ev_exdiv["oos_rank_ic_ens"],
            "weekly_5cohort_tstats": coh_t,
            "weekly_5cohort_mean_t": round(float(np.mean(coh_t)), 4),
        }
        if m_key in ("mom21_lowvol", "hgbdt", "mlp_tanh", "rff_krr", "qcml_herm_d8"):
            surv_out[m_key] = survivorship_stress_bound(
                p_ens, y_mn, di_mn, si_mn, k_grid=(3, 5, 7), n_mc=50, l_hac=L_HAC_DEFAULT
            )

    _write_json(out_dir / "t4_multioutput.json", t4_out)
    _write_json(out_dir / "robustness_grid.json", rob_grid)
    _write_json(out_dir / "survivorship_bounds.json", surv_out)
    _write_json(out_dir / "liquidity_and_cohorts.json", liq_and_cohorts)

    t4_q = t4_out["qcml_herm_d8_t4"]
    h15_q = rob_grid["h15_purged_gap15"]["qcml_herm_d8"]
    return _write_suite_summary(
        out_dir,
        suite="multioutput_robustness",
        wall_time_s=time.perf_counter() - t0,
        backend={
            "framework": "pytorch_exact_eigh+scikit-learn",
            "device": device_label,
            "diff_method": "tikhonov_eigh",
            "shots": None,
            "noise_model": "ideal",
            "n_qubits": 0,
            "depth": 0,
            "n_2q_gates": 0,
            "n_params": 1408,
        },
        seeds={"data_split": DATA_SPLIT_SEED, "param_init": 101, "shot_sampler": None, "noise_channel": None},
        metrics={
            "t4_qcml_mean_ic": t4_q["mean_single_seed_rank_ic"],
            "t4_qcml_mean_r2": t4_q["mean_single_seed_r2"],
            "t4_4xq1_mean_ic": t4_out["qcml_herm_d8_4x_q1"]["mean_single_seed_rank_ic"],
            "t4_mlp_multi_mean_ic": t4_out["mlp_tanh_multi"]["mean_single_seed_rank_ic"],
            "h15_purged_qcml_ic": h15_q["oos_rank_ic_single_mean"],
            "h15_purged_hgbdt_ic": rob_grid["h15_purged_gap15"]["hgbdt"]["oos_rank_ic_single_mean"],
            "w126_qcml_ic": rob_grid["w126_rolling_train"]["qcml_herm_d8"]["oos_rank_ic_single_mean"],
            "t4_multioutput": t4_out,
            "robustness_grid": rob_grid,
            "liquidity_and_cohorts": liq_and_cohorts,
            "survivorship_bounds": surv_out,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--suite",
        default="all",
        help=f"Comma-separated suites from {{{','.join(ALL_SUITES)}}} or 'all'.",
    )
    ap.add_argument("--jobs", type=int, default=1, help="Worker processes across (fold, seed) fits.")
    ap.add_argument("--out-dir", type=Path, default=RUNS_DIR, help="Output directory (default: data/runs).")
    args = ap.parse_args()

    if args.suite.strip().lower() == "all":
        suites = list(ALL_SUITES)
    else:
        suites = [s.strip().lower() for s in args.suite.split(",") if s.strip()]
        invalid = [s for s in suites if s not in ALL_SUITES]
        if invalid:
            raise ValueError(f"Unknown --suite entries {invalid}; expected {ALL_SUITES}")

    n_jobs = max(1, int(args.jobs))
    torch.set_num_threads(1)
    ensure_panels()
    device_label = _device_label()
    preds_dir = PREDS_CACHE_DIR
    preds_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        suite_dir = args.out_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{suite}] running (jobs={n_jobs})...", flush=True)
        if suite == "synthetic_controls":
            s = run_suite_synthetic_controls(suite_dir, device_label)
        elif suite == "classical_controls":
            s = run_suite_classical_controls(suite_dir, preds_dir, n_jobs, device_label)
        elif suite == "vqc_controls":
            s = run_suite_vqc_controls(suite_dir, preds_dir, n_jobs, device_label)
        elif suite == "qcml_benchmark":
            s = run_suite_qcml_benchmark(suite_dir, preds_dir, n_jobs, device_label)
        elif suite == "spectral_stress":
            s = run_suite_spectral_stress(suite_dir, n_jobs, device_label)
        elif suite == "multioutput_robustness":
            s = run_suite_multioutput_robustness(suite_dir, preds_dir, n_jobs, device_label)
        else:
            raise ValueError(suite)
        print(f"[{suite}] finished in {s['wall_time_s']:.2f}s -> {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
