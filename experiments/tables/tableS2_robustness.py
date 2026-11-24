"""Table S2: Robustness audit on MOEX M+N: liquidity tiers, ex-dividend masking, five-cohort weekly
t-statistics, and 2022-2024 vs. 2024-2025 subperiod splits (Panel A); survivorship bounds and
pre-2020 continuous listings (Panel B); and strictly purged h = 15 and rolling W_train = 126d
evaluations (Panel C; Supplementary S3).

Usage: python -I experiments/tables/tableS2_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_suite_csv, load_suite_metrics


def _subperiod_means(suite: str, col: str, cut_date: str = "2024-04-01") -> tuple[float, float]:
    rows = load_suite_csv(suite, "daily_rank_ic.csv")
    early = [float(r[col]) for r in rows if r["date"] < cut_date]
    late = [float(r[col]) for r in rows if r["date"] >= cut_date]
    return sum(early) / len(early), sum(late) / len(late)


def main() -> None:
    rob = load_suite_metrics("multioutput_robustness")
    liq = rob["liquidity_and_cohorts"]
    surv = rob["survivorship_bounds"]
    grid = rob["robustness_grid"]

    print("[Panel A: Liquidity Tiers, Ex-Dividend Masking, Weekly Cohorts, and Subperiod Split]")
    hdr_a = (
        f"{'Model':<20}  {'Top-30':>7}  {'31-50':>7}  {'Ex-Div':>7}  "
        f"{'t_cohort [Min-Max]':>20}  {'2022-24 (505d)':>14}  {'2024-25 (254d)':>14}"
    )
    print(hdr_a)
    print("-" * len(hdr_a))

    panel_a = [
        ("Mom21-LowVol", "mom21_lowvol", "classical_controls", "mn_mom21_lowvol"),
        ("QCML Real (d=11)", "qcml_realsym_d11", "qcml_benchmark", "mn_qcml_realsym_d11"),
        ("QCML Herm (d=8)", "qcml_herm_d8", "qcml_benchmark", "mn_qcml_herm_d8"),
        ("MLP-tanh", "mlp_tanh", "classical_controls", "mn_mlp_tanh"),
        ("HistGBDT", "hgbdt", "classical_controls", "mn_hgbdt"),
        ("RFF-KRR", "rff_krr", "classical_controls", "mn_rff_krr"),
        ("Ridge", "ridge", "classical_controls", "mn_ridge"),
    ]
    for label, key, csv_suite, csv_col in panel_a:
        st = liq[key]
        ts = st["weekly_5cohort_tstats"]
        t_str = f"{st['weekly_5cohort_mean_t']:.2f} [{min(ts):.2f}-{max(ts):.2f}]"
        e_ic, l_ic = _subperiod_means(csv_suite, csv_col)
        print(
            f"{label:<20}  {st['top30_ens_ic']:+7.4f}  {st['ranks_31_50_ens_ic']:+7.4f}  "
            f"{st['bluechip_exdiv_sens_ens_ic']:+7.4f}  {t_str:>20}  {e_ic:+14.4f}  {l_ic:+14.4f}"
        )

    print("\n[Panel B: Survivorship Stress Bounds (k in {3,5,7} Exits/Fold) and Pre-2020 Continuous Subset]")
    hdr_b = (
        f"{'Model':<20}  {'Base':>8}  {'k=3 Mean (p05)':>18}  {'k=5 Mean (p05)':>18}  "
        f"{'k=7 Mean (p05; d)':>26}  {'Pre-2020 (SE_NW)':>18}"
    )
    print(hdr_b)
    print("-" * len(hdr_b))
    for label, key in [
        ("Mom21-LowVol", "mom21_lowvol"),
        ("QCML Herm (d=8)", "qcml_herm_d8"),
        ("MLP-tanh", "mlp_tanh"),
        ("HistGBDT", "hgbdt"),
        ("RFF-KRR", "rff_krr"),
    ]:
        sv = surv[key]
        ds = sv["distressed_exit_stress"]
        k3, k5, k7 = ds["k_3_exits_per_fold"], ds["k_5_exits_per_fold"], ds["k_7_exits_per_fold"]
        p20 = sv["pre2020_continuous"]
        k3_s = f"{k3['rank_ic_mean']:+.5f} ({k3['rank_ic_p05']:.5f})"
        k5_s = f"{k5['rank_ic_mean']:+.5f} ({k5['rank_ic_p05']:.5f})"
        k7_s = f"{k7['rank_ic_mean']:+.5f} ({k7['rank_ic_p05']:.5f}; {k7['delta_vs_base']:+.5f})"
        p20_s = f"{p20['rank_ic']:+.5f} ({p20['nw_se']:.5f})"
        print(f"{label:<20}  {sv['baseline_rank_ic']:+8.5f}  {k3_s:>18}  {k5_s:>18}  {k7_s:>26}  {p20_s:>18}")

    print("\n[Panel C: Strictly Purged Horizon (h=15, g_purge=15d) and Rolling Window (W_train=126d)]")
    hdr_c = (
        f"{'Regime':<13}  {'Model':<18}  {'Seeds':>5}  {'1s IC +- sd':>16}  "
        f"{'Ens IC (t_NW)':>16}  {'1s R2':>9}  {'Ens R2':>9}  {'FM IC (p_NW)':>16}  {'Win':>4}"
    )
    print(hdr_c)
    print("-" * len(hdr_c))
    for r_label, r_key, order in [
        ("Purged h=15", "h15_purged_gap15", [("QCML Herm (d=8)", "qcml_herm_d8"), ("HistGBDT", "hgbdt"), ("MLP-tanh", "mlp_tanh"), ("Ridge", "ridge")]),
        ("Rolling 126d", "w126_rolling_train", [("QCML Herm (d=8)", "qcml_herm_d8"), ("MLP-tanh", "mlp_tanh"), ("HistGBDT", "hgbdt"), ("Ridge", "ridge")]),
    ]:
        for m_label, m_key in order:
            ev = grid[r_key][m_key]
            ens = ev["ensemble"]
            wins = sum(1 for v in ens["fold_rank_ics"] if v > 0.0)
            ic_1s = f"{ev['oos_rank_ic_single_mean']:+.4f} +- {ev['oos_rank_ic_seed_std']:.4f}"
            ic_ens = f"{ev['oos_rank_ic_ens']:+.4f} ({ens['rank_ic_nw_t']:+.2f})"
            fm_s = f"{ens['orth_fm_ic']:+.4f} ({ens['orth_fm_nw_pval']:.4f})"
            print(
                f"{r_label:<13}  {m_label:<18}  {ev['n_seeds']:>5d}  {ic_1s:>16}  "
                f"{ic_ens:>16}  {ev['oos_r2']:+9.5f}  {ev['oos_r2_ens']:+9.5f}  "
                f"{fm_s:>16}  {f'{wins}/6':>4}"
            )


if __name__ == "__main__":
    main()
