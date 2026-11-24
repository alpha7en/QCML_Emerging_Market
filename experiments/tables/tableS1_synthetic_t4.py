"""Table S1: Synthetic non-commutative positive controls (Panel A; Supplementary S1) and four-target
joint market-state forecasting y^{joint}_{i,t,5} on MOEX M+N (37,048 OOS rows, N_seeds = 5; Panel B;
Supplementary S2).

Usage: python -I experiments/tables/tableS1_synthetic_t4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_suite_metrics


def main() -> None:
    syn = load_suite_metrics("synthetic_controls")
    rob = load_suite_metrics("multioutput_robustness")
    b0a = syn["b0a"]
    b0b = syn["b0b"]

    print("[Panel A: Synthetic Non-Commutative Positive Controls (S^1, S^2, and d=3 Spin-1 Quadrupole)]")
    s1 = b0a["s1_unsupervised"]
    s2 = b0a["s2_supervised_3way"]
    print(
        f"  S^1 unsup. (d=2)        : Herm E0 = {s1['hermitian']['test_mean_E0']:.5f} "
        f"(k_nc={s1['hermitian']['kappa_nc']:.4f}), Real E0 = {s1['real_sym']['test_mean_E0']:.5f}, "
        f"Diag E0 = {s1['commuting_diag']['test_mean_E0']:.5f}"
    )
    print(
        f"  S^2 3-way (d=2, q=3)    : Herm R2 = {s2['hermitian']['test_r2']:.5f} "
        f"(k_nc={s2['hermitian']['kappa_nc']:.4f}), Real R2 = {s2['real_sym']['test_r2']:.5f} "
        f"(k_nc={s2['real_sym']['kappa_nc']:.4f}), Diag Gibbs/hard R2 = "
        f"{s2['commuting_diag_gibbs']['test_r2']:.5f} / {s2['commuting_diag_hard']['test_r2']:.5f}"
    )
    for label, key in [
        ("Quadrupole QCML Herm (d=3)", "herm_d3"),
        ("Quadrupole QCML Real (d=3)", "realsym_d3"),
        ("Quadrupole QCML Diag (d=3)", "diag_d3"),
        ("Quadrupole QCML Diag (d=9, iso)", "diag_d9"),
        ("Quadrupole HistGBDT", "hgbdt"),
        ("Quadrupole Ridge", "ridge"),
    ]:
        st = b0b[key]
        print(
            f"  {label:<32}: params={st['n_params']:>4d}  "
            f"R2 = {st['mean_r2']:+.5f} +- {st['std_r2']:.5f}"
        )

    print("\n[Panel B: Four-Target Joint Market State y^{joint}_{i,t,5} (q=4, M+N, 37,048 OOS Rows)]")
    t4 = rob["t4_multioutput"]
    targets = ["y_h5_cs", "fwd_vol_5d", "fwd_dd_5d", "fwd_vol_shock_5d"]
    hdr = (
        f"{'Model':<18}  {'Params':>6}  {'1s IC':>7}  {'1s R2':>7}  "
        f"{'Return (1s/Ens, R2)':>22}  {'5d Vol':>20}  {'5d Drawdown':>22}  {'5d Vol Shock':>20}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, key in [
        ("QCML Herm (q=4)", "qcml_herm_d8_t4"),
        ("QCML Herm 4x1", "qcml_herm_d8_4x_q1"),
        ("MLP-tanh-Multi", "mlp_tanh_multi"),
        ("Ridge-Multi", "ridge_multi"),
    ]:
        row = t4[key]
        pt = row["per_target"]
        cells = [
            f"{pt[t]['oos_rank_ic_single_mean']:.4f}/{pt[t]['oos_rank_ic_ens']:.4f} ({pt[t]['oos_r2_ens']:+.3f})"
            for t in targets
        ]
        print(
            f"{label:<18}  {row['n_params']:>6,d}  {row['mean_single_seed_rank_ic']:+7.4f}  "
            f"{row['mean_single_seed_r2']:+7.4f}  {cells[0]:>22}  {cells[1]:>20}  {cells[2]:>22}  {cells[3]:>20}"
        )


if __name__ == "__main__":
    main()
