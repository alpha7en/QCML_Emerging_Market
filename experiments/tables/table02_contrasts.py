"""Table 2: Paired daily RankIC contrasts (759 trading days) testing empirical de-quantization,
three-qubit VQC entanglement, and date-wise news-shuffle placebos (Sec. 4.1).

Usage: python -I experiments/tables/table02_contrasts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fmt_p, load_suite_metrics


def main() -> None:
    cls_m = load_suite_metrics("classical_controls")
    vqc_m = load_suite_metrics("vqc_controls")
    q_m = load_suite_metrics("qcml_benchmark")

    paired_mn = q_m["paired_tests"]["market_news"]
    q_mn = q_m["models"]["market_news"]

    header = f"{'Contrast (first minus second)':<46}  {'Diagnostic':<26}  {'dRankIC':>8}  {'p_NW':>7}  {'p_boot':>7}  {'p_Holm':>7}"
    print(header)
    print("-" * len(header))

    print("[Panel A: Operator-swap and baseline contrasts (M+N, 10-seed ensembles)]")
    panel_a = [
        (
            "QCML Herm (d=8) - QCML Real (d=11)",
            f"k_nc={q_mn['qcml_herm_d8']['kappa_nc_mean']:.4f} vs {q_mn['qcml_realsym_d11']['kappa_nc_mean']:.4f}",
            "herm_d8_vs_realsym_d11",
        ),
        (
            "QCML Herm (d=8) - QCML Diag (d=64, b*)",
            f"sd={q_mn['qcml_herm_d8']['oos_rank_ic_seed_std']:.4f} vs {q_mn['qcml_diag_d64_bstar']['oos_rank_ic_seed_std']:.4f}",
            "herm_d8_vs_diag_bstar",
        ),
        (
            "QCML Herm (le=0.05) - QCML Herm (le=0)",
            f"k_nc={q_mn['qcml_herm_d8']['kappa_nc_mean']:.4f} vs {q_mn['qcml_herm_d8_le0']['kappa_nc_mean']:.4f}",
            "herm_d8_vs_le0",
        ),
        ("QCML Herm (d=8) - HistGBDT", "MCS_0.05^IC tie", "herm_d8_vs_hgbdt"),
        ("QCML Herm (d=8) - MLP-tanh", "MCS_0.05^IC tie", "herm_d8_vs_mlp_tanh"),
        ("QCML Herm (d=8) - RFF-KRR", "Static RBF control", "herm_d8_vs_rff_krr"),
        ("QCML Herm (d=8) - VQC Affine (CNOT)", "Iso-param VQC control", "herm_d8_vs_vqc_n3_affine_cnot_fullherm"),
        ("QCML Herm (d=8) - FiLM-MLP (iso)", "Bounded classical control", "herm_d8_vs_film_mlp_iso"),
        ("QCML Herm (d=8) - Mom21-LowVol", "Factor ceiling", "herm_d8_vs_mom21_lowvol"),
    ]
    for label, diag_str, key in panel_a:
        st = paired_mn[key]["ic"]
        print(
            f"{label:<46}  {diag_str:<26}  {st['diff_mean']:+8.4f}  "
            f"{fmt_p(st['nw_pval'], 1e-4):>7}  {fmt_p(st['boot_pval'], 5e-4):>7}  {st['holm_pval']:7.4f}"
        )

    print("\n[Panel B: Three-qubit VQC entanglement contrasts (CNOT minus NoCNOT)]")
    vqc_ent = vqc_m["entanglement_tests"]
    vqc_models = vqc_m["models"]
    panel_b = [
        ("MKT Affine, full-Hermitian (N=10)", "market_only", "affine_fullherm", "vqc_n3_affine_cnot_fullherm"),
        ("MKT Affine, local-Z/X (N=5)", "market_only", "affine_localzx", "vqc_n3_affine_cnot_localzx"),
        ("MKT Cyclic, full-Hermitian (N=10)", "market_only", "cyclic_fullherm", "vqc_n3_cyclic_cnot_fullherm"),
        ("M+N Affine, full-Hermitian (N=10)", "market_news", "affine_fullherm", "vqc_n3_affine_cnot_fullherm"),
        ("M+N Affine, local-Z/X (N=5)", "market_news", "affine_localzx", "vqc_n3_affine_cnot_localzx"),
        ("M+N Cyclic, full-Hermitian (N=10)", "market_news", "cyclic_fullherm", "vqc_n3_cyclic_cnot_fullherm"),
    ]
    for label, mod, pair_key, model_key in panel_b:
        st = vqc_ent[mod][pair_key]["ens_paired_ic"]
        qmw = vqc_models[mod][model_key]["q_mw_mean"]
        print(
            f"{label:<46}  {f'Q_MW = {qmw:.3f}':<26}  {st['diff_mean']:+8.4f}  "
            f"{st['nw_pval']:7.4f}  {st['boot_pval']:7.4f}  {'---':>7}"
        )

    print("\n[Panel C: Date-wise news-shuffle placebos (real vs. 5-seed shuffled news, M+N)]")
    placebo_cls = cls_m["news_placebo"]
    panel_c_qcml = [
        ("QCML Herm (d=8)", "Projective ground state", paired_mn["herm_d8_vs_newsshuf"]["ic"], True),
        ("QCML Diag (d=64, beta=2)", "Commuting Gibbs kernel", paired_mn["diag_b2_vs_newsshuf"]["ic"], True),
        ("HistGBDT", "Boosted decision trees", placebo_cls["hgbdt"]["real_minus_shuffled_ic"], False),
        ("RFF-KRR", "Static random Fourier", placebo_cls["rff_krr"]["real_minus_shuffled_ic"], False),
    ]
    for label, diag_str, st, has_holm in panel_c_qcml:
        holm_s = f"{st['holm_pval']:7.4f}" if has_holm else f"{'---':>7}"
        print(
            f"{label:<46}  {diag_str:<26}  {st['diff_mean']:+8.4f}  "
            f"{st['nw_pval']:7.4f}  {st['boot_pval']:7.4f}  {holm_s}"
        )


if __name__ == "__main__":
    main()
