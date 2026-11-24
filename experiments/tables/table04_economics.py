"""Table 4: Causal MOEX portfolio frictions (M+N, 151 weeks, one-day lag, 38 masked short-ban
rebalances, 500 bps/yr borrow; Panel A) and unclipped February-March 2022 crisis out-of-distribution
bounds (320 rows / 13 acute days vs. pre-crisis training mean; Panel B; Sec. 4.3).

Usage: python -I experiments/tables/table04_economics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_suite_metrics


def main() -> None:
    suites = {
        s: load_suite_metrics(s)
        for s in ("classical_controls", "vqc_controls", "qcml_benchmark", "spectral_stress")
    }

    print("[Panel A: Causal M+N portfolio performance (15 bps flat and 10/30 bps tiered costs)]")
    hdr_a = (
        f"{'Model':<26}  {'IR_LO [95% CI]':>22}  {'Tier IR':>7}  {'1s IR':>6}  "
        f"{'SR_LS [95% CI]':>22}  {'DSR_LO / DSR_LS':>15}  {'Ex-CBR SR':>9}"
    )
    print(hdr_a)
    print("-" * len(hdr_a))

    panel_a = [
        ("Mom21-LowVol", "classical_controls", "mom21_lowvol"),
        ("HistGBDT", "classical_controls", "hgbdt"),
        ("LightGBM", "classical_controls", "lgbm"),
        ("MLP-tanh", "classical_controls", "mlp_tanh"),
        ("FiLM-MLP (iso)", "classical_controls", "film_mlp_iso"),
        ("VQC Cyclic (CNOT)", "vqc_controls", "vqc_n3_cyclic_cnot_fullherm"),
        ("QCML Herm (d=8)", "qcml_benchmark", "qcml_herm_d8"),
        ("QCML Real (d=11)", "qcml_benchmark", "qcml_realsym_d11"),
        ("QCML Diag (d=64, b*)", "qcml_benchmark", "qcml_diag_d64_bstar"),
        ("QCML Real (le=0)", "qcml_benchmark", "qcml_realsym_d11_le0"),
        ("50/50 Blend (MomLV+Herm)", "qcml_benchmark", "blend_50_50_momlv_herm_d8"),
    ]
    for label, s_name, key in panel_a:
        ec = suites[s_name]["models"]["market_news"][key]["economics"]
        ci_lo = ec["net_ir_lo_active_15bps_ci"]
        ci_ls = ec["net_sharpe_ls_15bps_ci"]
        lo_s = f"{ec['net_ir_lo_active_15bps']:+.3f} [{ci_lo[0]:+.2f}, {ci_lo[1]:+.2f}]"
        ls_s = f"{ec['net_sharpe_ls_15bps']:+.3f} [{ci_ls[0]:+.2f}, {ci_ls[1]:+.2f}]"
        dsr_s = f"{ec['dsr_lo_active_15bps']:.3f} / {ec['dsr_ls_15bps']:.3f}"
        print(
            f"{label:<26}  {lo_s:>22}  {ec['net_ir_lo_active_tiered']:+7.3f}  "
            f"{ec['single_seed_net_ir_lo_15bps_mean']:+6.3f}  {ls_s:>22}  "
            f"{dsr_s:>15}  {ec['net_sharpe_lo_ex_cbr_15bps']:+9.3f}"
        )

    print("\n[Panel B: Unclipped February-March 2022 crisis shock bounds (320 rows / 13 acute days)]")
    hdr_b = (
        f"{'Model':<26}  {'M+N Acute R2':>12}  {'M+N Reopen R2':>13}  {'M+N max|y|':>10}  "
        f"{'MKT Acute R2':>12}  {'MKT Reopen R2':>13}  {'MKT max|y|':>10}"
    )
    print(hdr_b)
    print("-" * len(hdr_b))

    ood_mn = suites["spectral_stress"]["ood_2022"]["market_news"]["raw_unclipped_bounds"]
    ood_mkt = suites["spectral_stress"]["ood_2022"]["market_only"]["raw_unclipped_bounds"]
    panel_b = [
        ("QCML Herm (d=8)", "qcml_herm_d8"),
        ("QCML Real (d=11)", "qcml_realsym_d11"),
        ("QCML Diag (d=64, beta=2)", "qcml_diag_d64_b2"),
        ("VQC Affine (CNOT)", "vqc_n3_affine_cnot_fullherm"),
        ("FiLM-MLP (iso)", "film_mlp_iso"),
        ("Ridge", "ridge"),
        ("MLP-tanh", "mlp_tanh"),
        ("HistGBDT", "hgbdt"),
        ("MLP-ReLU", "mlp_relu"),
    ]
    for label, key in panel_b:
        a_mn, r_mn = ood_mn[key]["acute"], ood_mn[key]["post_reopen"]
        a_mk, r_mk = ood_mkt[key]["acute"], ood_mkt[key]["post_reopen"]
        print(
            f"{label:<26}  {a_mn['r2_vs_precrisis_train_mean']:+12.4f}  "
            f"{r_mn['r2_vs_precrisis_train_mean']:+13.4f}  {a_mn['max_abs_pred']:10.4f}  "
            f"{a_mk['r2_vs_precrisis_train_mean']:+12.4f}  "
            f"{r_mk['r2_vs_precrisis_train_mean']:+13.4f}  {a_mk['max_abs_pred']:10.4f}"
        )


if __name__ == "__main__":
    main()
