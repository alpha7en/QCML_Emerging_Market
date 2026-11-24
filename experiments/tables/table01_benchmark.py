"""Table 1: Out-of-sample cross-sectional forecasting benchmark (h = 5, 37,048 unfiltered stock-day
rows, 759 trading days, one-day execution lag, 15 bps one-way transaction cost; Sec. 4.1).

Usage: python -I experiments/tables/table01_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_suite_metrics

ROWS = [
    ("Mom21-LowVol", "classical_controls", "mom21_lowvol", True),
    ("Ridge", "classical_controls", "ridge", False),
    ("RFF-KRR", "classical_controls", "rff_krr", False),
    ("HistGBDT", "classical_controls", "hgbdt", False),
    ("LightGBM", "classical_controls", "lgbm", False),
    ("MLP-tanh", "classical_controls", "mlp_tanh", False),
    ("FiLM-MLP (iso)", "classical_controls", "film_mlp_iso", False),
    ("VQC Cyclic (CNOT)", "vqc_controls", "vqc_n3_cyclic_cnot_fullherm", False),
    ("VQC Cyclic (NoCNOT)", "vqc_controls", "vqc_n3_cyclic_nocnot_fullherm", False),
    ("VQC Affine (CNOT)", "vqc_controls", "vqc_n3_affine_cnot_fullherm", False),
    ("QCML Herm (d=8)", "qcml_benchmark", "qcml_herm_d8", False),
    ("QCML Real (d=11)", "qcml_benchmark", "qcml_realsym_d11", False),
    ("QCML Diag (d=64, b*)", "qcml_benchmark", "qcml_diag_d64_bstar", False),
]


def main() -> None:
    suites = {
        s: load_suite_metrics(s)
        for s in ("classical_controls", "vqc_controls", "qcml_benchmark")
    }
    header = (
        f"{'Model':<22}  {'Params (MKT/M+N)':>16}  {'MKT 1s / Ens IC':>17}  "
        f"{'M+N 1s / Ens IC':>17}  {'M+N R2_OOS':>10}  {'IR_LO (15bp)':>12}  "
        f"{'c* (bp)':>7}  {'DSR_LO':>6}"
    )
    print(header)
    print("-" * len(header))
    for label, suite_name, key, rank_only in ROWS:
        mkt = suites[suite_name]["models"]["market_only"][key]
        mn = suites[suite_name]["models"]["market_news"][key]
        econ = mn["economics"]
        params_str = f"{mkt['n_params']:,} / {mn['n_params']:,}"
        mkt_ic = f"{mkt['oos_rank_ic_single_mean']:+.4f} / {mkt['oos_rank_ic_ens']:+.4f}"
        mn_ic = f"{mn['oos_rank_ic_single_mean']:+.4f} / {mn['oos_rank_ic_ens']:+.4f}"
        r2_str = "n/a" if rank_only else f"{mn['oos_r2_ens']:+.5f}"
        print(
            f"{label:<22}  {params_str:>16}  {mkt_ic:>17}  {mn_ic:>17}  "
            f"{r2_str:>10}  {econ['net_ir_lo_active_15bps']:+12.3f}  "
            f"{econ['breakeven_cost_lo_bps']:7.1f}  {econ['dsr_lo_active_15bps']:6.3f}"
        )


if __name__ == "__main__":
    main()
