"""Table 3: Norm-controlled spectral-stress audit over 759 out-of-sample trading days (684 normal,
75 stress; N_seeds = 3 per QCML training regime; 10 untrained GUE(d=8) draws; Sec. 4.2).

Usage: python -I experiments/tables/table03_spectral.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fmt_p, load_suite_metrics


def main() -> None:
    m = load_suite_metrics("spectral_stress")
    mn = m["b4"]["market_news"]
    mkt = m["b4"]["market_only"]

    header = (
        f"{'Specification':<28}  {'R2(E0)':>7}  {'R2(dE)':>7}  {'dE':>6}  {'dE_til':>6}  "
        f"{'dE_sph':>6}  {'E0_sph':>6}  {'d_FS':>6}  {'dE_prp':>6}  "
        f"{'Raw W_G (p)':>16}  {'Ctrl W_G (p)':>16}"
    )
    print(header)
    print("-" * len(header))

    print("[Baselines and untrained random-matrix null (759 days)]")
    g_mn = mn["baselines"]["garch11"]["auroc_contemp"]
    n_mn = mn["baselines"]["norm2"]["auroc_contemp"]
    n_mkt = mkt["baselines"]["norm2"]["auroc_contemp"]
    print(f"{'GARCH(1,1) (M+N/MKT)':<28}  {'---':>7}  {'---':>7}  {g_mn:6.3f}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>16}  {'---':>16}")
    print(f"{'||x||_2^2 (M+N/MKT)':<28}  {'---':>7}  {'---':>7}  {f'{n_mn:.3f}/{n_mkt:.3f}':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>16}  {'---':>16}")

    for label, blk in [("Untrained GUE (d=8, M+N)", mn), ("Untrained GUE (d=8, MKT)", mkt)]:
        gue = blk["gue_null_759d"]
        print(
            f"{label:<28}  {gue['r2_E0_on_norm2_daily_mean']:7.4f}  "
            f"{gue['r2_dE_on_norm_poly_daily_mean']:7.4f}  {'---':>6}  {'---':>6}  "
            f"{'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>16}  {'---':>16}"
        )

    for panel_title, blk in [
        ("\n[Trained matrix-QCML (d=8, N_seeds=3) on market-plus-news (M+N)]", mn),
        ("\n[Trained matrix-QCML (d=8, N_seeds=3) on market-only (MKT)]", mkt),
    ]:
        print(panel_title)
        for label, key in [
            ("Supervised (lambda_E=0.05)", "qcml_sup_le005"),
            ("Unsupervised (lambda_E=1.0)", "qcml_unsup_le100"),
            ("Candelori (w_var=0.1)", "qcml_candelori_w01"),
        ]:
            md = blk["models"][key]
            g_raw = md["granger_dE_raw"]
            g_ctl = md["granger_dE_norm_controlled"]
            raw_s = f"{g_raw['fwd_wald']:5.2f} ({fmt_p(g_raw['fwd_pval'])})"
            ctl_s = f"{g_ctl['fwd_wald']:5.2f} ({fmt_p(g_ctl['fwd_pval'])})"
            print(
                f"{label:<28}  {md['r2_E0_on_norm2']:7.4f}  {md['r2_dE_on_norm_poly']:7.4f}  "
                f"{md['dE']['auroc_contemp']:6.3f}  {md['dE_tilde']['auroc_contemp']:6.3f}  "
                f"{md['dE_sphere']['auroc_contemp']:6.3f}  {md['E0_sphere']['auroc_contemp']:6.3f}  "
                f"{md['fubini_study_step']['auroc_contemp']:6.3f}  {md['dE_perp_causal']['auroc_contemp']:6.3f}  "
                f"{raw_s:>16}  {ctl_s:>16}"
            )


if __name__ == "__main__":
    main()
