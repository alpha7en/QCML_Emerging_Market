"""Figure 5: Norm-controlled spectral-gap refutation across 759 trading days: contemporaneous
stress ROC (a), crisis AUROC dumbbells across 6 cells (b), norm-explained R^2_OLS against
the untrained GUE(8) null (c), and forward VAR(5) Granger Wald statistics (d; Sec. 4.3).

Usage: python -I experiments/figures/fig05_spectral.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in

ROW_SPECS: list[tuple[str, str, str, float]] = [
    ("market_news", "qcml_sup_le005", "Sup.", 0.00),
    ("market_news", "qcml_unsup_le100", "Unsup.", 1.00),
    ("market_news", "qcml_candelori_w01", "Cand.", 2.00),
    ("market_only", "qcml_sup_le005", "Sup.", 3.70),
    ("market_only", "qcml_unsup_le100", "Unsup.", 4.70),
    ("market_only", "qcml_candelori_w01", "Cand.", 5.70),
]
BLOCK_SPANS: dict[str, tuple[float, float]] = {
    "market_news": (-0.42, 2.42),
    "market_only": (3.28, 6.12),
}


def empirical_roc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.isfinite(y_true) & np.isfinite(scores)
    yt = y_true[mask].astype(np.int64)
    sc = scores[mask].astype(np.float64)
    order = np.argsort(-sc, kind="mergesort")
    yt_s = yt[order]
    sc_s = sc[order]
    distinct_idx = np.where(np.diff(sc_s))[0]
    thresh_idx = np.r_[distinct_idx, yt_s.size - 1]
    tps = np.cumsum(yt_s)[thresh_idx]
    fps = (1 + thresh_idx) - tps
    tpr = np.r_[0, tps] / float(tps[-1])
    fpr = np.r_[0, fps] / float(fps[-1])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def build(runs: Path, out_dir: Path) -> list[Path]:
    spec_dir = runs / "spectral_stress"
    df = pd.read_csv(spec_dir / "daily_regime_series.csv")
    roc_data = load_json(spec_dir / "roc_and_spectral.json")
    granger_data = load_json(spec_dir / "granger_var5.json")
    if len(df) != 759:
        raise ValueError(f"Expected 759 OOS trading days, got {len(df)}")

    pal = setup_mpl(base_pt=7.3, tick_pt=6.8)
    panels_dir = out_dir / "panels"
    h_mm = 55.0
    bot, height = 0.235, 0.510
    written: list[Path] = []

    fig_a = plt.figure(figsize=subfig_size_in(0.265, h_mm))
    ax_a = fig_a.add_axes([0.270, bot, 0.675, height])
    style_axis(ax_a)

    y_true = df["mn_stress_contemp"].to_numpy(dtype=np.int64)
    mn_roc = roc_data["market_news"]
    curves = [
        ("mn_garch11", float(mn_roc["baselines"]["garch11"]["auroc_contemp"]), "GARCH", pal["slate"], ":", 1.20),
        ("mn_norm2", float(mn_roc["baselines"]["norm2"]["auroc_contemp"]), r"${\|\mathbf{x}\|_2}^{\,2}$", pal["ochre"], "-", 1.25),
        ("mn_qcml_sup_le005_dE", float(mn_roc["models"]["qcml_sup_le005"]["dE"]["auroc_contemp"]), r"$\Delta E$", pal["quantum"], "-", 1.35),
        ("mn_qcml_sup_le005_dE_tilde", float(mn_roc["models"]["qcml_sup_le005"]["dE_tilde"]["auroc_contemp"]), r"$\widetilde{\Delta E}$", pal["quantum_light"], "--", 1.20),
    ]
    ax_a.plot([0.0, 1.0], [0.0, 1.0], color=pal["light"], ls=":", lw=0.8, zorder=1)
    for col_name, j_auc, lbl, col, ls, lw in curves:
        fpr, tpr, emp_auc = empirical_roc(y_true, df[col_name].to_numpy(dtype=np.float64))
        if abs(emp_auc - j_auc) >= 1e-4:
            raise ValueError(f"ROC AUROC mismatch on {col_name}: {emp_auc:.6f} vs {j_auc:.6f}")
        ax_a.plot(fpr, tpr, color=col, ls=ls, lw=lw, label=f"{lbl} ({j_auc:.2f})")

    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_xticks(np.array([0.0, 0.5, 1.0]))
    ax_a.set_yticks(np.array([0.0, 0.5, 1.0]))
    ax_a.xaxis.set_minor_locator(mticker.FixedLocator([0.25, 0.75]))
    ax_a.yaxis.set_minor_locator(mticker.FixedLocator([0.25, 0.75]))
    ax_a.set_xlabel("False positive rate", labelpad=3.4)
    ax_a.set_ylabel("True positive rate", labelpad=4.4)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.44, 1.04),
        ncol=2,
        columnspacing=0.72,
        handlelength=0.95,
        handletextpad=0.38,
        labelspacing=0.34,
    )
    written.append(save_panel_pdf(fig_a, panels_dir, "fig05a_stress_roc"))
    plt.close(fig_a)

    fig_b = plt.figure(figsize=subfig_size_in(0.305, h_mm))
    ax_b = fig_b.add_axes([0.285, bot, 0.640, height])
    style_axis(ax_b)
    ax_b.axhline(2.85, color=pal["faint"], lw=0.75, zorder=0)

    for mod, (y_lo, y_hi) in BLOCK_SPANS.items():
        auc_n2 = float(roc_data[mod]["baselines"]["norm2"]["auroc_contemp"])
        auc_ga = float(roc_data[mod]["baselines"]["garch11"]["auroc_contemp"])
        ax_b.plot([0.50, 0.50], [y_lo, y_hi], color=pal["light"], ls=":", lw=0.85, zorder=1)
        ax_b.plot([auc_n2, auc_n2], [y_lo, y_hi], color=pal["ochre"], ls="-", lw=1.10, zorder=2)
        ax_b.plot([auc_ga, auc_ga], [y_lo, y_hi], color=pal["slate"], ls=":", lw=1.10, zorder=2)

    for mod, key, _, y in ROW_SPECS:
        m_rec = roc_data[mod]["models"][key]
        de_auc, de_ci = float(m_rec["dE"]["auroc_contemp"]), m_rec["dE"]["auroc_contemp_ci"]
        det_auc, det_ci = float(m_rec["dE_tilde"]["auroc_contemp"]), m_rec["dE_tilde"]["auroc_contemp_ci"]
        ax_b.plot([det_auc, de_auc], [y, y], color=pal["light"], lw=0.90, zorder=3)
        ax_b.plot([de_ci[0], de_ci[1]], [y, y], color=pal["quantum"], lw=1.10, alpha=0.55, zorder=4)
        ax_b.plot([det_ci[0], det_ci[1]], [y, y], color=pal["quantum_light"], lw=1.10, alpha=0.80, zorder=4)
        ax_b.scatter([de_auc], [y], s=20, marker="o", facecolor=pal["quantum"], edgecolor="white", linewidths=0.35, zorder=6)
        ax_b.scatter([det_auc], [y], s=20, marker="o", facecolor="white", edgecolor=pal["quantum"], linewidths=0.95, zorder=7)

    ax_b.set_yticks([y for _, _, _, y in ROW_SPECS])
    ax_b.set_yticklabels([lbl for _, _, lbl, _ in ROW_SPECS])
    ax_b.text(-0.06, -0.68, "M+N", transform=ax_b.get_yaxis_transform(), fontsize=6.8, fontweight="bold", color=pal["ink"], ha="right", va="center")
    ax_b.text(-0.06, 3.05, "MKT", transform=ax_b.get_yaxis_transform(), fontsize=6.8, fontweight="bold", color=pal["ink"], ha="right", va="center")
    ax_b.set_ylim(6.35, -1.05)
    ax_b.set_xlim(0.18, 1.00)
    ax_b.set_xticks(np.array([0.25, 0.50, 0.75, 1.00]))
    ax_b.set_xlabel("AUROC (95% CI)", labelpad=3.4)

    hb = [
        mlines.Line2D([], [], color=pal["quantum"], marker="o", ms=3.6, lw=0, label=r"Raw $\Delta E$"),
        mlines.Line2D([], [], color=pal["quantum"], marker="o", mfc="white", mew=0.95, ms=3.6, lw=0, label=r"$\widetilde{\Delta E}$"),
        mlines.Line2D([], [], color=pal["ochre"], ls="-", lw=1.1, label=r"${\|\mathbf{x}\|_2}^{\,2}$"),
        mlines.Line2D([], [], color=pal["slate"], ls=":", lw=1.1, label="GARCH"),
    ]
    ax_b.legend(
        handles=hb,
        loc="lower center",
        bbox_to_anchor=(0.42, 1.04),
        ncol=2,
        columnspacing=0.78,
        handlelength=0.95,
        handletextpad=0.38,
        labelspacing=0.34,
    )
    written.append(save_panel_pdf(fig_b, panels_dir, "fig05b_crisis_auroc"))
    plt.close(fig_b)

    fig_c = plt.figure(figsize=subfig_size_in(0.200, h_mm))
    ax_c = fig_c.add_axes([0.105, bot, 0.820, height])
    style_axis(ax_c)
    ax_c.axhline(2.85, color=pal["faint"], lw=0.75, zorder=0)

    for mod, (y_lo, y_hi) in BLOCK_SPANS.items():
        gue = roc_data[mod]["gue_null_759d"]
        ax_c.plot([float(gue["r2_E0_on_norm2_daily_mean"])] * 2, [y_lo, y_hi], color=pal["ink"], ls="--", lw=0.90, alpha=0.65, zorder=2)
        ax_c.plot([float(gue["r2_dE_on_norm_poly_daily_mean"])] * 2, [y_lo, y_hi], color=pal["quantum"], ls="--", lw=0.90, alpha=0.65, zorder=2)

    for mod, key, _, y in ROW_SPECS:
        m_rec = roc_data[mod]["models"][key]
        ax_c.scatter([float(m_rec["r2_E0_on_norm2"])], [y], s=20, marker="s", facecolor=pal["ink"], edgecolor="white", linewidths=0.4, zorder=5)
        ax_c.scatter([float(m_rec["r2_dE_on_norm_poly"])], [y], s=22, marker="o", facecolor=pal["quantum"], edgecolor="white", linewidths=0.4, zorder=6)

    ax_c.set_ylim(6.35, -1.05)
    ax_c.set_xlim(0.0, 1.08)
    ax_c.set_xticks(np.array([0.0, 0.5, 1.0]))
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    ax_c.set_xlabel(r"${R^2}_{\,\mathrm{OLS}}$", labelpad=3.4)

    hc = [
        mlines.Line2D([], [], color=pal["ink"], marker="s", ms=3.4, lw=0, label=r"$E_0$"),
        mlines.Line2D([], [], color=pal["ink"], ls="--", lw=0.9, label=r"GUE $E_0$"),
        mlines.Line2D([], [], color=pal["quantum"], marker="o", ms=3.6, lw=0, label=r"$\Delta E$"),
        mlines.Line2D([], [], color=pal["quantum"], ls="--", lw=0.9, label=r"GUE $\Delta E$"),
    ]
    ax_c.legend(
        handles=hc,
        loc="lower center",
        bbox_to_anchor=(0.48, 1.04),
        ncol=2,
        columnspacing=0.68,
        handlelength=0.95,
        handletextpad=0.38,
        labelspacing=0.34,
    )
    written.append(save_panel_pdf(fig_c, panels_dir, "fig05c_norm_r2"))
    plt.close(fig_c)

    fig_d = plt.figure(figsize=subfig_size_in(0.200, h_mm))
    ax_d = fig_d.add_axes([0.105, bot, 0.820, height])
    style_axis(ax_d)
    ax_d.axhline(2.85, color=pal["faint"], lw=0.75, zorder=0)

    # Chi-squared critical values for VAR(5) forward Granger Wald test (df = 5, 6-cell Bonferroni).
    crit_nom = float(chi2.ppf(0.95, df=5))
    crit_bonf = float(chi2.ppf(1.0 - 0.05 / 6.0, df=5))
    ax_d.plot([crit_nom, crit_nom], [-0.38, 6.20], color=pal["slate"], ls=":", lw=0.90, zorder=1)
    ax_d.plot([crit_bonf, crit_bonf], [-0.38, 6.20], color=pal["ochre"], ls="--", lw=0.95, zorder=1)
    ax_d.text(crit_nom - 2.6, -0.68, "0.05", fontsize=6.5, color=pal["slate"], ha="right", va="center")
    ax_d.text(crit_bonf + 2.6, -0.68, "Bonf.", fontsize=6.6, color=pal["ochre"], ha="left", va="center")

    for mod, key, _, y in ROW_SPECS:
        g_rec = granger_data[mod][key]
        w_raw = float(g_rec["granger_dE_raw"]["fwd_wald"])
        w_ctl = float(g_rec["granger_dE_norm_controlled"]["fwd_wald"])
        if abs(w_raw - w_ctl) > 4.2:
            ax_d.annotate(
                "",
                xy=(w_ctl, y),
                xytext=(w_raw, y),
                arrowprops=dict(arrowstyle="->", color=pal["slate"], lw=0.75, shrinkA=3.4, shrinkB=3.4, mutation_scale=4.5),
                zorder=3,
            )
        ax_d.scatter([w_raw], [y], s=16, marker="o", facecolor=pal["quantum"], edgecolor="white", linewidths=0.35, zorder=5)
        ax_d.scatter([w_ctl], [y], s=16, marker="o", facecolor="white", edgecolor=pal["quantum"], linewidths=0.95, zorder=6)

    ax_d.text(60.0, 1.0, r"$5/6 \to 0/6$", fontsize=6.7, color=pal["ochre"], ha="right", va="center")
    ax_d.set_ylim(6.35, -1.05)
    ax_d.set_xlim(-2.0, 62.0)
    ax_d.set_xticks(np.array([0, 20, 40, 60]))
    ax_d.tick_params(axis="y", left=False, labelleft=False)
    ax_d.set_xlabel(r"$W_{\mathrm{Granger}}$ (${\chi^2}_{\,5}$)", fontsize=8.65, labelpad=3.4)

    hd = [
        mlines.Line2D([], [], color=pal["quantum"], marker="o", ms=3.5, lw=0, label="Raw"),
        mlines.Line2D([], [], color=pal["quantum"], marker="o", mfc="white", mew=0.95, ms=3.5, lw=0, label="+Norm"),
    ]
    ax_d.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.50, 1.04), ncol=1, handlelength=0.95, handletextpad=0.38, labelspacing=0.34)
    written.append(save_panel_pdf(fig_d, panels_dir, "fig05d_granger_var5"))
    plt.close(fig_d)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=runs_dir())
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    for path in build(args.runs, args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
