"""Figure 6: Economic frictions and out-of-distribution bounds: cumulative active long-only
return at 15 bps (a), February-March 2022 halt bounds and max |y_hat| (b), net active IR_LO (c),
break-even cost c* (d), and deflated Sharpe ratio DSR_LO (e; Secs. 4.3-4.4).

Usage: python -I experiments/figures/fig06_frictions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in

PANEL_A_SPECS: list[tuple[str, str, str, str, Any, float, int]] = [
    ("qcml_herm_d8", "qcml_benchmark", r"QCML Herm ($d=8$)", "quantum", "-", 1.35, 8),
    ("qcml_realsym_d11", "qcml_benchmark", r"QCML Real ($d=11$)", "quantum_light", "--", 1.20, 7),
    ("qcml_diag_d64_bstar", "qcml_benchmark", r"QCML Diag ($d=64,\beta^*$)", "ochre", ":", 1.30, 6),
    ("hgbdt", "classical_controls", "HistGBDT", "ink", "-", 1.10, 5),
    ("mom21_lowvol", "classical_controls", "Mom21-LowVol", "slate", "-.", 0.95, 4),
    ("ridge", "classical_controls", "Ridge", "muted", ":", 1.00, 3),
]

PANEL_B_MODELS: list[tuple[str, str]] = [
    ("film_mlp_iso", "FiLM-MLP (iso)"),
    ("qcml_diag_d64_b2", r"QCML Diag ($d=64,\beta=2$)"),
    ("qcml_herm_d8", r"QCML Herm ($d=8$)"),
    ("qcml_realsym_d11", r"QCML Real ($d=11$)"),
    ("vqc_n3_affine_cnot_fullherm", "VQC Affine (CNOT)"),
    ("ridge", "Ridge"),
    ("mlp_tanh", "MLP-tanh"),
    ("hgbdt", "HistGBDT"),
    ("mlp_relu", "MLP-ReLU"),
]

PANEL_CDE_MODELS: list[tuple[str, str, str, str, str]] = [
    ("mom21_lowvol", "classical_controls", "Mom21-LowVol", "ink", "*"),
    ("qcml_herm_d8", "qcml_benchmark", r"QCML Herm ($d=8$)", "quantum", "o"),
    ("qcml_realsym_d11", "qcml_benchmark", r"QCML Real ($d=11$)", "quantum_light", "s"),
    ("qcml_diag_d64_bstar", "qcml_benchmark", r"QCML Diag ($d=64,\beta^*$)", "ochre", "p"),
    ("lgbm", "classical_controls", "LightGBM", "slate", ">"),
    ("hgbdt", "classical_controls", "HistGBDT", "ink", "^"),
    ("vqc_n3_cyclic_cnot_fullherm", "vqc_controls", "VQC Cyclic (CNOT)", "quantum", "D"),
    ("mlp_tanh", "classical_controls", "MLP-tanh", "slate", "s"),
    ("rff_krr", "classical_controls", "RFF-KRR", "muted", "o"),
    ("ridge", "classical_controls", "Ridge", "muted", "v"),
]


def _edge(pal: dict[str, str], ckey: str) -> str:
    return pal["quantum"] if ckey == "quantum_light" else pal[ckey]


def build(runs: Path, out_dir: Path) -> list[Path]:
    suites = ("classical_controls", "vqc_controls", "qcml_benchmark")
    summaries = {s: load_json(runs / s / "summary.json") for s in suites}
    sweeps = {s: load_json(runs / s / "cost_sweep.json")["market_news"] for s in suites}
    ood_data = load_json(runs / "spectral_stress" / "ood_2022_bounds.json")
    eq_cls = pd.read_csv(runs / "classical_controls" / "equity_curves.csv")
    eq_qcml = pd.read_csv(runs / "qcml_benchmark" / "equity_curves.csv")
    eq_qcml["cohort0_date"] = pd.to_datetime(eq_qcml["cohort0_date"])

    pal = setup_mpl(base_pt=7.5, tick_pt=6.9)
    panels_dir = out_dir / "panels"
    written: list[Path] = []

    fig_a = plt.figure(figsize=subfig_size_in(0.485, 54.0))
    ax_a = fig_a.add_axes([0.185, 0.220, 0.785, 0.665])
    style_axis(ax_a)

    dates = eq_qcml["cohort0_date"]
    ax_a.axhline(0.0, color=pal["light"], ls="--", lw=0.70, zorder=1)
    for mkey, suite, label, ckey, ls, lw, zo in PANEL_A_SPECS:
        df_eq = eq_qcml if suite == "qcml_benchmark" else eq_cls
        cum_pct = (np.cumprod(1.0 + df_eq[f"mn_{mkey}_lo_act_net15"].to_numpy(dtype=float)) - 1.0) * 100.0
        ax_a.plot(dates, cum_pct, color=pal[ckey], ls=ls, lw=lw, zorder=zo, label=label)

    ax_a.set_xlim(dates.iloc[0] - pd.Timedelta(days=12), dates.iloc[-1] + pd.Timedelta(days=16))
    ax_a.set_ylim(-11.0, 76.0)
    ax_a.set_yticks(np.array([-10, 0, 20, 40, 60]))
    ax_a.yaxis.set_minor_locator(mticker.FixedLocator([10, 30, 50, 70]))
    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_a.set_xlabel("Rebalance date (151 weeks)", labelpad=3.4)
    ax_a.set_ylabel("Active LO return (%)", labelpad=4.4)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.01, 1.03), ncol=1, handlelength=1.35, handletextpad=0.42, labelspacing=0.28)
    written.append(save_panel_pdf(fig_a, panels_dir, "fig06a_equity_curves"))
    plt.close(fig_a)

    fig_b = plt.figure(figsize=subfig_size_in(0.485, 54.0))
    ax_b = fig_b.add_axes([0.420, 0.220, 0.560, 0.665])
    style_axis(ax_b)

    raw_b = ood_data["market_news"]["raw_unclipped_bounds"]
    win_b = ood_data["market_news"]["winsorized_bounds"]
    yb = np.arange(len(PANEL_B_MODELS), dtype=float)
    ax_b.axvline(0.0, color=pal["light"], ls=":", lw=0.80, zorder=1)

    for i, (mkey, _lbl) in enumerate(PANEL_B_MODELS):
        r2_ac = float(raw_b[mkey]["acute"]["r2_vs_precrisis_train_mean"])
        r2_wi = float(win_b[mkey]["acute"]["r2_vs_precrisis_train_mean"])
        r2_po = float(raw_b[mkey]["post_reopen"]["r2_vs_precrisis_train_mean"])
        max_yh = float(raw_b[mkey]["acute"]["max_abs_pred"])

        ax_b.plot([min(r2_ac, r2_wi, r2_po), max(r2_ac, r2_wi, r2_po)], [yb[i], yb[i]], color=pal["light"], lw=0.85, zorder=2)
        ax_b.scatter([r2_po], [yb[i]], s=19, marker="s", facecolor="white", edgecolor=pal["quantum"], linewidths=0.95, zorder=4)
        ax_b.scatter([r2_ac], [yb[i]], s=18, marker="o", facecolor=pal["ochre"], edgecolor=pal["ochre"], linewidths=0.5, zorder=5)
        ax_b.scatter([r2_wi], [yb[i]], s=22, marker="^", facecolor="none", edgecolor=pal["slate"], linewidths=0.95, zorder=6)
        ax_b.text(0.046, yb[i], f"{max_yh:.4f}", fontsize=6.7, color=pal["ink"], ha="right", va="center")

    ax_b.text(0.046, -0.78, r"$\max\,|\hat{y}|$", fontsize=6.8, color=pal["slate"], ha="right", va="bottom")
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([lbl for _, lbl in PANEL_B_MODELS])
    ax_b.set_ylim(len(PANEL_B_MODELS) - 0.42, -0.88)
    ax_b.set_xlim(-0.092, 0.048)
    ax_b.spines["bottom"].set_bounds(-0.092, 0.010)
    ax_b.set_xticks(np.array([-0.08, -0.04, 0.00]))
    ax_b.set_xticklabels(["\u22120.08", "\u22120.04", "0.00"])
    ax_b.xaxis.set_minor_locator(mticker.FixedLocator([-0.06, -0.02]))
    ax_b.set_xlabel(r"Crisis ${R^2}_{\,\mathrm{OOS}}$", labelpad=3.4)

    hb = [
        mlines.Line2D([], [], color=pal["ochre"], marker="o", ms=3.5, lw=0, label="Acute"),
        mlines.Line2D([], [], color=pal["slate"], marker="^", mfc="none", mew=0.95, ms=3.8, lw=0, label=r"$\pm 5\sigma$ clip"),
        mlines.Line2D([], [], color=pal["quantum"], marker="s", mfc="white", mew=0.95, ms=3.5, lw=0, label="Reopen"),
    ]
    ax_b.legend(handles=hb, loc="lower center", bbox_to_anchor=(0.36, 1.030), ncol=3, columnspacing=0.80, handlelength=0.90, handletextpad=0.38)
    written.append(save_panel_pdf(fig_b, panels_dir, "fig06b_halt_bounds"))
    plt.close(fig_b)

    h_row2_mm = 52.0
    bot_c, hgt_c = 0.220, 0.665
    yc = np.arange(len(PANEL_CDE_MODELS), dtype=float) * 1.18
    yc[1:] += 0.32
    ylim_cde = (yc[-1] + 0.60, -0.95)

    fig_c = plt.figure(figsize=subfig_size_in(0.420, h_row2_mm))
    ax_c = fig_c.add_axes([0.450, bot_c, 0.520, hgt_c])
    style_axis(ax_c)
    ax_c.axhline(0.74, color=pal["faint"], lw=0.75, zorder=0)
    ax_c.axvline(0.0, color=pal["light"], ls="--", lw=0.70, zorder=1)

    fig_d = plt.figure(figsize=subfig_size_in(0.275, h_row2_mm))
    ax_d = fig_d.add_axes([0.095, bot_c, 0.850, hgt_c])
    style_axis(ax_d)
    ax_d.axhline(0.74, color=pal["faint"], lw=0.75, zorder=0)
    ax_d.plot([15.0, 15.0], [-0.42, yc[-1] + 0.60], color=pal["ochre"], ls="--", lw=0.90, zorder=3)
    ax_d.text(15.0, -0.75, "15 bps", fontsize=6.7, color=pal["ochre"], ha="center", va="bottom")

    fig_e = plt.figure(figsize=subfig_size_in(0.275, h_row2_mm))
    ax_e = fig_e.add_axes([0.095, bot_c, 0.850, hgt_c])
    style_axis(ax_e)
    ax_e.axhline(0.74, color=pal["faint"], lw=0.75, zorder=0)
    ax_e.plot([0.95, 0.95], [-0.42, yc[-1] + 0.60], color=pal["ochre"], ls="--", lw=0.95, zorder=3)
    ax_e.text(0.95, -0.75, "0.95 hurdle", fontsize=6.7, color=pal["ochre"], ha="right", va="bottom")

    for i, (mkey, suite, _lbl, ckey, marker) in enumerate(PANEL_CDE_MODELS):
        col, ec = pal[ckey], _edge(pal, ckey)
        sw = sweeps[suite][mkey]
        econ = summaries[suite]["metrics"]["models"]["market_news"][mkey]["economics"]
        ir15 = float(sw["net_ir_lo_active_15bps"])
        ci_lo, ci_hi = float(sw["net_ir_lo_active_15bps_ci"][0]), float(sw["net_ir_lo_active_15bps_ci"][1])
        be_bps = float(sw["breakeven_cost_lo_bps"])
        dsr = float(econ["dsr_lo_active_15bps"])
        sig_ir = ci_lo > 0.0
        ms_c = 5.0 if marker == "*" else (3.4 if marker == "D" else 3.8)

        ax_c.errorbar(
            ir15,
            yc[i],
            xerr=[[ir15 - ci_lo], [ci_hi - ir15]],
            fmt=marker,
            color=ec,
            mfc=col if sig_ir else "white",
            mec=ec,
            mew=0.6 if sig_ir else 0.9,
            ms=ms_c,
            elinewidth=1.1 if sig_ir else 0.85,
            capsize=0.0,
            zorder=4,
        )
        bar_col = pal["quantum"] if ckey in ("quantum", "quantum_light") else (pal["ochre"] if ckey == "ochre" else pal["light"])
        ax_d.barh(yc[i], be_bps, height=0.52, color=bar_col, alpha=0.85, edgecolor="none", zorder=2)

        ax_e.plot([0.0, dsr], [yc[i], yc[i]], color=pal["light"], lw=0.95, zorder=2)
        ax_e.plot(dsr, yc[i], marker=marker, color=ec, mfc=col, mec=ec, mew=0.5, ms=ms_c, ls="none", zorder=4)

    ax_c.set_ylim(*ylim_cde)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([row[2] for row in PANEL_CDE_MODELS])
    ax_c.set_xlim(-0.90, 2.15)
    ax_c.set_xticks(np.array([-0.5, 0.0, 1.0, 2.0]))
    ax_c.set_xlabel(r"Net $\mathrm{IR}_{\mathrm{LO}}$ (95% CI)", labelpad=3.4)
    h_s = mlines.Line2D([], [], color=pal["ink"], marker="o", mfc=pal["ink"], ms=3.5, lw=0, label="CI > 0")
    h_n = mlines.Line2D([], [], color=pal["ink"], marker="o", mfc="white", mew=0.9, ms=3.5, lw=0, label="CI incl. 0")
    ax_c.legend(handles=[h_s, h_n], loc="lower center", bbox_to_anchor=(0.48, 1.02), ncol=2, columnspacing=0.85, handletextpad=0.40)
    written.append(save_panel_pdf(fig_c, panels_dir, "fig06c_net_ir"))
    plt.close(fig_c)

    ax_d.set_ylim(*ylim_cde)
    ax_d.tick_params(axis="y", left=False, labelleft=False)
    ax_d.set_xlim(0.0, 48.0)
    ax_d.set_xticks(np.array([0, 15, 30, 45]))
    ax_d.set_xlabel(r"Break-even $c^*$ (bps)", labelpad=3.4)
    written.append(save_panel_pdf(fig_d, panels_dir, "fig06d_breakeven"))
    plt.close(fig_d)

    ax_e.set_ylim(*ylim_cde)
    ax_e.tick_params(axis="y", left=False, labelleft=False)
    ax_e.set_xlim(0.0, 1.03)
    ax_e.set_xticks(np.array([0.0, 0.5, 1.0]))
    ax_e.xaxis.set_minor_locator(mticker.FixedLocator([0.25, 0.75]))
    ax_e.set_xlabel(r"Deflated Sharpe $\mathrm{DSR}_{\mathrm{LO}}$", labelpad=3.4)
    ax_e.text(0.89, yc[-1] - 0.15, "Max (70): 0.719", fontsize=6.7, color=pal["slate"], ha="right", va="bottom")
    written.append(save_panel_pdf(fig_e, panels_dir, "fig06e_dsr"))
    plt.close(fig_e)
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
