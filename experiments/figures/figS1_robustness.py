"""Supplementary Figure S1: Synthetic positive controls (a), 4-target joint market-state
forecasting (b), liquidity/high-rate/horizon robustness (c), and survivorship stress bounds
(d; Supplementary Material Secs. S1-S3).

Usage: python -I experiments/figures/figS1_robustness.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in


def build(runs: Path, out_dir: Path) -> list[Path]:
    b0_syn = load_json(runs / "synthetic_controls" / "synthetic_pos_control.json")
    cls_daily = pd.read_csv(runs / "classical_controls" / "daily_rank_ic.csv")
    qcml_daily = pd.read_csv(runs / "qcml_benchmark" / "daily_rank_ic.csv")
    rob_dir = runs / "multioutput_robustness"
    t4 = load_json(rob_dir / "t4_multioutput.json")
    liq = load_json(rob_dir / "liquidity_and_cohorts.json")
    rob = load_json(rob_dir / "robustness_grid.json")
    surv = load_json(rob_dir / "survivorship_bounds.json")

    pal = setup_mpl(base_pt=7.5, tick_pt=6.9)
    supp_dir = out_dir / "supp_panels"
    written: list[Path] = []

    fig_a = plt.figure(figsize=subfig_size_in(0.485, 54.0, supp=True))
    ax_a = fig_a.add_axes([0.430, 0.220, 0.540, 0.730])
    style_axis(ax_a)

    s2 = b0_syn["b0a"]["s2_supervised_3way"]
    b0b = b0_syn["b0b"]
    rows_a = [
        (8.5, r"Herm ($d=2$, 24 par.)", float(s2["hermitian"]["test_r2"]), 0.0, "quantum", "o"),
        (7.5, r"Real ($d=2$, 18 par.)", float(s2["real_sym"]["test_r2"]), 0.0, "quantum_light", "s"),
        (6.5, r"Diag ($d=2$, 12 par.)", float(s2["commuting_diag_gibbs"]["test_r2"]), 0.0, "ochre", "p"),
        (5.0, r"Herm ($d=3$, 36 par.)", float(b0b["herm_d3"]["mean_r2"]), float(b0b["herm_d3"]["std_r2"]), "quantum", "o"),
        (4.0, r"Real ($d=3$, 24 par.)", float(b0b["realsym_d3"]["mean_r2"]), float(b0b["realsym_d3"]["std_r2"]), "quantum_light", "s"),
        (3.0, r"Diag ($d=3$, 12 par.)", float(b0b["diag_d3"]["mean_r2"]), float(b0b["diag_d3"]["std_r2"]), "ochre", "p"),
        (2.0, r"Diag ($d=9$, 36 par.)", float(b0b["diag_d9"]["mean_r2"]), float(b0b["diag_d9"]["std_r2"]), "ochre", "p"),
        (1.0, "HistGBDT (3,720 par.)", float(b0b["hgbdt"]["mean_r2"]), float(b0b["hgbdt"]["std_r2"]), "ink", "^"),
        (0.0, "Ridge (4 par.)", float(b0b["ridge"]["mean_r2"]), float(b0b["ridge"]["std_r2"]), "muted", "v"),
    ]
    ax_a.axvline(0.0, color=pal["light"], ls=":", lw=0.70, zorder=1)
    ax_a.axvline(1.0, color=pal["light"], ls=":", lw=0.65, zorder=1)
    ax_a.axhline(5.75, color=pal["faint"], ls="-", lw=0.80, zorder=1)

    for y, _lbl, val, err, ckey, mk in rows_a:
        col = pal[ckey]
        ec = pal["quantum"] if ckey == "quantum_light" else col
        ax_a.plot([0.0, val], [y, y], color=col, lw=1.05, alpha=0.55, zorder=2)
        if err > 0.0:
            ax_a.errorbar(val, y, xerr=err, fmt=mk, color=ec, mfc=col, mec=ec, ms=3.9, elinewidth=0.95, capsize=1.5, zorder=4)
        else:
            ax_a.plot(val, y, marker=mk, mfc=col, mec=ec, ms=3.9, ls="none", zorder=4)

    ax_a.text(0.50, 6.5, r"$S^2$ ($q=3$)", fontsize=6.7, style="italic", color=pal["slate"], ha="left", va="center")
    ax_a.text(0.42, 0.2, r"Quadrupole $u_x u_y$", fontsize=6.7, style="italic", color=pal["slate"], ha="left", va="center")
    ax_a.set_yticks([r[0] for r in rows_a])
    ax_a.set_yticklabels([r[1] for r in rows_a])
    ax_a.set_xlim(-0.08, 1.12)
    ax_a.set_ylim(-0.65, 9.15)
    ax_a.set_xticks(np.array([0.0, 0.5, 1.0]))
    ax_a.xaxis.set_minor_locator(mticker.FixedLocator([0.25, 0.75]))
    ax_a.set_xlabel(r"Out-of-sample ${R^2}_{\,\mathrm{OOS}}$", labelpad=3.4)
    written.append(save_panel_pdf(fig_a, supp_dir, "figS1a_synthetic"))
    plt.close(fig_a)

    fig_b = plt.figure(figsize=subfig_size_in(0.485, 54.0, supp=True))
    ax_b = fig_b.add_axes([0.220, 0.220, 0.750, 0.730])
    style_axis(ax_b)

    targets = [("y_h5_cs", "Return"), ("fwd_dd_5d", "Drawdown"), ("fwd_vol_5d", "Real. Vol"), ("fwd_vol_shock_5d", "Vol Shock")]
    models_b = [
        ("qcml_herm_d8_t4", "QCML joint", pal["quantum"]),
        ("qcml_herm_d8_4x_q1", r"QCML $4{\times}1$", pal["quantum_light"]),
        ("mlp_tanh_multi", "MLP-tanh", pal["slate"]),
        ("ridge_multi", "Ridge", pal["light"]),
    ]
    x = np.arange(len(targets), dtype=float)
    bar_w = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w
    for idx, (m_key, m_lbl, col) in enumerate(models_b):
        md = t4[m_key]
        vals = [float(md["per_target"][tk]["oos_rank_ic_single_mean"]) for tk, _ in targets]
        ci95 = [1.96 * float(md["per_target"][tk]["single_seed"]["rank_ic_nw_se"]) for tk, _ in targets]
        ax_b.bar(x + offsets[idx], vals, width=bar_w * 0.88, color=col, edgecolor="none", label=m_lbl, zorder=3)
        ax_b.errorbar(x + offsets[idx], vals, yerr=ci95, fmt="none", ecolor=pal["ink"], elinewidth=0.70, capsize=1.3, zorder=4)

    ax_b.axhline(0.0, color=pal["light"], ls=":", lw=0.70, zorder=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([tl for _, tl in targets])
    ax_b.set_ylim(-0.02, 0.49)
    ax_b.set_yticks(np.array([0.0, 0.2, 0.4]))
    ax_b.yaxis.set_minor_locator(mticker.FixedLocator([0.1, 0.3]))
    ax_b.set_xlabel(r"Forecast target ($h=5$)", labelpad=3.4)
    ax_b.set_ylabel(r"Single-seed $\overline{\mathrm{RankIC}}$", labelpad=4.6)
    ax_b.legend(loc="upper left", ncol=1, handlelength=1.15, handletextpad=0.42, labelspacing=0.40)
    written.append(save_panel_pdf(fig_b, supp_dir, "figS1b_multioutput"))
    plt.close(fig_b)

    fig_c = plt.figure(figsize=subfig_size_in(0.485, 66.0, supp=True))
    ax_c = fig_c.add_axes([0.220, 0.205, 0.750, 0.565])
    style_axis(ax_c)

    m2425_cls = cls_daily["date"] >= "2024-04-01"
    m2425_qcml = qcml_daily["date"] >= "2024-04-01"
    split_2425 = {
        "qcml_realsym_d11": float(qcml_daily.loc[m2425_qcml, "mn_qcml_realsym_d11"].mean()),
        "qcml_herm_d8": float(qcml_daily.loc[m2425_qcml, "mn_qcml_herm_d8"].mean()),
        "mom21_lowvol": float(cls_daily.loc[m2425_cls, "mn_mom21_lowvol"].mean()),
        "hgbdt": float(cls_daily.loc[m2425_cls, "mn_hgbdt"].mean()),
        "mlp_tanh": float(cls_daily.loc[m2425_cls, "mn_mlp_tanh"].mean()),
        "ridge": float(cls_daily.loc[m2425_cls, "mn_ridge"].mean()),
    }
    cats = [
        ("Top-30", {m: float(liq[m]["top30_ens_ic"]) for m in split_2425}),
        ("31\u201350", {m: float(liq[m]["ranks_31_50_ens_ic"]) for m in split_2425}),
        ("2024\u201325", split_2425),
        (r"$h=15$", {m: float(rob["h15_purged_gap15"][m]["oos_rank_ic_ens"]) for m in ("qcml_herm_d8", "hgbdt", "mlp_tanh", "ridge")}),
        ("126d roll", {m: float(rob["w126_rolling_train"][m]["oos_rank_ic_ens"]) for m in ("qcml_herm_d8", "hgbdt", "mlp_tanh", "ridge")}),
    ]
    mspecs = [
        ("qcml_herm_d8", r"Herm ($d=8$)", "quantum", "o", 3.6),
        ("qcml_realsym_d11", r"Real ($d=11$)", "quantum_light", "s", 3.5),
        ("mom21_lowvol", "Mom21-LowVol", "ink", "*", 4.6),
        ("hgbdt", "HistGBDT", "ink", "^", 3.6),
        ("mlp_tanh", "MLP-tanh", "slate", "s", 3.4),
        ("ridge", "Ridge", "muted", "v", 3.5),
    ]
    smap = {s[0]: s for s in mspecs}
    for ci, (_, mvals) in enumerate(cats):
        present = [m for m, *_ in mspecs if m in mvals]
        offs = (np.arange(len(present)) - 0.5 * (len(present) - 1)) * 0.145
        for dx, m in zip(offs, present):
            _, _, ck, mk, ms = smap[m]
            col = pal[ck]
            ec = pal["quantum"] if ck == "quantum_light" else col
            ax_c.plot(ci + dx, mvals[m], marker=mk, mfc=col, mec=ec, mew=0.6, ms=ms, ls="none", zorder=4)

    ax_c.axvline(2.5, color=pal["faint"], lw=0.85, zorder=1)
    hc = [
        mlines.Line2D([], [], marker=mk, mfc=pal[ck], mec=pal["quantum"] if ck == "quantum_light" else pal[ck], ms=ms * 0.92, lw=0, label=lbl)
        for _, lbl, ck, mk, ms in mspecs
    ]
    ax_c.legend(
        handles=hc,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.035),
        ncol=3,
        columnspacing=0.68,
        handletextpad=0.38,
        labelspacing=0.32,
    )
    ax_c.set_xticks(np.arange(len(cats)))
    ax_c.set_xticklabels([c[0] for c in cats])
    ax_c.set_xlim(-0.55, len(cats) - 0.45)
    ax_c.set_ylim(0.0, 0.106)
    ax_c.set_yticks(np.array([0.0, 0.05, 0.10]))
    ax_c.yaxis.set_minor_locator(mticker.FixedLocator([0.025, 0.075]))
    ax_c.set_xlabel("Evaluation regime", labelpad=3.4)
    ax_c.set_ylabel(r"Ensemble $\overline{\mathrm{RankIC}}$", labelpad=4.8)
    written.append(save_panel_pdf(fig_c, supp_dir, "figS1c_regime_robustness"))
    plt.close(fig_c)

    fig_d = plt.figure(figsize=subfig_size_in(0.485, 66.0, supp=True))
    ax_d = fig_d.add_axes([0.340, 0.205, 0.630, 0.565])
    style_axis(ax_d)

    m_ord = [
        ("mom21_lowvol", "Mom21-LowVol"),
        ("qcml_herm_d8", r"QCML Herm ($d=8$)"),
        ("mlp_tanh", "MLP-tanh"),
        ("hgbdt", "HistGBDT"),
        ("rff_krr", "RFF-KRR"),
    ]
    ys = np.arange(len(m_ord) - 1, -1, -1, dtype=float) * 1.85
    st_specs = [
        ("k_3_exits_per_fold", 0.16, pal["slate"], pal["slate"], "s", r"$k=3$ exits"),
        ("k_5_exits_per_fold", -0.18, "white", pal["ochre"], "^", r"$k=5$ exits"),
        ("k_7_exits_per_fold", -0.52, pal["ochre"], pal["ochre"], "v", r"$k=7$ exits"),
    ]
    for y, (m_key, _) in zip(ys, m_ord):
        ent = surv[m_key]
        b_ic = float(ent["baseline_rank_ic"])
        p_ic = float(ent["pre2020_continuous"]["rank_ic"])
        ax_d.plot([b_ic, b_ic], [y - 0.60, y + 0.58], color=pal["light"], ls=":", lw=0.75, zorder=1)
        ax_d.plot([b_ic, p_ic], [y + 0.50, y + 0.50], color=pal["quantum_light"], lw=0.90, zorder=2)
        ax_d.plot(b_ic, y + 0.50, marker="o", mfc=pal["ink"], mec=pal["ink"], ms=3.2, ls="none", zorder=5)
        ax_d.plot(p_ic, y + 0.50, marker="D", mfc="white", mec=pal["quantum"], mew=0.9, ms=3.2, ls="none", zorder=5)
        for kk, dy, mfc, mec, mk, _ in st_specs:
            kd = ent["distressed_exit_stress"][kk]
            mv, p05 = float(kd["rank_ic_mean"]), float(kd["rank_ic_p05"])
            ax_d.plot([p05, mv], [y + dy, y + dy], color=mec, lw=1.05, zorder=3)
            ax_d.plot(mv, y + dy, marker=mk, mfc=mfc, mec=mec, mew=0.75, ms=2.55, ls="none", zorder=4)

    hd = [
        mlines.Line2D([], [], marker="o", mfc=pal["ink"], mec=pal["ink"], ms=3.2, lw=0, label="Base"),
        mlines.Line2D([], [], marker="D", mfc="white", mec=pal["quantum"], mew=0.9, ms=3.2, lw=0, label="Pre-2020"),
        mlines.Line2D([], [], color=pal["slate"], marker="s", mfc=pal["slate"], mec=pal["slate"], ms=2.7, lw=1.0, label=r"$k=3$ exits"),
        mlines.Line2D([], [], color=pal["ochre"], marker="^", mfc="white", mec=pal["ochre"], mew=0.8, ms=2.7, lw=1.0, label=r"$k=5$ exits"),
        mlines.Line2D([], [], color=pal["ochre"], marker="v", mfc=pal["ochre"], mec=pal["ochre"], ms=2.7, lw=1.0, label=r"$k=7$ exits"),
    ]
    ax_d.legend(
        handles=hd,
        loc="lower center",
        bbox_to_anchor=(0.42, 1.035),
        ncol=3,
        columnspacing=0.78,
        handlelength=1.05,
        handletextpad=0.40,
        labelspacing=0.32,
    )
    ax_d.set_yticks(ys)
    ax_d.set_yticklabels([m[1] for m in m_ord])
    ax_d.set_ylim(-0.90, ys[0] + 0.85)
    ax_d.set_xlim(0.046, 0.098)
    ax_d.set_xticks(np.array([0.05, 0.07, 0.09]))
    ax_d.xaxis.set_minor_locator(mticker.FixedLocator([0.06, 0.08]))
    ax_d.set_xlabel(r"Ensemble $\overline{\mathrm{RankIC}}$", labelpad=3.4)
    written.append(save_panel_pdf(fig_d, supp_dir, "figS1d_survivorship"))
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
