"""Figure 4: Out-of-sample cross-sectional RankIC forest across 13 models (a), paired de-quantization
and news-placebo contrasts (b), Hilbert-dimension parameter frontier (c), and commuting Gibbs-RBF
inverse-temperature sweep beta in [0.05, 16] (d; 37,048 rows, 759 days; Sec. 4.1).

Usage: python -I experiments/figures/fig04_dequant.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in

MODELS_13: list[tuple[str, str, str, str, str, int]] = [
    ("mom21_lowvol", "classical_controls", "Mom21-LowVol", "ink", "*", 0),
    ("qcml_herm_d8", "qcml_benchmark", r"QCML Herm ($d=8$)", "quantum", "o", 1),
    ("qcml_realsym_d11", "qcml_benchmark", r"QCML Real ($d=11$)", "quantum_light", "s", 1),
    ("qcml_diag_d64_bstar", "qcml_benchmark", r"QCML Diag ($d=64,\beta^*$)", "ochre", "p", 1),
    ("lgbm", "classical_controls", "LightGBM", "slate", ">", 2),
    ("hgbdt", "classical_controls", "HistGBDT", "ink", "^", 2),
    ("vqc_n3_cyclic_cnot_fullherm", "vqc_controls", "VQC Cyclic (CNOT)", "quantum", "D", 3),
    ("vqc_n3_cyclic_nocnot_fullherm", "vqc_controls", "VQC Cyclic (NoCNOT)", "slate", "d", 3),
    ("vqc_n3_affine_cnot_fullherm", "vqc_controls", "VQC Affine (CNOT)", "quantum_light", "D", 3),
    ("mlp_tanh", "classical_controls", "MLP-tanh", "slate", "s", 4),
    ("rff_krr", "classical_controls", "RFF-KRR", "muted", "o", 4),
    ("film_mlp_iso", "classical_controls", "FiLM-MLP (iso)", "muted", "P", 4),
    ("ridge", "classical_controls", "Ridge", "muted", "v", 4),
]

BETA_SWEEP: list[tuple[float, str]] = [
    (0.05, "qcml_diag_d64_b0.05"),
    (0.25, "qcml_diag_d64_b0.25"),
    (1.0, "qcml_diag_d64_b1"),
    (2.0, "qcml_diag_d64_b2"),
    (4.0, "qcml_diag_d64_b4"),
    (8.0, "qcml_diag_d64_b8"),
    (16.0, "qcml_diag_d64_b16"),
]


def _edge(pal: dict[str, str], ckey: str) -> str:
    return pal["quantum"] if ckey == "quantum_light" else pal[ckey]


def build_figure(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    rdir = runs_dir()
    summaries = {
        s: load_json(rdir / s / "summary.json")
        for s in ("classical_controls", "vqc_controls", "qcml_benchmark")
    }
    paired_qcml = load_json(rdir / "qcml_benchmark" / "paired_tests_all.json")
    vqc_ent = load_json(rdir / "vqc_controls" / "vqc_entanglement_tests.json")
    mcs_cls = load_json(rdir / "classical_controls" / "mcs_and_paired.json")
    opswap = load_json(rdir / "qcml_benchmark" / "opswap_grid.json")
    mcs_mn_ic = opswap["market_news"]["mcs_unified_ic"]

    pal = setup_mpl(base_pt=7.6, tick_pt=7.0)
    panels_dir = out_dir / "panels"
    written: list[Path] = []

    mom_rec = summaries["classical_controls"]["metrics"]["models"]["market_news"]["mom21_lowvol"]
    mom_ic = float(mom_rec["oos_rank_ic_single_mean"])
    mom_se = float(mom_rec["single_seed"]["rank_ic_nw_se"])

    fig_a = plt.figure(figsize=subfig_size_in(0.485, 68.0))
    ax_a = fig_a.add_axes([0.420, 0.165, 0.550, 0.735])
    style_axis(ax_a)
    z_95, dodge = 1.96, 0.19
    y_coords, dividers = [], []
    y_cur, prev_grp = 0.0, 0
    for idx, row in enumerate(MODELS_13):
        grp = row[5]
        if idx > 0:
            if grp != prev_grp:
                dividers.append(y_cur + 0.70)
                y_cur += 1.40
            else:
                y_cur += 1.12
        y_coords.append(y_cur)
        prev_grp = grp

    ax_a.axvline(0.0, color=pal["light"], ls="--", lw=0.70, zorder=1)
    ax_a.plot([mom_ic, mom_ic], [-0.35, y_coords[-1] + 0.65], color=pal["ink"], ls="-.", lw=0.80, zorder=1)
    for div_y in dividers:
        ax_a.axhline(div_y, color=pal["faint"], ls="-", lw=0.75, zorder=0)

    y_labels_a: list[str] = []
    for idx, (mkey, src, label, ckey, marker, _grp) in enumerate(MODELS_13):
        yc = y_coords[idx]
        col, ec = pal[ckey], _edge(pal, ckey)
        in_mcs = int(mcs_mn_ic.get(mkey, {}).get("in_mcs_05", 0))
        y_labels_a.append(f"{label} \u2020" if in_mcs == 0 else label)
        if mkey == "mom21_lowvol":
            ax_a.errorbar(mom_ic, yc, xerr=z_95 * mom_se, fmt="*", color=col, mfc=col, mec=col, ms=5.4, elinewidth=1.05, capsize=0.0, zorder=5)
            continue
        d_mo = summaries[src]["metrics"]["models"]["market_only"][mkey]
        d_mn = summaries[src]["metrics"]["models"]["market_news"][mkey]
        ic_mo, se_mo = float(d_mo["oos_rank_ic_single_mean"]), float(d_mo["single_seed"]["rank_ic_nw_se"])
        ic_mn, se_mn = float(d_mn["oos_rank_ic_single_mean"]), float(d_mn["single_seed"]["rank_ic_nw_se"])
        ms = 3.5 if marker not in ("D", "d") else 3.2
        ax_a.errorbar(ic_mo, yc + dodge, xerr=z_95 * se_mo, fmt=marker, color=ec, mfc="white", mec=ec, mew=0.80, ms=ms, elinewidth=0.75, capsize=0.0, alpha=0.85, zorder=3)
        ax_a.errorbar(ic_mn, yc - dodge, xerr=z_95 * se_mn, fmt=marker, color=ec, mfc=col, mec=ec, mew=0.60, ms=ms + 0.15, elinewidth=1.05, capsize=0.0, zorder=4)

    ax_a.text(mom_ic, -0.58, f"+{mom_ic:.4f}", fontsize=6.8, color=pal["ink"], ha="center", va="bottom")
    ax_a.set_yticks(y_coords)
    ax_a.set_yticklabels(y_labels_a)
    for tick_lbl, (mkey, *_) in zip(ax_a.get_yticklabels(), MODELS_13):
        if int(mcs_mn_ic.get(mkey, {}).get("in_mcs_05", 0)) == 0:
            tick_lbl.set_color(pal["slate"])
    ax_a.set_ylim(y_coords[-1] + 0.65, -1.15)
    ax_a.set_xlim(-0.012, 0.128)
    ax_a.set_xticks(np.array([0.00, 0.04, 0.08, 0.12]))
    ax_a.xaxis.set_minor_locator(mticker.FixedLocator([0.02, 0.06, 0.10]))
    ax_a.set_xlabel(r"Single-seed $\overline{\mathrm{RankIC}}$ ($\pm 95\%$ CI)", labelpad=3.8)
    h_mn = mlines.Line2D([], [], color=pal["ink"], marker="o", mfc=pal["ink"], mec=pal["ink"], ms=3.5, lw=1.0, label="M+N")
    h_mo = mlines.Line2D([], [], color=pal["ink"], marker="o", mfc="white", mec=pal["ink"], mew=0.8, ms=3.5, lw=0.75, label="MKT")
    h_dag = mlines.Line2D([], [], color="none", marker="", lw=0, label="\u2020 excl. MCS")
    ax_a.legend(handles=[h_mn, h_mo, h_dag], loc="lower center", bbox_to_anchor=(0.45, 1.03), ncol=3, columnspacing=0.85, handlelength=1.15, handletextpad=0.42, borderpad=0.0)
    written.append(save_panel_pdf(fig_a, panels_dir, "fig04a_oos_rankic"))
    plt.close(fig_a)

    fig_b = plt.figure(figsize=subfig_size_in(0.485, 68.0))
    ax_b = fig_b.add_axes([0.420, 0.165, 0.550, 0.735])
    style_axis(ax_b)
    b_groups = [
        (
            "Algebra / CNOT",
            [
                (r"Herm $-$ Real", paired_qcml["market_news"]["herm_d8_vs_realsym_d11"]["ic"], "quantum"),
                (r"Herm $-$ Diag ($\beta^*$)", paired_qcml["market_news"]["herm_d8_vs_diag_bstar"]["ic"], "ochre"),
                (r"VQC Cyc. ($\Delta$CNOT)", vqc_ent["market_news"]["cyclic_fullherm"]["ens_paired_ic"], "quantum"),
                (r"VQC Aff. ($\Delta$CNOT)", vqc_ent["market_news"]["affine_fullherm"]["ens_paired_ic"], "quantum"),
            ],
        ),
        (
            r"Anchor ($\lambda_E$: $0.05 - 0$)",
            [
                (r"QCML Herm ($d=8$)", paired_qcml["market_news"]["herm_d8_vs_le0"]["ic"], "quantum"),
                (r"QCML Real ($d=11$)", paired_qcml["market_news"]["realsym_d11_vs_le0"]["ic"], "quantum"),
            ],
        ),
        (
            "News shuffle placebo",
            [
                (r"QCML Herm ($d=8$) *", paired_qcml["market_news"]["herm_d8_vs_newsshuf"]["ic"], "quantum"),
                (r"QCML Diag ($\beta=2$) *", paired_qcml["market_news"]["diag_b2_vs_newsshuf"]["ic"], "ochre"),
                ("RFF-KRR", mcs_cls["news_placebo"]["rff_krr"]["real_minus_shuffled_ic"], "slate"),
                ("HistGBDT", mcs_cls["news_placebo"]["hgbdt"]["real_minus_shuffled_ic"], "slate"),
            ],
        ),
    ]
    ax_b.axvline(0.0, color=pal["light"], ls="--", lw=0.70, zorder=1)
    b_yticks, b_ylabels = [], []
    cur_y, row_step = 0.0, 0.90
    for g_idx, (g_title, rows) in enumerate(b_groups):
        if g_idx > 0:
            div_y = cur_y - 0.18
            ax_b.axhline(div_y, color=pal["faint"], ls="-", lw=0.75, zorder=0)
            hdr_y, cur_y = div_y + 0.30, div_y + 1.24
        else:
            hdr_y, cur_y = -0.92, 0.0
        ax_b.text(0.044, hdr_y, g_title, fontsize=6.8, style="italic", color=pal["slate"], ha="right", va="top")
        for row_lbl, rec, ckey in rows:
            diff, lo, hi = float(rec["diff_mean"]), float(rec["boot_ci_lo"]), float(rec["boot_ci_hi"])
            sig = (lo > 0.0) or (hi < 0.0)
            col = pal[ckey] if sig else pal["slate"]
            bar_col = pal[ckey] if sig else pal["muted"]
            ax_b.errorbar(
                diff, cur_y, xerr=[[diff - lo], [hi - diff]], fmt="D" if sig else "s", color=bar_col, mfc=col, mec=col,
                ms=3.7 if sig else 3.0, elinewidth=1.15 if sig else 0.85, capsize=0.0, zorder=5 if sig else 3,
            )
            b_yticks.append(cur_y)
            b_ylabels.append(row_lbl)
            cur_y += row_step

    ax_b.set_yticks(b_yticks)
    ax_b.set_yticklabels(b_ylabels)
    ax_b.set_ylim(cur_y + 0.20, -1.25)
    ax_b.set_xlim(-0.014, 0.0455)
    ax_b.set_xticks(np.array([-0.01, 0.00, 0.02, 0.04]))
    ax_b.set_xticklabels(["\u22120.01", "0.00", "+0.02", "+0.04"])
    ax_b.xaxis.set_minor_locator(mticker.FixedLocator([0.01, 0.03]))
    ax_b.set_xlabel(r"Paired $\Delta\overline{\mathrm{RankIC}}$ ($95\%$ CI)", labelpad=3.8)
    h_sig = mlines.Line2D([], [], color=pal["quantum"], marker="D", ms=3.5, lw=1.1, label="CI excl. 0 (*)")
    h_nul = mlines.Line2D([], [], color=pal["muted"], marker="s", mfc=pal["slate"], mec=pal["slate"], ms=3.0, lw=0.85, label="CI incl. 0")
    ax_b.legend(handles=[h_sig, h_nul], loc="lower center", bbox_to_anchor=(0.45, 1.025), ncol=2, columnspacing=1.0, handletextpad=0.42)
    written.append(save_panel_pdf(fig_b, panels_dir, "fig04b_paired_contrasts"))
    plt.close(fig_b)

    fig_c = plt.figure(figsize=subfig_size_in(0.485, 48.0))
    ax_c = fig_c.add_axes([0.215, 0.235, 0.750, 0.710])
    style_axis(ax_c)
    ax_c.set_xscale("log")
    ax_c.axvspan(960, 1254, color=pal["faint"], alpha=0.95, zorder=0)
    ax_c.axhline(mom_ic, color=pal["ink"], ls="-.", lw=0.80, zorder=2)
    ax_c.text(13.0, mom_ic + 0.0028, f"Mom21-LowVol (+{mom_ic:.4f})", fontsize=6.8, color=pal["ink"], ha="left", va="bottom")
    ax_c.text(1460, 0.016, "Iso-param", fontsize=6.8, style="italic", color=pal["slate"], ha="left", va="bottom")

    q_mn = summaries["qcml_benchmark"]["metrics"]["models"]["market_news"]
    herm_keys = ["qcml_herm_d2", "qcml_herm_d4", "qcml_herm_d8", "qcml_herm_d16"]
    herm_x = np.array([int(q_mn[k]["n_params"]) for k in herm_keys], dtype=float)
    herm_y = np.array([float(q_mn[k]["oos_rank_ic_single_mean"]) for k in herm_keys], dtype=float)
    herm_sd = np.array([float(q_mn[k]["oos_rank_ic_seed_std"]) for k in herm_keys], dtype=float)
    ax_c.fill_between(herm_x, herm_y - herm_sd, herm_y + herm_sd, color=pal["quantum_pale"], alpha=0.85, lw=0.0, zorder=2)
    ax_c.plot(herm_x, herm_y, color=pal["quantum"], lw=1.30, marker="o", ms=3.8, mfc=pal["quantum"], mec="white", mew=0.45, zorder=6)

    leader_kw = dict(arrowstyle="-", lw=0.45, color=pal["muted"], shrinkA=1.2, shrinkB=3.2)
    for mkey, src, mfc, mec, mew, mk, lbl, xy_txt, ha, use_l in [
        ("qcml_realsym_d11", "qcml_benchmark", "white", pal["quantum"], 0.95, "s", r"Real $d{=}11$", (2800, 0.0755), "left", True),
        ("qcml_diag_d64_bstar", "qcml_benchmark", pal["ochre"], pal["ochre"], 0.55, "p", r"Diag $d{=}64$", (2800, 0.0515), "left", True),
        ("hgbdt", "classical_controls", pal["ink"], pal["ink"], 0.55, "^", "HistGBDT", (420, 0.0685), "right", True),
        ("ridge", "classical_controls", pal["muted"], pal["muted"], 0.55, "v", "Ridge", (32, 0.0283), "left", False),
    ]:
        d = summaries[src]["metrics"]["models"]["market_news"][mkey]
        xv, yv = float(d["n_params"]), float(d["oos_rank_ic_single_mean"])
        ax_c.plot(xv, yv, marker=mk, mfc=mfc, mec=mec, mew=mew, ms=3.6, ls="none", zorder=8 if mkey == "qcml_realsym_d11" else 7)
        ax_c.annotate(lbl, xy=(xv, yv), xytext=xy_txt, fontsize=6.8, color=pal["ink"], ha=ha, va="center", arrowprops=leader_kw if use_l else None, zorder=9)

    for idx_h, lbl_h, xy_t, ha_h, use_l in [
        (0, r"$d{=}2$", (54, 0.0365), "right", False),
        (2, r"Herm $d{=}8$", (500, 0.0815), "right", True),
        (3, r"$d{=}16$", (6500, 0.0645), "left", False),
    ]:
        ax_c.annotate(
            lbl_h, xy=(herm_x[idx_h], herm_y[idx_h]), xytext=xy_t, fontsize=6.8,
            color=pal["quantum"] if idx_h == 2 else pal["ink"], ha=ha_h, va="center", arrowprops=leader_kw if use_l else None, zorder=8,
        )

    ax_c.set_xlim(10.0, 2.8e4)
    ax_c.set_xticks([1e1, 1e2, 1e3, 1e4])
    ax_c.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax_c.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_c.set_ylim(0.013, 0.102)
    ax_c.set_yticks(np.array([0.02, 0.04, 0.06, 0.08]))
    ax_c.yaxis.set_minor_locator(mticker.FixedLocator([0.03, 0.05, 0.07, 0.09]))
    ax_c.set_xlabel("Trainable parameters", labelpad=3.4)
    ax_c.set_ylabel(r"M+N $\overline{\mathrm{RankIC}}$", labelpad=4.8)
    written.append(save_panel_pdf(fig_c, panels_dir, "fig04c_hilbert_frontier"))
    plt.close(fig_c)

    fig_d = plt.figure(figsize=subfig_size_in(0.485, 48.0))
    ax_d = fig_d.add_axes([0.215, 0.235, 0.750, 0.710])
    style_axis(ax_d)
    ax_d.set_xscale("log")
    betas = np.array([b for b, _ in BETA_SWEEP], dtype=float)
    diag_sic = np.array([float(q_mn[k]["oos_rank_ic_single_mean"]) for _, k in BETA_SWEEP])
    diag_sd = np.array([float(q_mn[k]["oos_rank_ic_seed_std"]) for _, k in BETA_SWEEP])
    diag_ens = np.array([float(q_mn[k]["oos_rank_ic_ens"]) for _, k in BETA_SWEEP])

    herm_d8 = q_mn["qcml_herm_d8"]
    h_sic, h_sd, h_ens = float(herm_d8["oos_rank_ic_single_mean"]), float(herm_d8["oos_rank_ic_seed_std"]), float(herm_d8["oos_rank_ic_ens"])
    ax_d.axvspan(1.0, 4.0, color=pal["ochre_pale"], alpha=0.65, zorder=0)
    ax_d.axhspan(h_sic - h_sd, h_sic + h_sd, color=pal["quantum_pale"], alpha=0.75, zorder=1)
    ax_d.axhline(h_sic, color=pal["quantum"], ls="-", lw=0.85, zorder=2)
    ax_d.axhline(h_ens, color=pal["quantum"], ls="--", lw=0.90, zorder=2)
    ax_d.fill_between(betas, diag_sic - diag_sd, diag_sic + diag_sd, color=pal["ochre_light"], alpha=0.32, lw=0.0, zorder=3)
    ax_d.plot(betas, diag_sic, color=pal["ochre"], ls="-", lw=1.25, marker="p", ms=4.0, mfc=pal["ochre"], mec="white", mew=0.4, zorder=5)
    ax_d.plot(betas, diag_ens, color=pal["ochre"], ls="--", lw=1.10, marker="p", ms=3.7, mfc="white", mec=pal["ochre"], mew=0.85, zorder=6)
    ax_d.text(0.042, 0.016, r"High-$T$", fontsize=6.8, style="italic", color=pal["slate"], ha="left", va="bottom")
    ax_d.text(2.0, 0.016, "Boltzmann", fontsize=6.8, style="italic", color=pal["ochre"], ha="center", va="bottom")
    ax_d.text(21.0, 0.016, "Voronoi", fontsize=6.8, style="italic", color=pal["slate"], ha="right", va="bottom")
    h_de = mlines.Line2D([], [], color=pal["ochre"], ls="--", lw=1.05, marker="p", ms=3.5, mfc="white", mec=pal["ochre"], mew=0.8, label="Diag ens.")
    h_d1 = mlines.Line2D([], [], color=pal["ochre"], ls="-", lw=1.2, marker="p", ms=3.5, mfc=pal["ochre"], mec=pal["ochre"], label="Diag 1s")
    h_he = mlines.Line2D([], [], color=pal["quantum"], ls="--", lw=0.95, label="Herm ens.")
    h_h1 = mpatches.Patch(facecolor=pal["quantum_pale"], edgecolor=pal["quantum"], lw=0.7, label="Herm 1s")
    ax_d.legend(handles=[h_de, h_d1, h_he, h_h1], loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=0.90, handlelength=1.15, handletextpad=0.42, labelspacing=0.30)
    ax_d.set_xlim(0.032, 26.0)
    ax_d.set_xticks([0.05, 0.25, 1.0, 4.0, 16.0])
    ax_d.set_xticklabels(["0.05", "0.25", "1", "4", "16"])
    ax_d.xaxis.set_minor_locator(mticker.FixedLocator([0.1, 0.5, 2.0, 8.0]))
    ax_d.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d.set_ylim(0.013, 0.102)
    ax_d.set_yticks(np.array([0.02, 0.04, 0.06, 0.08]))
    ax_d.yaxis.set_minor_locator(mticker.FixedLocator([0.03, 0.05, 0.07, 0.09]))
    ax_d.set_xlabel(r"Inverse temperature $\beta$ (Diag $d=64$)", labelpad=3.4)
    ax_d.set_ylabel(r"M+N $\overline{\mathrm{RankIC}}$", labelpad=4.8)
    written.append(save_panel_pdf(fig_d, panels_dir, "fig04d_gibbs_beta_sweep"))
    plt.close(fig_d)

    for p in written:
        print(f"Wrote {p}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build_figure(args.out_dir)


if __name__ == "__main__":
    main()
