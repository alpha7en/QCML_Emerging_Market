"""Figure 2: Exact positive-semidefinite shifted Stark level fans along a fixed feature ray for
anchored Hermitian (d = 8, kappa_nc = 0.30; a) and untrained GUE(8) operators (b), and 759-day
empirical M+N spectrum superimposed on 10 untrained GUE(8) draws: ground-state cone E_0 (c), raw
gap Delta E (d), and scale-free gap tilde{Delta E} (e; Sec. 2.2).

Usage: python -I experiments/figures/fig02_geometry.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in


def _kappa_nc(a_ops: np.ndarray) -> float:
    p = a_ops.shape[0]
    norms = [float(np.linalg.norm(a_ops[j], ord="fro")) for j in range(p)]
    num, den = 0.0, 1e-12
    for j in range(p):
        for k in range(j + 1, p):
            comm = a_ops[j] @ a_ops[k] - a_ops[k] @ a_ops[j]
            num += float(np.linalg.norm(comm, ord="fro"))
            den += norms[j] * norms[k]
    return float(num / den)


def _draw_gue_ops(seed: int, d: int = 8, p: int = 18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real_part = rng.normal(scale=1.0 / math.sqrt(2.0 * d), size=(p, d, d))
    imag_part = rng.normal(scale=1.0 / math.sqrt(2.0 * d), size=(p, d, d))
    return 0.5 * (real_part + np.swapaxes(real_part, -1, -2)) + 1j * 0.5 * (
        imag_part - np.swapaxes(imag_part, -1, -2)
    )


def _compute_psd_rays(a_gue: np.ndarray, x_hat: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate diagonal and GUE operators to match trained kappa_nc = 0.300 and evaluate
    shifted Stark eigenvalues tilde{E}_alpha(r) = eig(C_0 - r sum_k x_hat_k A_k) for r in [0, 9].
    """
    p, d, _ = a_gue.shape
    rng_d = np.random.default_rng(2026)
    d_ops = np.array([np.diag(rng_d.normal(scale=1.0 / math.sqrt(2.0), size=d)) for _ in range(p)], dtype=np.complex128)

    lo, hi = 0.0, 1.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if _kappa_nc((1.0 - mid) * d_ops + mid * a_gue) < 0.300:
            lo = mid
        else:
            hi = mid
    a_herm = (1.0 - 0.5 * (lo + hi)) * d_ops + 0.5 * (lo + hi) * a_gue
    a_herm *= float(np.mean([np.linalg.norm(a, "fro") for a in a_gue])) / float(
        np.mean([np.linalg.norm(a, "fro") for a in a_herm])
    )

    r_grid = np.linspace(0.0, 9.0, 361)

    def _eval(a_ops: np.ndarray) -> np.ndarray:
        c0 = 0.5 * np.sum(a_ops @ a_ops, axis=0)
        a_x = np.tensordot(x_hat, a_ops, axes=([0], [0]))
        return np.array([np.linalg.eigvalsh(c0 - r * a_x) for r in r_grid], dtype=np.float64)

    return {"r_grid": r_grid, "e_herm": _eval(a_herm), "e_gue": _eval(a_gue)}


def _binned_gue_band(x: np.ndarray, y_draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.concatenate([np.quantile(x, np.linspace(0.0, 0.92, 13)), np.array([6.05, 6.95, 7.85, 9.05])])
    imin = int(np.argmin(x))
    bx, bm, bs = [float(x[imin])], [float(y_draws[:, imin].mean())], [float(y_draws[:, imin].std(ddof=1))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if int(m.sum()) >= 3:
            xm = float(x[m].mean())
            if xm > bx[-1] + 0.08:
                dm = y_draws[:, m].mean(axis=1)
                bx.append(xm)
                bm.append(float(dm.mean()))
                bs.append(float(dm.std(ddof=1)))
    imax = int(np.argmax(x))
    if float(x[imax]) > bx[-1] + 0.12:
        bx.append(float(x[imax]))
        bm.append(float(y_draws[:, imax].mean()))
        bs.append(float(y_draws[:, imax].std(ddof=1)))
    return np.asarray(bx), np.asarray(bm), np.asarray(bs)


def _draw_ray_panel(
    out_dir: Path,
    stem: str,
    r_grid: np.ndarray,
    e_sh: np.ndarray,
    tag_text: str,
    ylim: tuple[float, float],
    yticks: np.ndarray,
    pal: dict[str, str],
    *,
    show_ylabel: bool,
) -> Path:
    fig = plt.figure(figsize=subfig_size_in(0.32, 42.5))
    ax = fig.add_axes([0.255 if show_ylabel else 0.165, 0.250, 0.650 if show_ylabel else 0.740, 0.700])
    style_axis(ax)

    ax.axvspan(3.4, 4.4, color=pal["quantum_pale"], alpha=0.65, lw=0, zorder=0)
    ax.axvspan(6.0, 8.9, color=pal["ochre_pale"], alpha=0.65, lw=0, zorder=0)

    r_fl = r_grid[0.5 * r_grid * r_grid <= -ylim[0] + 0.4]
    ax.plot(r_fl, -0.5 * r_fl * r_fl, color=pal["slate"], ls=":", lw=0.90, zorder=2)

    for alpha in range(2, e_sh.shape[1]):
        ax.plot(r_grid, e_sh[:, alpha], color=pal["light"], lw=0.65, zorder=3)
    ax.plot(r_grid, e_sh[:, 1], color=pal["ink"], lw=1.10, zorder=5)
    ax.plot(r_grid, e_sh[:, 0], color=pal["quantum"], lw=1.40, zorder=6)

    idx = int(round(7.6 / 9.0 * (len(r_grid) - 1)))
    y0, y1 = float(e_sh[idx, 0]), float(e_sh[idx, 1])
    ax.annotate(
        "",
        xy=(7.6, y1),
        xytext=(7.6, y0),
        arrowprops=dict(arrowstyle="<->", color=pal["ochre"], lw=0.85, shrinkA=0, shrinkB=0, mutation_scale=5.0),
        zorder=7,
    )
    ax.text(7.55, y0 - 1.25, r"$\Delta E$", fontsize=6.8, color=pal["ochre"], ha="center", va="top", zorder=8)

    ax.text(9.26, float(e_sh[-1, 1]), r"$\tilde{E}_1$", fontsize=6.8, color=pal["ink"], ha="left", va="center", clip_on=False)
    ax.text(9.26, float(e_sh[-1, 0]), r"$\tilde{E}_0$", fontsize=6.8, color=pal["quantum"], ha="left", va="center", clip_on=False)
    ax.text(0.28, 16.0, tag_text, fontsize=6.8, color=pal["ink"], ha="left", va="top")
    if show_ylabel:
        ax.text(3.9, 10.9, "Normal", fontsize=6.6, color=pal["slate"], ha="center", va="bottom")
        ax.text(7.45, 10.9, "Stress", fontsize=6.6, color=pal["ochre"], ha="center", va="bottom")
        ax.text(0.28, -6.2, r"$-0.5\,r^2$", fontsize=6.8, color=pal["slate"], ha="left", va="center")

    ax.set_xlim(0.0, 9.0)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.array([0, 3, 6, 9]))
    ax.xaxis.set_minor_locator(mticker.FixedLocator([1.5, 4.5, 7.5]))
    ax.set_yticks(yticks)
    ax.set_xlabel(r"Ray radius $r = \|\mathbf{x}\|_2$", labelpad=3.4)
    if show_ylabel:
        ax.set_ylabel(r"$\tilde{E}_\alpha = E_\alpha - 0.5\,r^2$", labelpad=4.4)

    pdf_path = save_panel_pdf(fig, out_dir, stem)
    plt.close(fig)
    return pdf_path


def build_figure(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    spec_dir = runs_dir() / "spectral_stress"
    df = pd.read_csv(spec_dir / "daily_regime_series.csv")
    roc_data = load_json(spec_dir / "roc_and_spectral.json")
    mn_roc = roc_data["market_news"]
    sup_meta = mn_roc["models"]["qcml_sup_le005"]
    gue_json = mn_roc["gue_null_759d"]

    with np.load(spec_dir / "gue_daily_null.npz", allow_pickle=False) as z:
        x_norm = z["n1_day"].astype(np.float64)
        e0_draws = z["e0_draws"].astype(np.float64)
        de_draws = z["de_draws"].astype(np.float64)
        det_draws = z["det_draws"].astype(np.float64)
        x_hat = z["x_hat"].astype(np.float64)

    a_gue_0 = _draw_gue_ops(1000, d=8, p=18)
    rays = _compute_psd_rays(a_gue_0, x_hat)
    pal = setup_mpl(base_pt=7.6, tick_pt=7.0)
    panels_dir = out_dir / "panels"

    shared_ylim = (-8.8, 16.6)
    shared_yticks = np.array([-5, 0, 5, 10, 15])
    written: list[Path] = [
        _draw_ray_panel(
            panels_dir, "fig02a_herm_stark_fan", rays["r_grid"], rays["e_herm"],
            r"Herm ($d=8$, $\kappa_{\mathrm{nc}}=0.30$)", shared_ylim, shared_yticks, pal, show_ylabel=True,
        ),
        _draw_ray_panel(
            panels_dir, "fig02b_gue_stark_fan", rays["r_grid"], rays["e_gue"],
            r"GUE ($d=8$, $\kappa_{\mathrm{nc}}=0.50$)", shared_ylim, shared_yticks, pal, show_ylabel=False,
        ),
    ]

    stress = df["mn_stress_contemp"].to_numpy(dtype=np.int64) == 1
    normal = ~stress
    x_grid = np.linspace(float(x_norm.min()) - 0.1, float(x_norm.max()) + 0.15, 200)

    fig_c = plt.figure(figsize=subfig_size_in(0.32, 42.5))
    ax_c = fig_c.add_axes([0.245, 0.250, 0.715, 0.700])
    style_axis(ax_c)
    e0 = df["mn_qcml_sup_le005_E0"].to_numpy(dtype=np.float64)
    bx_c, bm_c, bs_c = _binned_gue_band(x_norm, e0_draws)
    ax_c.scatter(x_norm[normal], e0[normal], s=6.0, c=pal["quantum_light"], alpha=0.45, lw=0, zorder=2)
    ax_c.scatter(x_norm[stress], e0[stress], s=8.0, c=pal["ochre"], alpha=0.85, lw=0, zorder=3)
    ax_c.fill_between(bx_c, bm_c - bs_c, bm_c + bs_c, color=pal["faint"], alpha=0.9, lw=0, zorder=1)
    ax_c.plot(bx_c, bm_c, color=pal["slate"], ls="--", lw=1.0, zorder=4)
    ax_c.plot(x_grid, 0.5 * x_grid**2, color=pal["ink"], ls="-", lw=1.05, zorder=5)
    r2_e0 = float(sup_meta["r2_E0_on_norm2"])
    r2_e0_gue = float(gue_json["r2_E0_on_norm2_daily_mean"])
    ax_c.text(3.0, 38.5, r"$0.5\,{\|\mathbf{x}\|_2}^{\,2}$ cone", fontsize=6.8, color=pal["ink"], ha="left", va="top")
    ax_c.text(8.9, 3.0, f"${{R^2}}_{{\\,\\mathrm{{OLS}}}} = {r2_e0:.3f}$\n(GUE {r2_e0_gue:.3f})", fontsize=6.8, linespacing=1.30, color=pal["slate"], ha="right", va="bottom")
    ax_c.set_xlim(2.5, 9.1)
    ax_c.set_ylim(0.0, 42.5)
    ax_c.set_xticks(np.array([3, 5, 7, 9]))
    ax_c.set_yticks(np.array([0, 20, 40]))
    ax_c.set_xlabel(r"Daily norm $\|\mathbf{x}\|_2$", labelpad=3.4)
    ax_c.set_ylabel(r"Ground energy $E_0$", labelpad=4.6)
    written.append(save_panel_pdf(fig_c, panels_dir, "fig02c_daily_e0_cone"))
    plt.close(fig_c)

    fig_d = plt.figure(figsize=subfig_size_in(0.485, 45.5))
    ax_d = fig_d.add_axes([0.185, 0.240, 0.785, 0.535])
    style_axis(ax_d)
    de = df["mn_qcml_sup_le005_dE"].to_numpy(dtype=np.float64)
    bx_d, bm_d, bs_d = _binned_gue_band(x_norm, de_draws)
    coef_d = np.polyfit(x_norm, de, 1)
    ax_d.fill_between(bx_d, bm_d - bs_d, bm_d + bs_d, color=pal["faint"], alpha=0.95, lw=0, zorder=1)
    ax_d.scatter(x_norm[normal], de[normal], s=6.5, c=pal["quantum_light"], alpha=0.45, lw=0, zorder=2)
    ax_d.scatter(x_norm[stress], de[stress], s=9.0, c=pal["ochre"], alpha=0.85, lw=0, zorder=3)
    ax_d.plot(bx_d, bm_d, color=pal["slate"], ls="--", lw=1.05, zorder=4)
    ax_d.plot(x_grid, np.polyval(coef_d, x_grid), color=pal["ink"], ls="-", lw=1.10, zorder=5)
    r_de = float(np.corrcoef(x_norm, de)[0, 1])
    auc_de = float(sup_meta["dE"]["auroc_contemp"])
    gue_poly = float(gue_json["r2_dE_on_norm_poly_daily_mean"])
    ax_d.text(2.75, 3.58, f"$r_{{\\mathrm{{level}}}} = +{r_de:.2f}$", fontsize=8.65, color=pal["ink"], ha="left", va="top")
    ax_d.text(7.55, 3.55, f"AUROC = {auc_de:.3f}", fontsize=6.8, color=pal["ink"], ha="right", va="top")
    ax_d.text(8.92, 0.56, f"GUE ${{R^2}}_{{\\,\\mathrm{{OLS}}}} = {gue_poly:.3f}$", fontsize=6.8, color=pal["slate"], ha="right", va="bottom")
    h_n = mlines.Line2D([], [], color=pal["quantum_light"], marker="o", ms=3.5, lw=0, label="Normal (684d)")
    h_s = mlines.Line2D([], [], color=pal["ochre"], marker="o", ms=3.5, lw=0, label="Stress (75d)")
    h_ols = mlines.Line2D([], [], color=pal["ink"], ls="-", lw=1.1, label="Linear trend")
    h_g = mlines.Line2D([], [], color=pal["slate"], ls="--", lw=1.0, label="GUE(8) band")
    ax_d.legend(handles=[h_n, h_s, h_ols, h_g], loc="lower center", bbox_to_anchor=(0.48, 1.03), ncol=2, columnspacing=1.05, handletextpad=0.42, labelspacing=0.30)
    ax_d.set_xlim(2.5, 9.1)
    ax_d.set_ylim(0.45, 3.74)
    ax_d.set_xticks(np.array([3, 4, 5, 6, 7, 8, 9]))
    ax_d.set_yticks(np.array([1, 2, 3]))
    ax_d.set_xlabel(r"Daily feature norm $\|\mathbf{x}\|_2$ (759d)", labelpad=3.4)
    ax_d.set_ylabel(r"Raw gap $\Delta E$", labelpad=4.6)
    written.append(save_panel_pdf(fig_d, panels_dir, "fig02d_daily_raw_gap"))
    plt.close(fig_d)

    fig_e = plt.figure(figsize=subfig_size_in(0.485, 45.5))
    ax_e = fig_e.add_axes([0.185, 0.240, 0.785, 0.535])
    style_axis(ax_e)
    det = df["mn_qcml_sup_le005_dE_tilde"].to_numpy(dtype=np.float64)
    bx_e, bm_e, bs_e = _binned_gue_band(x_norm, det_draws)
    coef_e = np.polyfit(x_norm, det, 1)
    ax_e.fill_between(bx_e, bm_e - bs_e, bm_e + bs_e, color=pal["faint"], alpha=0.95, lw=0, zorder=1)
    ax_e.scatter(x_norm[normal], det[normal], s=6.5, c=pal["quantum_light"], alpha=0.45, lw=0, zorder=2)
    ax_e.scatter(x_norm[stress], det[stress], s=9.0, c=pal["ochre"], alpha=0.85, lw=0, zorder=3)
    ax_e.plot(bx_e, bm_e, color=pal["slate"], ls="--", lw=1.05, zorder=4)
    ax_e.plot(x_grid, np.polyval(coef_e, x_grid), color=pal["ink"], ls="-", lw=1.10, zorder=5)
    r_det = float(np.corrcoef(x_norm, det)[0, 1])
    auc_det = float(sup_meta["dE_tilde"]["auroc_contemp"])
    ax_e.text(2.75, 0.625, f"$r_{{\\mathrm{{level}}}} = {r_det:.2f}$", fontsize=8.65, color=pal["ink"], ha="left", va="top")
    ax_e.text(8.92, 0.622, f"AUROC = {auc_det:.3f}", fontsize=6.8, color=pal["ink"], ha="right", va="top")
    ax_e.legend(handles=[h_ols, h_g], loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, columnspacing=1.05, handletextpad=0.42, labelspacing=0.30)
    ax_e.set_xlim(2.5, 9.1)
    ax_e.set_ylim(0.14, 0.65)
    ax_e.set_xticks(np.array([3, 4, 5, 6, 7, 8, 9]))
    ax_e.set_yticks(np.array([0.2, 0.4, 0.6]))
    ax_e.set_xlabel(r"Daily feature norm $\|\mathbf{x}\|_2$ (759d)", labelpad=3.4)
    ax_e.set_ylabel(r"Scale-free $\widetilde{\Delta E}$", labelpad=4.8)
    written.append(save_panel_pdf(fig_e, panels_dir, "fig02e_daily_scalefree_gap"))
    plt.close(fig_e)

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
