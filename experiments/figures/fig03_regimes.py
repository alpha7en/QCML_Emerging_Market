"""Figure 3: MOEX 2020-2025 index regimes (a), 5-day realized volatility and CBR key rate (b),
six purged walk-forward folds F1-F6 (c), causal forecast clock (d), and staggered five-cohort
weekly execution schedule (e; Sec. 3.1-3.3). Requires `tectonic` for panels (d)-(e).

Usage: python -I experiments/figures/fig03_regimes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, compile_tikz_panel, load_json, runs_dir, save_panel_pdf, setup_mpl, style_axis, subfig_size_in

HERE = Path(__file__).resolve().parent


def _break_on_halt(dates: pd.Series, values: np.ndarray, gap_days: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN across multi-week exchange closures so trajectories do not bridge non-trading halts."""
    dt_vals = dates.to_numpy()
    diffs = np.diff(dt_vals).astype("timedelta64[D]").astype(np.int64)
    break_idxs = np.where(diffs > gap_days)[0]
    if len(break_idxs) == 0:
        return dt_vals, values.astype(np.float64)
    out_dates, out_vals, prev = [], [], 0
    for idx in break_idxs:
        out_dates.extend(dt_vals[prev : idx + 1])
        out_vals.extend(values[prev : idx + 1])
        out_dates.append(dt_vals[idx] + np.timedelta64(1, "D"))
        out_vals.append(np.nan)
        prev = idx + 1
    out_dates.extend(dt_vals[prev:])
    out_vals.extend(values[prev:])
    return np.array(out_dates, dtype="datetime64[ns]"), np.array(out_vals, dtype=np.float64)


def _draw_qband(ax: plt.Axes, q_start: pd.Timestamp, q_end: pd.Timestamp, pal: dict[str, str]) -> None:
    ax.axvspan(q_start, q_end, color=pal["faint"], alpha=0.95, lw=0, zorder=1.0)
    for edge in (q_start, q_end):
        ax.axvline(edge, color=pal["light"], lw=0.55, ls="--", zorder=1.1)


def build_figure(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    syn_dir = runs_dir() / "synthetic_controls"
    df = pd.read_csv(syn_dir / "daily_regime_series.csv")
    df["dt"] = pd.to_datetime(df["tradedate"])
    h5_folds = load_json(syn_dir / "fold_schedule.json")["h5_folds"]

    q_rows = df.loc[df["is_quarantine_h5"] == 1, "dt"]
    q_start, q_end = q_rows.iloc[0], q_rows.iloc[-1]
    sb_rows = df.loc[df["short_ban"] == 1, "dt"]
    sb_start, sb_end = sb_rows.iloc[0], sb_rows.iloc[-1]

    vol_raw = (df["imoex_roll_vol_5d"] * np.sqrt(252.0) * 100.0).to_numpy(dtype=np.float64)
    cbr_rate = df["cbr_key_rate_pct"].to_numpy(dtype=np.float64)
    dt_imoex, imoex_broken = _break_on_halt(df["dt"], df["imoex_close"].to_numpy(dtype=np.float64))
    dt_vol, vol_raw_broken = _break_on_halt(df["dt"], vol_raw)

    pal = setup_mpl(base_pt=7.6, tick_pt=7.0)
    panels_dir = out_dir / "panels"
    x_min, x_max = pd.Timestamp("2020-04-01"), pd.Timestamp("2025-05-05")
    written: list[Path] = []

    fig_a = plt.figure(figsize=subfig_size_in(0.485, 39.0))
    ax_a = fig_a.add_axes([0.215, 0.230, 0.755, 0.630])
    style_axis(ax_a)
    _draw_qband(ax_a, q_start, q_end, pal)
    ax_a.plot(dt_imoex, imoex_broken, color=pal["ink"], lw=1.05, zorder=3.0)
    idx_max, idx_min = int(df["imoex_close"].idxmax()), int(df["imoex_close"].idxmin())
    dt_pk, v_pk = df.loc[idx_max, "dt"], float(df.loc[idx_max, "imoex_close"])
    dt_tr, v_tr = df.loc[idx_min, "dt"], float(df.loc[idx_min, "imoex_close"])
    ax_a.scatter([dt_pk, dt_tr], [v_pk, v_tr], s=9, color=pal["ink"], zorder=4.0)
    ax_a.text(dt_pk - pd.Timedelta(days=30), v_pk + 90, f"{v_pk:,.0f}", fontsize=6.8, color=pal["slate"], ha="right", va="bottom")
    ax_a.text(dt_tr + pd.Timedelta(days=45), v_tr - 30, f"{v_tr:,.0f}", fontsize=6.8, color=pal["slate"], ha="left", va="center")
    ax_a.plot([sb_start, sb_start, sb_end, sb_end], [4430, 4580, 4580, 4430], color=pal["ochre"], lw=1.2, zorder=4.0)
    ax_a.text(sb_end + pd.Timedelta(days=30), 4550, "Short ban (47d)", fontsize=6.8, color=pal["ochre"], ha="left", va="center")
    ax_a.annotate(
        r"$\mathcal{Q}(5)$ halt (13d)",
        xy=(q_end, 3880),
        xytext=(q_end + pd.Timedelta(days=80), 3880),
        fontsize=6.8,
        color=pal["slate"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=pal["slate"], lw=0.55, shrinkA=1.5, shrinkB=2.0),
    )
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylim(1550, 4880)
    ax_a.set_yticks(np.array([2000, 3000, 4000]))
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_a.set_ylabel("IMOEX (pts)", labelpad=4.4)
    written.append(save_panel_pdf(fig_a, panels_dir, "fig03a_imoex_regimes"))
    plt.close(fig_a)

    fig_b = plt.figure(figsize=subfig_size_in(0.485, 39.0))
    ax_b = fig_b.add_axes([0.205, 0.230, 0.675, 0.630])
    style_axis(ax_b)
    _draw_qband(ax_b, q_start, q_end, pal)
    ax_b.plot(dt_vol, vol_raw_broken, color=pal["quantum"], lw=0.85, zorder=3.0, label="5d realized vol.")
    ax_b.step(df["dt"].to_numpy(), cbr_rate, where="post", color=pal["ochre"], lw=1.25, zorder=4.0, label="CBR key rate")
    ax_b.set_yscale("log")
    ax_b.set_xlim(x_min, x_max)
    ax_b.set_ylim(1.20, 520.0)
    ax_b.set_yticks([3, 10, 30, 100, 300])
    ax_b.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_b.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_b.set_ylabel("Rate / vol. (% p.a.)", labelpad=4.4)
    idx_vmax = int(np.nanargmax(vol_raw))
    ax_b.text(df.loc[idx_vmax, "dt"] + pd.Timedelta(days=40), float(vol_raw[idx_vmax]) * 0.88, f"{vol_raw[idx_vmax]:.0f}%", fontsize=6.8, color=pal["quantum"], ha="left", va="center")
    ax_b.text(pd.Timestamp("2020-12-20"), 2.80, "4.25%", fontsize=6.8, color=pal["ochre"], ha="center", va="top")
    ax_b.text(pd.Timestamp("2025-05-22"), 21.0, "21.00%", fontsize=6.8, color=pal["ochre"], ha="left", va="center", clip_on=False)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.48, 1.04), ncol=2, columnspacing=1.05, handlelength=1.2, handletextpad=0.42)
    written.append(save_panel_pdf(fig_b, panels_dir, "fig03b_vol_and_cbr"))
    plt.close(fig_b)

    fig_c = plt.figure(figsize=subfig_size_in(0.99, 38.0))
    ax_c = fig_c.add_axes([0.095, 0.255, 0.885, 0.525])
    style_axis(ax_c)
    _draw_qband(ax_c, q_start, q_end, pal)
    bar_h, n_folds = 0.60, len(h5_folds)
    q_start_num = mdates.date2num(q_start)
    q_pre_end_num = mdates.date2num(q_start - pd.Timedelta(days=1))
    q_post_start_num = mdates.date2num(q_end + pd.Timedelta(days=1))

    for fold_info in h5_folds:
        k = int(fold_info["fold"])
        y = float(n_folds - 1 - k)
        tr0, tr1 = mdates.date2num(pd.Timestamp(fold_info["train_range"][0])), mdates.date2num(pd.Timestamp(fold_info["train_range"][1]))
        va0, va1 = mdates.date2num(pd.Timestamp(fold_info["val_range"][0])), mdates.date2num(pd.Timestamp(fold_info["val_range"][1]))
        te0, te1 = mdates.date2num(pd.Timestamp(fold_info["test_range"][0])), mdates.date2num(pd.Timestamp(fold_info["test_range"][1]))
        train_segs = [(tr0, tr1 - tr0)] if tr1 < q_start_num else [(tr0, q_pre_end_num - tr0), (q_post_start_num, tr1 - q_post_start_num)]
        ax_c.broken_barh(train_segs, (y - bar_h / 2.0, bar_h), facecolors=pal["slate"], edgecolors="none", zorder=2.5)
        ax_c.add_patch(mpatches.Rectangle((tr1, y - bar_h / 2.0), va0 - tr1, bar_h, facecolor="white", edgecolor=pal["ochre"], lw=0.75, zorder=3.2))
        ax_c.broken_barh([(va0, va1 - va0)], (y - bar_h / 2.0, bar_h), facecolors=pal["quantum_light"], edgecolors="none", zorder=2.5)
        if k == 0:
            ax_c.add_patch(mpatches.Rectangle((va1, y - bar_h / 2.0), te0 - va1, bar_h, facecolor="none", edgecolor=pal["ochre"], lw=0.9, zorder=3.6))
            ax_c.plot([va1, te0], [y - bar_h / 2.0, y + bar_h / 2.0], color=pal["ochre"], lw=0.75, zorder=3.6)
        else:
            ax_c.broken_barh([(va1, te0 - va1)], (y - bar_h / 2.0, bar_h), facecolors=pal["ochre"], edgecolors="none", zorder=2.8)
        ax_c.broken_barh([(te0, te1 - te0)], (y - bar_h / 2.0, bar_h), facecolors=pal["quantum"], edgecolors="none", zorder=2.8)

    ax_c.text(pd.Timestamp("2020-07-15"), 2.5, "Warm-up\n(127d)", fontsize=6.8, color=pal["slate"], ha="center", va="center")
    ax_c.set_xlim(x_min, x_max)
    ax_c.set_ylim(-0.6, n_folds - 0.35)
    ax_c.set_yticks(np.arange(n_folds))
    ax_c.set_yticklabels([f"F{n_folds - i}" for i in range(n_folds)])
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_c.set_ylabel("Fold", labelpad=4.4)
    ax_c.set_xlabel("Trading date", labelpad=3.4)
    leg_items = [
        mpatches.Patch(facecolor=pal["slate"], label="Train"),
        mpatches.Patch(facecolor="white", edgecolor=pal["ochre"], lw=0.8, label="Purge (5d)"),
        mpatches.Patch(facecolor=pal["quantum_light"], label="Val (63d)"),
        mpatches.Patch(facecolor=pal["ochre"], label="Purge+embargo (10\u201313d)"),
        mpatches.Patch(facecolor=pal["quantum"], label="OOS test"),
        mpatches.Patch(facecolor=pal["faint"], edgecolor=pal["light"], lw=0.6, label=r"$\mathcal{Q}(5)$ excised"),
    ]
    ax_c.legend(handles=leg_items, loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=6, columnspacing=0.95, handlelength=1.1, handletextpad=0.42)
    written.append(save_panel_pdf(fig_c, panels_dir, "fig03c_walkforward_folds"))
    plt.close(fig_c)

    written.append(compile_tikz_panel(HERE / "fig03d_causal_clock.tex", panels_dir, "fig03d_causal_clock"))
    written.append(compile_tikz_panel(HERE / "fig03e_weekly_cohorts.tex", panels_dir, "fig03e_weekly_cohorts"))

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
