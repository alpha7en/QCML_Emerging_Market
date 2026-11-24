"""Shared paths, serif print style, and vector TikZ/Matplotlib panel exporters (Figs. 1-6, S1)."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO_ROOT / "data" / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent / "out"
TIKZ_STYLE = Path(__file__).resolve().with_name("tikz_style.tex")

MM_PER_IN = 25.4
PT_PER_IN = 72.27
MAIN_TEXTWIDTH_MM = 522.0 / PT_PER_IN * MM_PER_IN
SUPP_TEXTWIDTH_MM = 491.4 / PT_PER_IN * MM_PER_IN

PAL: dict[str, str] = {
    "ink": "#1B1F23",
    "slate": "#555E68",
    "muted": "#8E96A0",
    "light": "#C6CCD2",
    "faint": "#E6E9ED",
    "quantum": "#2C5E8A",
    "quantum_light": "#6C9AC3",
    "quantum_pale": "#DCE7F2",
    "ochre": "#B86B19",
    "ochre_light": "#E2B98B",
    "ochre_pale": "#F6E9D8",
}


def runs_dir() -> Path:
    env = os.environ.get("RUNS")
    return Path(env).resolve() if env else DEFAULT_RUNS


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_mpl(*, base_pt: float = 7.6, tick_pt: float = 7.0) -> dict[str, str]:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": base_pt,
            "axes.labelsize": base_pt,
            "axes.titlesize": base_pt,
            "xtick.labelsize": tick_pt,
            "ytick.labelsize": tick_pt,
            "legend.fontsize": tick_pt,
            "axes.linewidth": 0.60,
            "lines.linewidth": 1.20,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.55,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.edgecolor": PAL["ink"],
            "axes.labelcolor": PAL["ink"],
            "text.color": PAL["ink"],
            "xtick.color": PAL["ink"],
            "ytick.color": PAL["ink"],
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
            "figure.constrained_layout.use": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return PAL


def subfig_size_in(frac: float, height_mm: float, *, supp: bool = False) -> tuple[float, float]:
    base_mm = SUPP_TEXTWIDTH_MM if supp else MAIN_TEXTWIDTH_MM
    return (round(base_mm * frac / MM_PER_IN, 4), round(height_mm / MM_PER_IN, 4))


def style_axis(ax: plt.Axes, *, hide_left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(PAL["ink"])
    ax.spines["bottom"].set_linewidth(0.60)
    if hide_left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color(PAL["ink"])
        ax.spines["left"].set_linewidth(0.60)


def save_panel_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf")
    return pdf_path


def resolve_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name
    if brew.is_file():
        return str(brew)
    raise FileNotFoundError(f"Required executable `{name}` not found on PATH.")


def compile_tikz_panel(tex_source: Path, out_dir: Path, stem: str) -> Path:
    tectonic = resolve_bin("tectonic")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = out_dir / f"{stem}.pdf"
    with tempfile.TemporaryDirectory(prefix="tikz_") as td_str:
        td = Path(td_str)
        shutil.copy2(TIKZ_STYLE, td / TIKZ_STYLE.name)
        tmp_tex = td / tex_source.name
        shutil.copy2(tex_source, tmp_tex)
        res = subprocess.run([tectonic, "-X", "compile", str(tmp_tex)], cwd=td, capture_output=True, text=True)
        tmp_pdf = td / tmp_tex.with_suffix(".pdf").name
        if res.returncode != 0 or not tmp_pdf.is_file():
            raise RuntimeError(f"tectonic failed for {tex_source.name}:\n{res.stderr}")
        shutil.copy2(tmp_pdf, pdf_out)
    return pdf_out
