"""Figure 1: Matrix-form QCML error Hamiltonian and Rayleigh-Ritz readout (a), equal-capacity
operator-swap hierarchy with exact 1D readout slices (b), and three-qubit data-reuploading
variational circuit with toggleable ring-CNOT entanglement (c; Sec. 2.1). Requires `tectonic`.

Usage: python -I experiments/figures/fig01_circuits.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, compile_tikz_panel

HERE = Path(__file__).resolve().parent
PANELS = (
    "fig01a_qcml_spectrum",
    "fig01b_operator_swap",
    "fig01c_vqc_circuit",
)


def build_figure(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    panels_dir = out_dir / "panels"
    written: list[Path] = []
    for stem in PANELS:
        pdf_path = compile_tikz_panel(HERE / f"{stem}.tex", panels_dir, stem)
        print(f"Wrote {pdf_path}")
        written.append(pdf_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build_figure(args.out_dir)


if __name__ == "__main__":
    main()
