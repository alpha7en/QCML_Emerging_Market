# QCML_Emerging_Market

[![DOI](https://zenodo.org/badge/1471430141.svg)](https://doi.org/10.5281/zenodo.24525011)

Replication package and Python library (`qcml_moex`) for *"Empirical De-Quantization and Norm-Controlled Spectral Audit of Quantum Cognition Machine Learning in an Emerging Equity Market"* (P. Chuklanov, 2026).

The repository implements matrix-form Quantum Cognition Machine Learning (QCML), iso-parameter real-symmetric ($`\mathrm{Sym}(\mathbb{R}^{11})`$) and commuting Gibbs ($`\mathrm{diag}(\mathbb{R}^{64})`$) operator swaps, three-qubit data-reuploading variational quantum circuits (VQCs) with an explicit `CNOT` entanglement toggle, classical asset-pricing controls, and norm-controlled $`\mathrm{GUE}(8)`$ spectral-stress diagnostics on Moscow Exchange (MOEX) equities with causally lagged economic news (`2020–2025`, `37,048` out-of-sample stock-days across `759` sessions).

## Quick start

```bash
pip install -e ".[experiments]"
```

All paper tables and figures read only the shipped aggregate outputs in `data/runs/` and execute in seconds:

```bash
python -I experiments/tables/table01_benchmark.py
python -I experiments/checks/appendix_identities.py
```

Direct library usage on standardized features $`\mathbf{x} \in \mathbb{R}^{18}`$:

```python
import torch
from qcml_moex import QCMLModel

model = QCMLModel(n_features=18, n_outputs=1, hilbert_dim=8, operator_mode="hermitian", seed=101)
x = torch.randn(32, 18, dtype=torch.float64)
diag = model.forward_with_diagnostics(x)
# diag["y_hat"]: bounded projective forecast <psi_0(x)|B|psi_0(x)>
# diag["E_0"], diag["delta_E"]: ground-state energy and first spectral gap
# model.parameter_counts() -> (1216, 1153) raw / identifiable modulo PU(8)
```

## Operator-swap and control hierarchy (Sections 2–3)

| Family | Operator space / ansatz | Gauge orbit | Curvature $`F_{jk}`$ | Non-commutativity $`\kappa_{\mathrm{nc}}`$ | `MKT` / `M+N` params |
|---|---|---|---|---|---|
| `QCML Herm` ($`d=8`$) | $`\mathrm{Herm}(\mathbb{C}^8)`$, ground state $`\rho_0(\mathbf{x})`$ | $`\mathrm{PU}(8)`$ | $`\neq 0`$ | $`\gt 0`$ | `832` / `1,216` (`1,153` eff.) |
| `QCML Real` ($`d=11`$) | $`\mathrm{Sym}(\mathbb{R}^{11})`$, ground state $`\rho_0(\mathbf{x})`$ | $`\mathrm{PO}(11)`$ | $`\equiv 0`$ | $`\gt 0`$ | `858` / `1,254` (`1,199` eff.) |
| `QCML Diag` ($`d=64, \beta^*`$) | $`\mathrm{diag}(\mathbb{R}^{64})`$, Gibbs mixture $`\rho_\beta(\mathbf{x})`$ | $`S_{64}`$ | $`\equiv 0`$ | $`\equiv 0`$ | `832` / `1,216` (`1,216` eff.) |
| `VQC` ($`n=3`$) | Cyclic ($`L=3`$) & affine ($`L=9`$), `CNOT` vs. `NoCNOT` | — | circuit | $`Q_{\mathrm{MW}} \in [0.38, 0.53]`$ vs. $`0`$ | `155–794` / `200–1,118` |
| Classical controls | `Mom21-LowVol`, `Ridge`, `RFF-KRR`, `HistGBDT`, `LightGBM`, `MLP-tanh`, `FiLM-MLP` | — | — | — | `0–1,025` / `0–1,221` |

## Repository map

| Path | Contents |
|---|---|
| `qcml_moex/qcml.py` | `QCMLModel`, Tikhonov-damped Rayleigh–Schrödinger and Daleckii–Krein autograd solvers, $`\mathrm{GUE}(d)`$ null, spectral observables (Sec. 2, Appendix A) |
| `qcml_moex/vqc.py` | Three-qubit statevector VQCs (cyclic and affine encodings, ring-`CNOT` toggle, Meyer–Wallach $`Q_{\mathrm{MW}}`$; Sec. 2.1, Sec. 3.2) |
| `qcml_moex/baselines.py` | Linear factors, kernel/tree/neural baselines, iso-parameter `FiLM-MLP`, and $`S^1 / S^2 / \mathrm{SU}(3)`$ synthetic controls (Sec. 3.2, Supplementary S1) |
| `qcml_moex/panel.py` | MOEX `TQBR` corporate-action adjustments, liquid Top-50 universe, causal $`[t_{\mathrm{prev}}, t-1]`$ news features, purged walk-forward splits (Sec. 3.1–3.2) |
| `qcml_moex/econometrics.py` | Daily Spearman $`\mathrm{RankIC}`$, four-factor $`\mathrm{RankIC}_{\mathrm{FM}}`$, Newey–West HAC, stationary bootstrap, Hansen $`\mathrm{MCS}_{0.05}`$, $`\mathrm{VAR}(5)`$ Granger tests, causal portfolio simulator, Deflated Sharpe Ratio (Sec. 3.3, Sec. 4) |
| `data/universe.json`, `data/news_aggregates.npz` | Active calendar (`1,251` sessions, `253` tickers, split factors) and fold-causal derived inter-session numerical news features (zero raw news text) |
| `data/runs/` | Outputs of the six benchmark suites reported in the paper; hardware details in `data/runs/MACHINE.md` |
| `experiments/fetch_moex.py`, `experiments/run_benchmark.py` | Public MOEX ISS API downloader (`data/raw/moex_equities.parquet`) and full benchmark suite runner writing `data/runs/` |
| `experiments/checks/` | Numerical verification of the Appendix A perturbation, resolvent, gauge-orbit, and $`\mathrm{GUE}(d)`$ identities |
| `experiments/tables/`, `experiments/figures/` | Read-only scripts reproducing Tables 1–4, Tables S1–S2, Figs. 1–6 (`panels/`), and Fig. S1 (`supp_panels/`) from `data/runs/` |

## Reproducing the paper

Scripts under `experiments/tables/` and `experiments/figures/` read only `data/runs/`. Figure scripts emit individual vector subfigure PDFs into `experiments/figures/out/panels/` (`fig01a`–`fig06e`) and `experiments/figures/out/supp_panels/` (`figS1a`–`figS1d`). Standalone TikZ panels in Fig. 1 (`a–c`) and Fig. 3 (`d–e`) compile via [`tectonic`](https://tectonic-typesetting.github.io/). Runtimes in the third column refer to full retraining from scratch on the 4-core CPU container documented in `data/runs/MACHINE.md`.

| Paper item | Suite in `data/runs/` (`run_benchmark.py --suite ...`) | Full suite time | Reproduction script (`python -I ...`) |
|---|---|---|---|
| Appendix A Eqs. (A.1)–(A.9) | — | — | `experiments/checks/appendix_identities.py` (`<5 s`) |
| Table 1 (Sec. 4.1) | `classical_controls`, `vqc_controls`, `qcml_benchmark` | `2.6 h` | `experiments/tables/table01_benchmark.py` |
| Table 2 (Sec. 4.1) | `classical_controls`, `vqc_controls`, `qcml_benchmark` | (above) | `experiments/tables/table02_contrasts.py` |
| Table 3 (Sec. 4.2) | `spectral_stress` | `7.1 min` | `experiments/tables/table03_spectral.py` |
| Table 4 (Sec. 4.3) | `classical_controls`, `vqc_controls`, `qcml_benchmark`, `spectral_stress` | (above) | `experiments/tables/table04_economics.py` |
| Table S1 (Supp. S1–S2) | `synthetic_controls`, `multioutput_robustness` | `19.0 min` | `experiments/tables/tableS1_synthetic_t4.py` |
| Table S2 (Supp. S3) | `multioutput_robustness`, `classical_controls`, `qcml_benchmark` | (above) | `experiments/tables/tableS2_robustness.py` |
| Fig. 1 (Sec. 2.1) | — | — | `experiments/figures/fig01_circuits.py` |
| Fig. 2 (Sec. 2.2) | `spectral_stress` | (above) | `experiments/figures/fig02_geometry.py` |
| Fig. 3 (Sec. 3.1–3.2) | `synthetic_controls` | `55 s` | `experiments/figures/fig03_regimes.py` |
| Fig. 4 (Sec. 4.1) | `classical_controls`, `vqc_controls`, `qcml_benchmark` | (above) | `experiments/figures/fig04_dequant.py` |
| Fig. 5 (Sec. 4.2–4.3) | `spectral_stress` | (above) | `experiments/figures/fig05_spectral.py` |
| Fig. 6 (Sec. 4.3–4.4) | `classical_controls`, `vqc_controls`, `qcml_benchmark`, `spectral_stress` | (above) | `experiments/figures/fig06_frictions.py` |
| Fig. S1 (Supp. S1–S3) | `synthetic_controls`, `multioutput_robustness`, `classical_controls`, `qcml_benchmark` | (above) | `experiments/figures/figS1_robustness.py` |

To re-run the entire benchmark from raw MOEX quotes:

```bash
python -I experiments/fetch_moex.py
python -I experiments/run_benchmark.py --suite all --jobs 4
```

## Citing

Please cite the accompanying paper:

> P. Chuklanov, "Empirical De-Quantization and Norm-Controlled Spectral Audit of Quantum Cognition Machine Learning in an Emerging Equity Market," 2026.

For the software and derived data package itself, cite `QCML_Emerging_Market` 1.0.0 (2026), [doi:10.5281/zenodo.24525012](https://doi.org/10.5281/zenodo.24525012) (version 1.0.0) or [doi:10.5281/zenodo.24525011](https://doi.org/10.5281/zenodo.24525011) (all versions). Machine-readable citation metadata is provided in `CITATION.cff` and `.zenodo.json`.

## License

Source code is licensed under `BSD-3-Clause` (`LICENSE`). Derived numerical aggregates and benchmark outputs under `data/` are licensed under `CC BY-NC 4.0` (`data/LICENSE`). Upstream market-data and corpus terms are documented in `THIRD_PARTY.md`.
