# External data sources and software licences

Source code in this repository is distributed under the BSD 3-Clause licence (`LICENSE`). Derived numerical aggregates and run outputs under `data/` are distributed under Creative Commons Attribution-NonCommercial 4.0 International (`data/LICENSE`, CC BY-NC 4.0). No third-party patent rights are granted or implied.

| Source or library | Scope | In this repository | Licence / terms |
|---|---|---|---|
| Moscow Exchange ISS (`https://iss.moex.com`) | Daily `TQBR` equity trades (`CLOSE`, `VOLUME`, `VALUE`) and `IMOEX` index closes, 2020-04-17 to 2025-04-17 | Not included; retrieved directly from the public ISS endpoints by `experiments/fetch_moex.py` into gitignored `data/raw/` | Moscow Exchange User Agreement and Market Data Policy (personal and academic research retrieval; raw exchange quote tables are not redistributed) |
| Central Bank of Russia (`https://www.cbr.ru`) | Official 27-step key-rate schedule (2020-02-10 to 2024-10-28) and 2022 short-selling restriction dates (2022-02-24 to 2022-05-31) | Public policy dates and rates recorded in `qcml_moex/panel.py` | Public official statistical and regulatory announcements of the Bank of Russia |
| Lenta.ru economic news (`https://lenta.ru`) | Inter-session economic headlines timestamped in $`[t_{\mathrm{prev}}, t - 1]`$ across 2020-04-18 to 2025-04-17 | Raw copyrighted headline texts are **not** included; only derived non-copyright daily numerical aggregates (6 scalar statistics per session per fold) are stored in `data/news_aggregates.npz` | Derived numerical aggregates in `data/` under CC BY-NC 4.0 (`data/LICENSE`); raw press texts excluded |
| NumPy, SciPy, pandas, PyArrow, scikit-learn, PyTorch | Core numerical, tabular I/O, econometric, and automatic-differentiation libraries | Installed via `pip` (`pyproject.toml`) | BSD 3-Clause, Apache 2.0 (`pyarrow`) |
| LightGBM, Matplotlib | Optional gradient-boosting baseline and figure generation (`pip install -e .[experiments]`) | Installed via `pip` | MIT, PSF/Matplotlib |

All Quantum Cognition Machine Learning (`qcml_moex/qcml.py`) and variational quantum circuit (`qcml_moex/vqc.py`) solvers in this repository were written from scratch directly from the mathematical formulations in the cited literature (Ishiki 2015; Steinacker 2021; Candelori et al. 2025; Abanov et al. 2025; Hammond 2026).
