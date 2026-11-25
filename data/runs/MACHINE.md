# Machine of the shipped runs

All six benchmark suites in `data/runs/` were executed in a 4-core `x86_64` Linux container (Intel Xeon, 16 GB RAM) running Python 3.11, PyTorch, NumPy, SciPy, scikit-learn, and LightGBM in CPU mode (`float64`/`complex128` for all matrix-QCML eigensolvers and econometric estimators; `float32`/`complex64` for statevector VQCs and tabular neural baselines). Total wall-clock time across all six suites is `10,998.8 s` (`3.06 h`).

| Suite (`data/runs/`) | Paper scope | Wall time | Primary files |
|---|---|---|---|
| `synthetic_controls/` | Supplementary S1 ($`S^1`$, $`S^2`$, $`d = 3`$ spin-1 quadrupole); Sec. 3.1–3.2 calendar & fold schedule | `55.1 s` | `synthetic_pos_control.json`, `fold_schedule.json`, `daily_regime_series.csv` |
| `classical_controls/` | Sec. 4.1, 4.3 Tier-0 factors & Tier-1 classical controls + news-shuffle placebos | `2,053.4 s` | `summary.json`, `daily_rank_ic.csv`, `daily_mse.csv`, `cost_sweep.json`, `equity_curves.csv`, `mcs_and_paired.json` |
| `vqc_controls/` | Sec. 4.1, 4.3 3-qubit cyclic & affine VQCs (`CNOT` vs. `NoCNOT`, full-Hermitian vs. local-$`Z/X`$) | `2,515.1 s` | `summary.json`, `vqc_entanglement_tests.json`, `circuit_specs.json`, `daily_rank_ic.csv`, `cost_sweep.json` |
| `qcml_benchmark/` | Sec. 4.1, 4.3 Hermitian ($`d \in \{2,4,8,16\}, d^*_f`$), real-symmetric ($`d = 11`$), commuting Gibbs ($`d = 64, \beta^*_f`$), $`\lambda_E = 0`$, news placebos, unified $`\mathrm{MCS}_{0.05}`$ | `4,862.3 s` | `summary.json`, `opswap_grid.json`, `paired_tests_all.json`, `daily_rank_ic.csv`, `cost_sweep.json`, `equity_curves.csv` |
| `spectral_stress/` | Sec. 4.2, 4.3 759-day norm-controlled spectral audit, GUE($`d = 8`$) null, $`\mathrm{VAR}(5)`$ Granger tests & Feb–Mar 2022 crisis OOD bounds | `425.4 s` | `summary.json`, `roc_and_spectral.json`, `granger_var5.json`, `ood_2022_bounds.json`, `daily_regime_series.csv`, `gue_daily_null.npz` |
| `multioutput_robustness/` | Supplementary S2–S3 four-target joint state ($`q = 4`$), liquidity cohorts, ex-dividend mask, survivorship ($`k \in \{3,5,7\}`$), purged $`h = 15`$, rolling $`W_{\mathrm{train}} = 126\mathrm{d}`$ | `1,087.4 s` | `summary.json`, `t4_multioutput.json`, `liquidity_and_cohorts.json`, `survivorship_bounds.json`, `robustness_grid.json` |
