# Quantum Forecasting of Many-Body Correlation Dynamics

Code accompanying the manuscript:

**Forecasting High-Dimensional Quantum Correlation Dynamics with Space-Time Projection**

This repository implements and benchmarks Space-Time Projection (STP) for forecasting the connected two-point Pauli correlation tensor of the random-field Heisenberg chain.

The workflow includes:

* exact finite-size dynamics for open Heisenberg chains with \(L=12,14\),
* disorder strengths \(W=2,3,4,5\),
* independent training, validation, and test disorder realizations,
* construction of the full connected tensor \(C_{ij}^{\alpha\beta}(t)\),
* canonical STP forecasting,
* ridge-regression, persistence, training-mean, and DMD benchmarks,
* within-disorder and leave-one-disorder-out transfer tests,
* paired bootstrap uncertainty estimates,
* analysis of the spatial spread of connected \(zz\) correlations,
* scripts for reproducing the manuscript figures.

Main files:

* `heisenberg_ed.py` — exact Heisenberg-chain dynamics
* `generate_ed_data.py` — reproducible dataset generation and seed manifest
* `canonical_stp.py` — canonical STP implementation
* `baselines.py` — forecasting baselines
* `run_correlation_tensor_forecast.py` — main tensor benchmark
* `run_dmd_tensor_benchmark.py` — DMD benchmark
* `plot_paired_rzz_errors.py` — correlation-spread-radius analysis
* `paper_figures.py` — manuscript figures
* `test_physics.py`, `test_splits.py`, `test_stp.py` — validation tests

The numerical experiments use validation data for hyperparameter selection and reserve independent test trajectories for final evaluation.

Space-Time Projection was introduced by Oliver T. Schmidt; this repository applies the method to quantum many-body correlation dynamics.

Licensed under the MIT License.
