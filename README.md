# Dissertation — Covariate Shift and Importance-Weighted Learning

This repository contains the experimental framework developed for the dissertation investigating covariate shift between multivariate Gaussian distributions and its effects on chi-squared divergence, importance weights, effective sample size (ESS), and regression model generalisation performance.

The experiments are divided into controlled synthetic experiments and benchmarking experiments using external datasets. The repository also contains utilities for constructing shifted distributions, estimating divergence, performing importance weighting, running regression models, and generating the figures and summary tables used in the dissertation.

---

## Project Structure

```text
Dissertation/
│
├── analysis/
│   ├── __init__.py
│   └── aggregation.py
│
├── experiments/
│   ├── __init__.py
│   ├── grid_runner.py
│   └── run_experiment.py
│
├── plotting/
│   ├── __init__.py
│   ├── figures_theoretical_chi_sq.py
│   └── helpers.py
│
├── results/
│    ├── plots/
│    │  ├── extended/
|    |  |     ├── mc_div_vs_true_div/
|    |  |             └── summary.csv
|    |  |      └── true_chi_sq/
|    |  |             ├── chi_sq_vs_alpha/
|    |  |             ├── chi_sq_vs_lambda/
|    |  |              ├── ess_vs_chi_sq_divergence/
|    |  |              ├── gen_gap_vs_chi_sq_div/
|    |  |              └── weight_variance_vs_chi_sq_divergence/
│    │  ├── external/
|    |  |      ├── mc_div_vs_true_div/
|    |  |              └── summary.csv
|    |  |      └── true_chi_sq/
|    |  |              ├── chi_sq_vs_alpha/
|    |  |              ├── chi_sq_vs_lambda/
|    |  |              ├── ess_vs_chi_sq_divergence/
|    |  |              ├── gen_gap_vs_chi_sq_div/
|    |  |              └── weight_variance_vs_chi_sq_divergence/
|    |  └── small/
|    |       ├── mc_div_vs_true_div/
|    |               └── summary.csv
|    |       └── true_chi_sq/
|    |               ├── chi_sq_vs_alpha/
|    |               ├── chi_sq_vs_lambda/
|    |               ├── ess_vs_chi_sq_divergence/
|    |               ├── gen_gap_vs_chi_sq_div/
|    |               └── weight_variance_vs_chi_sq_divergence/
│    └── tables/
|           ├── extended/
|           |       └── results.csv
|           ├── external/
|           |       └── results.csv
|           └── small/
|                   └── results.csv
│
├── scripts/
│   ├── plot_extended_results_external.py
│   ├── plot_extended_results_synthetic.py
│   ├── plot_small_results.py
│   ├── run_extended_grid_external.py
│   ├── run_extended_grid_synthetic.py
│   └── run_small_grid_synthetic.py
│
├── utils/
│   ├── __init__.py
│   ├── dataset_helper.py
│   ├── density.py
│   ├── distributions.py
│   ├── divergence.py
│   ├── importance_sampling.py
│   ├── metrics.py
│   ├── models.py
│   └── targets.py
│
├── .gitattributes
├── .gitignore
└── README.md

### Main components
`utils/`

Contains the core functions used throughout the experiments.

distributions.py — construction and sampling of the source, shifted, and contaminated distributions.
density.py — probability and log-density calculations.
divergence.py — theoretical and Monte Carlo chi-squared divergence calculations.
importance_sampling.py — density ratios, importance weights, and effective sample size calculations.
metrics.py — evaluation metrics used by the experiments.
models.py — regression models and their weighted/unweighted variants.
targets.py — generation of target functions used in the synthetic experiments.
dataset_helper.py — loading and preprocessing of external datasets.
experiments/

Contains the functions responsible for executing individual experiments and constructing experiment grids.

run_experiment.py — execution of individual experimental configurations.
grid_runner.py — construction and parallel execution of experiment grids.
analysis/

Contains functions used for aggregating and processing experimental results.

plotting/

Contains reusable plotting functions and plotting utilities used to generate the figures from the experimental results.

scripts/

Contains the executable scripts used to run the experiments and generate the final plots.

results/

Stores generated experimental outputs.

results/tables/ contains CSV files containing experiment results.
results/plots/ contains generated figures, organised according to experiment type and analysis.