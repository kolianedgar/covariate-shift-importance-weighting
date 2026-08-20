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
|    |  |             ├── ess_vs_chi_sq_divergence/
|    |  |             ├── gen_gap_vs_chi_sq_div/
|    |  |             └── weight_variance_vs_chi_sq_divergence/
│    │  ├── external/
|    |  |      ├── mc_div_vs_true_div/
|    |  |      |       └── summary.csv
|    |  |      └── true_chi_sq/
|    |  |              ├── chi_sq_vs_alpha/
|    |  |              ├── chi_sq_vs_lambda/
|    |  |              ├── ess_vs_chi_sq_divergence/
|    |  |              ├── gen_gap_vs_chi_sq_div/
|    |  |              └── weight_variance_vs_chi_sq_divergence/
|    |  └── small/
|    |       ├── mc_div_vs_true_div/
|    |       |       └── summary.csv
|    |       └── true_chi_sq/
|    |               ├── chi_sq_vs_alpha/
|    |               ├── chi_sq_vs_lambda/
|    |               ├── ess_vs_chi_sq_divergence/
|    |               ├── gen_gap_vs_chi_sq_div/
|    |               └── weight_variance_vs_chi_sq_divergence/
│    └── tables/
|           ├── extended/
|           |        └── results.csv
|           ├── external/
|           |        └── results.csv
|           └── small/
|                    └── results.csv
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
```

## Main Components

The project is organised into several components, each responsible for a specific part of the experimental pipeline:

- **`utils/`** — Core utilities used throughout the experiments:
  - `distributions.py` — Construction and sampling of source, shifted, and contaminated distributions.
  - `density.py` — Density and log-density calculations.
  - `divergence.py` — Theoretical and Monte Carlo estimation of chi-squared divergence.
  - `importance_sampling.py` — Importance-weight and effective sample size calculations.
  - `metrics.py` — Evaluation metrics used to assess model performance.
  - `models.py` — Regression models used in the experiments.
  - `targets.py` — Generation of synthetic target functions.
  - `dataset_helper.py` — Loading and preprocessing of external datasets.

- **`experiments/`** — Experimental execution logic. The grid runners construct and execute combinations of experimental parameters, while `run_experiment.py` contains the individual experiment workflow.

- **`analysis/`** — Functions for aggregating and processing experimental results before further analysis.

- **`plotting/`** — Plotting and result-export functions used to generate figures and summary tables.

- **`scripts/`** — Main entry points for running experiments and generating plots.

- **`results/`** — Stores generated experimental outputs:
  - `tables/` contains experiment results in CSV format.
  - `plots/` contains generated figures, organised by experiment type and analysis.

- **`README.md`** — Provides instructions for setting up the project and reproducing the experiments and figures.

## Running the Experiments

The experiments are executed through the scripts in the `scripts/` directory.

All commands below should be run from the root directory of the repository:

```bash
cd Dissertation
```

The Python environment used to run the project should contain the packages required by the source code, including PyTorch, NumPy, pandas, scikit-learn, matplotlib, and joblib.

### 1. Small-Scale Synthetic Experiments

The small synthetic experiment grid is executed using:

```bash
python scripts/run_small_grid_synthetic.py
```

This pipeline is intended for smaller-scale experiments and testing of the experimental framework.

The results are written to the corresponding location under:

`results/tables/`

The small-scale plotting script can then be used to generate the corresponding figures:

```bash
python scripts/plot_small_results.py
```

### 2. Extended Synthetic Experiments

The main large-scale synthetic experiment grid is executed using:

```bash
python scripts/run_extended_grid_synthetic.py
```

The extended synthetic experiments systematically vary the experimental parameters used to investigate the effects of distributional shift.

These include:

- dimensionality;
- mean shift (`lambda`);
- covariance inflation (`alpha`);
- contamination level (`epsilon`);
- model type;
- target-function type;
- shift type; and
- random seed.

The resulting experiments are stored in:

`results/tables/extended/results.csv`

The extended synthetic plotting pipeline is executed using:

```bash
python scripts/plot_extended_results_synthetic.py
```

The plotting script reads:

`results/tables/extended/results.csv`

and generates figures examining:

- chi-squared divergence against mean shift;
- chi-squared divergence against covariance shift;
- importance-weight variance against chi-squared divergence;
- effective sample size against chi-squared divergence; and
- generalisation gap against chi-squared divergence.

The resulting figures are organised under:

`results/plots/extended/true_chi_sq/`

with separate directories for the different analyses.

The Monte Carlo versus theoretical divergence summary is additionally exported to:

`results/plots/extended/mc_div_vs_true_div/summary.csv`

### 3. External Benchmarking Experiments

The external benchmarking experiments are executed using:

```bash
python scripts/run_extended_grid_external.py
```

These experiments use external datasets to construct approximations of multivariate non-standard Gaussian source distributions.

The benchmarking pipeline uses a smaller experimental grid than the synthetic pipeline due to computational constraints.

The results are stored in:

`results/tables/external/results.csv`

The external experiments use:

- OLS;
- RBF-kernel SVR;
- importance-weighted OLS; and
- importance-weighted RBF-kernel SVR.

The external datasets are processed by estimating their mean vectors and covariance matrices and using these estimates to construct the Gaussian distributions used in the theoretical divergence calculations.

### 4. Plotting External Results

After completing the external experiments, the corresponding figures can be generated using:

```bash
python scripts/plot_extended_results_external.py
```

The script reads:

`results/tables/external/results.csv`

and generates the same main categories of analyses as the synthetic pipeline:

- generalisation gap versus chi-squared divergence;
- importance-weight variance versus chi-squared divergence;
- effective sample size versus chi-squared divergence;
- chi-squared divergence versus covariance shift;
- chi-squared divergence versus mean shift; and
- Monte Carlo versus theoretical chi-squared divergence.

The resulting figures are stored under:

`results/plots/external/true_chi_sq/`

The Monte Carlo versus theoretical divergence summary is stored at:

`results/plots/external/mc_div_vs_true_div/summary.csv`

## Experimental Framework

The experiments investigate covariate shift by constructing a source distribution `P0(X)` and a shifted distribution `P1(X)`.

Three types of shift are considered:

1. Mean shift
2. Covariance shift
3. Combined mean and covariance shift

The testing distribution is constructed using an epsilon-contamination model:

`Pε(X) = (1 - ε)P0(X) + εP1(X)`

The experiments then evaluate how increasing distributional mismatch affects:

- theoretical chi-squared divergence;
- Monte Carlo estimates of chi-squared divergence;
- importance-weight variance;
- effective sample size;
- training and testing performance; and
- the generalisation gap.

## Synthetic Experiments

The synthetic pipeline uses a controlled multivariate standard Gaussian source distribution.

The main experimental parameters are:

- d       Dimensionality
- λ       Mean-shift magnitude
- α       Covariance inflation
- ε       Contamination level

The synthetic experiments also vary:

- Model type
- Target type
- Shift type
- Random seed

## External Benchmarking

The external benchmarking pipeline uses datasets to estimate a multivariate Gaussian distribution from the training data.

The estimated mean vector and covariance matrix are then used to construct the source distribution for the theoretical calculations.

The benchmarking experiments provide a less controlled setting than the synthetic experiments because the estimated distributions contain the geometric structure of the underlying datasets rather than the fixed zero-mean, identity-covariance structure used in the synthetic pipeline.

Because of computational constraints, the external experiments use a smaller parameter grid and a reduced set of regression models.

This controlled setup allows the effect of individual distributional changes to be investigated independently.

The synthetic experiments use both weighted and unweighted regression models, allowing the effect of importance weighting under increasing distributional mismatch to be examined.

Results

Generated results are separated into tables and figures.

**Tables**

```text
results/tables/
├── extended/
│   └── results.csv
└── external/
    └── results.csv
```

The CSV files contain the results of the individual experimental configurations and can be loaded using pandas:

```python
import pandas as pd

results = pd.read_csv(
    "results/tables/extended/results.csv"
)
```

or:

```python
results = pd.read_csv(
    "results/tables/external/results.csv"
)
```

**Figures**

Figures are organised according to experiment type and analysis:

```text
results/plots/
├── extended/
├── external/
└── small/
```

Within each experiment type, plots are grouped according to the quantity being analysed, for example:

```text
true_chi_sq/
├── chi_sq_vs_lambda/
├── chi_sq_vs_alpha/
├── ess_vs_chi_sq_divergence/
├── gen_gap_vs_chi_sq_div/
└── weight_variance_vs_chi_sq_divergence/
```

## Reproducibility

The experiment grids include multiple random seeds to account for variability caused by random sampling.

The experiment runners execute configurations independently and use parallel processing where applicable.

For a complete reproduction of the main results, run the relevant experiment script first and then run its corresponding plotting script.

Because the extended grids can contain a large number of individual configurations, execution time depends on the available computational resources and the level of parallelism used by the experiment runner.

## Notes

The theoretical chi-squared divergence is calculated directly for the Gaussian distributions used in the experiments, while the Monte Carlo estimate is obtained from samples drawn from the source distribution.

The benchmarking experiments approximate the empirical dataset distributions using multivariate Gaussian distributions. Therefore, the benchmarking results should be interpreted as an evaluation of the proposed framework under this Gaussian approximation rather than as an exact representation of the original dataset distributions.

The repository is organised so that the experimental implementation, analysis, and visualisation are separated. This allows the generated result tables to be reused for plotting without having to rerun the computationally expensive experiments.

## Author
- Edgar Kolian (https://github.com/kolianedgar)
