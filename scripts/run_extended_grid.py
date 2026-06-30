from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_experiment_grid
import os

PLOT_DIR = "results/plots/extended"

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# EXPERIMENT GRID
# ============================================================

GRID_CONFIG = {

    # --------------------------------------------------------
    # dimensionality
    # --------------------------------------------------------

    "dimensions": [
        2,
        10,
        50,
    ],

    # --------------------------------------------------------
    # mean-shift magnitude
    # --------------------------------------------------------

    "lambda_grid": [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ],

    # --------------------------------------------------------
    # covariance scaling
    # --------------------------------------------------------

    "alpha_grid": [
        1.00,
        1.15,
        1.30,
        1.45,
        1.60,
        1.75,
        1.90,
    ],

    # --------------------------------------------------------
    # contamination levels
    # --------------------------------------------------------

    "epsilon_grid": [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ],

    # --------------------------------------------------------
    # models
    # --------------------------------------------------------

    "model_types": [

        # unweighted
        "ols",
        "linear_svr",
        "rbf_svr",

        # weighted
        "weighted_ols",
        "weighted_linear_svr",
        "weighted_rbf_svr",
    ],

    # --------------------------------------------------------
    # target structures
    # --------------------------------------------------------

    "target_modes": [
        "linear",
        "nonlinear",
    ],

    # --------------------------------------------------------
    # shift categories
    # --------------------------------------------------------

    "shift_types": [
        "mean",
        "covariance",
        "combined",
    ],

    # --------------------------------------------------------
    # random seeds
    # --------------------------------------------------------

    "seeds": [
        0,
        1,
        2,
        3,
        4,
    ],

    # --------------------------------------------------------
    # fixed experimental parameters
    # --------------------------------------------------------

    "n_train": 1000,
    "n_test": 1000,

    "sigma": 0.1,
}

results_df = run_experiment_grid(
    config=GRID_CONFIG,
    save_path="results/tables/extended/results.csv",
    preview_rows=5,
)

results_df.to_csv(
    "results/tables/extended/results.csv",
    index=False
)
print("[SAVED] results/tables/extended/results.csv")