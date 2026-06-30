from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_experiment_grid
import os

PLOT_DIR = "results/plots/small"

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# EXPERIMENT GRID
# ============================================================

GRID_CONFIG_SMALL = {

    "dimensions": [2, 10],

    "lambda_grid": [0.0, 0.5, 1.0, 1.5, 2.0],

    "alpha_grid": [1.0, 1.5, 2.0],

    "epsilon_grid": [0.0, 0.1, 0.2, 0.3],

    "model_types": [
        "ols",
        "rbf_svr",
    ],

    "target_modes": [
        "linear",
    ],

    "shift_types": [
        "mean",
        "combined",
    ],

    "seeds": [0, 1, 2],

    "n_train": 1000,
    "n_test": 1000,

    "sigma": 0.1,
}

results_df = run_experiment_grid(
    config=GRID_CONFIG_SMALL,
    save_path="results/tables/small/results.csv",
    preview_rows=5,
)