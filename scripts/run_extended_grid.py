from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_external_experiment_grid
import os

PLOT_DIR = "results/plots/extended"

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# EXPERIMENT GRID
# ============================================================

GRID_CONFIG_EXTERNAL = {

    # --------------------------------------------------------
    # external datasets
    # --------------------------------------------------------

    "datasets": [
        {
            "source": "sklearn",
            "dataset_name": "wisconsin",
            "version": 1
        },
        {
            "source": "sklearn",
            "dataset_name": "tecator",
            "version": 1
        },
        {
            "source": "sklearn",
            "dataset_name": "sberbank_housing",
            "version": 1
        },
        # {
        #     "source": "csv",
        #     "file_path": "...",
        #     "target_column": "...",
        # },
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

    "sigma": 0.1,
}

results_df = run_external_experiment_grid(
    config=GRID_CONFIG_EXTERNAL,
    save_path="results/tables/results.csv",
    preview_rows=5,
)

results_df.to_csv(
    "results/tables/results.csv",
    index=False
)
print("[SAVED] results/tables/results.csv")