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
            "data_id": 1027             # quake
        },
        {
            "source": "sklearn",
            "data_id": 197            # cpu-act
        },
        {
            "source": "sklearn",
            "data_id": 44964          # Superconductivity
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
        0.50,
        1.00,
        1.50,
    ],

    # --------------------------------------------------------
    # covariance scaling
    # --------------------------------------------------------

    "alpha_grid": [
        1.00,
        1.30,
        1.60,
        1.90,
    ],

    # --------------------------------------------------------
    # contamination levels
    # --------------------------------------------------------

    "epsilon_grid": [
        0.0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ],

    # --------------------------------------------------------
    # models
    # --------------------------------------------------------

    "model_types": [

        # unweighted
        "ols",
        "rbf_svr",

        # weighted
        "weighted_ols",
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

if __name__ == "__main__":

    results_df = run_external_experiment_grid(
        config=GRID_CONFIG_EXTERNAL,
        save_path="results/tables/external/results.csv",
        preview_rows=5,
    )

    print("[SAVED] results/tables/external/results.csv")