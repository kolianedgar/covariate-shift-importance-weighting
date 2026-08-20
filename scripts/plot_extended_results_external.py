import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_generalisation_gap_vs_chi_squared_fixed_epsilon,
    plot_true_w_var_vs_chi_sq_fixed_epsilon,
    plot_chi_squared_vs_lambda,
    plot_chi_squared_vs_alpha,
    plot_true_ess_vs_chi_squared_fixed_epsilon,
    export_mc_vs_true_chi_squared_summary,
)

PLOT_DIR_GAP_VS_TRUE_CSD_EXTERNAL = (
    "results/plots/external/true_chi_sq/gen_gap_vs_chi_sq_div"
)

PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTERNAL = (
    "results/plots/external/true_chi_sq/weight_variance_vs_chi_sq_divergence"
)

PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTERNAL = (
    "results/plots/external/true_chi_sq/chi_sq_vs_lambda"
)

PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTERNAL = (
    "results/plots/external/true_chi_sq/chi_sq_vs_alpha"
)

PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTERNAL = (
    "results/plots/external/true_chi_sq/ess_vs_chi_sq_divergence"
)

CSV_PATH_MC_CSD_VS_TRUE_CSD_EXTERNAL = (
    "results/plots/external/mc_div_vs_true_div/summary.csv"
)

results_df_external = pd.read_csv(
    "results/tables/external/results.csv"
)

epsilon_grid_external = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]

dimensions = results_df_external["dimension"].unique()

for dimension in dimensions:

    dataset_df = results_df_external[
        results_df_external["dimension"] == dimension
    ]

    n_train = int(dataset_df["n_train"].iloc[0])

    for epsilon in epsilon_grid_external:

        plot_generalisation_gap_vs_chi_squared_fixed_epsilon(
            dataset_df,
            PLOT_DIR_GAP_VS_TRUE_CSD_EXTERNAL,
            epsilon=epsilon,
            dimension=dimension,
        )

        plot_true_w_var_vs_chi_sq_fixed_epsilon(
            dataset_df,
            PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTERNAL,
            epsilon=epsilon,
            dimension=dimension,
        )

        plot_true_ess_vs_chi_squared_fixed_epsilon(
            dataset_df,
            PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTERNAL,
            epsilon=epsilon,
            n_train=n_train,
            dimension=dimension,
        )

summary = export_mc_vs_true_chi_squared_summary(
    results_df_external,
    CSV_PATH_MC_CSD_VS_TRUE_CSD_EXTERNAL,
)

plot_chi_squared_vs_alpha(
    results_df_external,
    PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTERNAL,
)

plot_chi_squared_vs_lambda(
    results_df_external,
    PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTERNAL,
)

print("[DONE] External dataset plots generated.")