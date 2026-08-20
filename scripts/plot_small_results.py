import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_generalisation_gap_vs_chi_squared_fixed_epsilon,
    plot_true_w_var_vs_chi_sq_fixed_epsilon,
    plot_chi_squared_vs_lambda,
    plot_true_ess_vs_chi_squared_fixed_epsilon,
    export_mc_vs_true_chi_squared_summary   
)

PLOT_DIR_GAP_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/gen_gap_vs_chi_sq_div"
PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/weight_variance_vs_chi_sq_divergence"
PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/ess_vs_chi_sq_divergence"
PLOT_DIR_TRUE_CSD_VS_LAMBDA_SMALL = "results/plots/small/true_chi_sq/chi_sq_vs_lambda"
CSV_PATH_MC_CSD_VS_TRUE_CSD_SMALL = "results/plots/small/mc_div_vs_true_div/summary.csv"

results_df_small = pd.read_csv(
    "results/tables/small/results.csv"
)

epsilon_grid_small = [0.0, 0.1, 0.2, 0.3]
dimension_grid_small = [2, 10]
n_train = 1000

for epsilon in epsilon_grid_small:
    for dimension in dimension_grid_small:

        plot_generalisation_gap_vs_chi_squared_fixed_epsilon(results_df_small, PLOT_DIR_GAP_VS_TRUE_CSD_SMALL, epsilon=epsilon, dimension=dimension)
        plot_true_w_var_vs_chi_sq_fixed_epsilon(results_df_small, PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_SMALL, epsilon=epsilon, dimension=dimension)
        plot_true_ess_vs_chi_squared_fixed_epsilon(results_df_small, PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_SMALL, epsilon=epsilon, n_train=n_train, dimension=dimension)

plot_chi_squared_vs_lambda(results_df_small, PLOT_DIR_TRUE_CSD_VS_LAMBDA_SMALL)

summary=export_mc_vs_true_chi_squared_summary(results_df_small, CSV_PATH_MC_CSD_VS_TRUE_CSD_SMALL)