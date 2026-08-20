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
    export_mc_vs_true_chi_squared_summary   
)

PLOT_DIR_GAP_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/gen_gap_vs_chi_sq_div"
PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/weight_variance_vs_chi_sq_divergence"
PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTENDED = "results/plots/extended/true_chi_sq/chi_sq_vs_lambda"
PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTENDED = "results/plots/extended/true_chi_sq/chi_sq_vs_alpha"
PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/ess_vs_chi_sq_divergence"
CSV_PATH_MC_CSD_VS_TRUE_CSD_EXTENDED = "results/plots/extended/mc_div_vs_true_div/summary.csv"

results_df_extended = pd.read_csv(
    "results/tables/extended/results.csv"
)

epsilon_grid_extended = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
dimension_grid_extended = [2, 10, 50]
n_train = 1000

for epsilon in epsilon_grid_extended:
    for dimension in dimension_grid_extended:

        plot_generalisation_gap_vs_chi_squared_fixed_epsilon(results_df_extended, PLOT_DIR_GAP_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, dimension=dimension)
        plot_true_w_var_vs_chi_sq_fixed_epsilon(results_df_extended, PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, dimension=dimension)
        plot_true_ess_vs_chi_squared_fixed_epsilon(results_df_extended, PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, n_train=n_train, dimension=dimension)

summary=export_mc_vs_true_chi_squared_summary(results_df_extended, CSV_PATH_MC_CSD_VS_TRUE_CSD_EXTENDED)

plot_chi_squared_vs_alpha(results_df_extended, PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTENDED)

plot_chi_squared_vs_lambda(results_df_extended, PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTENDED)

print("[DONE] Extended synthetic plots generated.")