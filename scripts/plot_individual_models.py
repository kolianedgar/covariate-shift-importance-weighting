import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_generalisation_gap_vs_chi_squared_mc_fixed_epsilon,
    plot_w_var_vs_chi_sq_mc_fixed_epsilon,
    plot_ess_vs_chi_squared_mc_fixed_epsilon,
    plot_test_mse_vs_ess_fixed_epsilon,
    plot_chi_squared_mc_vs_lambda,
    plot_chi_squared_mc_vs_alpha,
    plot_generalisation_gap_vs_chi_squared_fixed_epsilon,
    plot_true_w_var_vs_chi_sq_fixed_epsilon,
    plot_empirical_ess_vs_chi_squared_fixed_epsilon,
    plot_chi_squared_vs_lambda,
    plot_chi_squared_vs_alpha ,
    plot_empirical_ess_vs_chi_squared_fixed_epsilon   
)

PLOT_DIR_GAP_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/gen_gap_vs_chi_sq_div"
PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/weight_variance_vs_chi_sq_divergence"
PLOT_DIR_ESS_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/ess_vs_chi_sq_divergence"
PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_SMALL = "results/plots/small/true_chi_sq/true_ess_vs_chi_sq_divergence"
PLOT_DIR_TRUE_CSD_VS_LAMBDA_SMALL = "results/plots/small/true_chi_sq/chi_sq_vs_lambda"

PLOT_DIR_GAP_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/gen_gap_vs_chi_sq_div"
PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/weight_variance_vs_chi_sq_divergence"
PLOT_DIR_ESS_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/ess_vs_chi_sq_divergence"
PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTENDED = "results/plots/extended/true_chi_sq/true_ess_vs_chi_sq_divergence"
PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTENDED = "results/plots/extended/true_chi_sq/chi_sq_vs_lambda"
PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTENDED = "results/plots/extended/true_chi_sq/chi_sq_vs_alpha"

results_df_small = pd.read_csv(
    "results/tables/small/results.csv"
)

results_df_extended = pd.read_csv(
    "results/tables/extended/results.csv"
)

epsilon_grid_small = [0.0, 0.1, 0.2, 0.3]
dimension_grid_small = [2, 10]
n_train = 1000

for epsilon in epsilon_grid_small:
    for dimension in dimension_grid_small:
        if epsilon==0.0:
            continue

        plot_generalisation_gap_vs_chi_squared_fixed_epsilon(results_df_small, PLOT_DIR_GAP_VS_TRUE_CSD_SMALL, epsilon=epsilon, dimension=dimension)
        # plot_true_w_var_vs_chi_sq_fixed_epsilon(results_df_small, PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_SMALL, epsilon=epsilon, dimension=dimension)
        # plot_empirical_ess_vs_chi_squared_fixed_epsilon(results_df_small, PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_SMALL, epsilon=epsilon, n_train=n_train, dimension=dimension)

plot_chi_squared_vs_lambda(results_df_small, PLOT_DIR_TRUE_CSD_VS_LAMBDA_SMALL)

epsilon_grid_extended = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
dimension_grid_extended = [2, 10, 50]

for epsilon in epsilon_grid_extended:
    for dimension in dimension_grid_extended:
        if epsilon==0.0:
            continue

        plot_generalisation_gap_vs_chi_squared_fixed_epsilon(results_df_extended, PLOT_DIR_GAP_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, dimension=dimension)
#         plot_true_w_var_vs_chi_sq_fixed_epsilon(results_df_extended, PLOT_DIR_WEIGHT_VARIANCE_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, dimension=dimension)
#         plot_empirical_ess_vs_chi_squared_fixed_epsilon(results_df_extended, PLOT_DIR_TRUE_ESS_VS_TRUE_CSD_EXTENDED, epsilon=epsilon, n_train=n_train, dimension=dimension)

# plot_chi_squared_vs_alpha(results_df_extended, PLOT_DIR_TRUE_CSD_VS_ALPHA_EXTENDED, alpha_max=1.5)

# plot_chi_squared_vs_lambda(results_df_extended, PLOT_DIR_TRUE_CSD_VS_LAMBDA_EXTENDED)
