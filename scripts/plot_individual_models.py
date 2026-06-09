import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_test_mse_vs_chi_squared_fixed_epsilon,
    plot_weight_variance_vs_chi_sq_fixed_epsilon,
    plot_ess_vs_chi_sq_fixed_epsilon,
    plot_test_mse_vs_ess_fixed_epsilon
)

PLOT_DIR_TEST_VS_CSD = "results/plots/small/test_mse_vs_chi_sq_divergence"
PLOT_DIR_WEIGHT_VARIANCE_VS_CSD = "results/plots/small/weight_variance_vs_chi_sq_divergence"
PLOT_DIR_ESS_VS_CSD = "results/plots/small/ess_vs_chi_sq_divergence"
PLOT_DIR_TEST_VS_ESS = "results/plots/small/test_mse_vs_ess"

results_df = pd.read_csv(
    "results/tables/small/results.csv"
)

epsilon_grid = [0.0, 0.1, 0.2, 0.3]
dimension_grid = [2, 10]

for epsilon in epsilon_grid:
    for dimension in dimension_grid:
        if epsilon==0.0:
            continue

        plot_test_mse_vs_chi_squared_fixed_epsilon(results_df, PLOT_DIR_TEST_VS_CSD, epsilon=epsilon, dimension=dimension)
        plot_ess_vs_chi_sq_fixed_epsilon(results_df, PLOT_DIR_ESS_VS_CSD, epsilon=epsilon, dimension=dimension)
        plot_weight_variance_vs_chi_sq_fixed_epsilon(results_df, PLOT_DIR_WEIGHT_VARIANCE_VS_CSD, epsilon=epsilon, dimension=dimension)
        plot_test_mse_vs_ess_fixed_epsilon(results_df, PLOT_DIR_TEST_VS_ESS, epsilon=epsilon, dimension=dimension)