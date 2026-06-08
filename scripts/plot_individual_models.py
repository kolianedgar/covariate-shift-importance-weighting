import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_test_mse_vs_kl_fixed_epsilon,
    plot_weight_variance_vs_kl_fixed_epsilon,
    plot_ess_vs_kl_fixed_epsilon,
)

PLOT_DIR_TEST_VS_KLD = "results/plots/extended/test_mse_vs_kl_divergence"
PLOT_DIR_WEIGHT_VARIANCE_VS_KLD = "results/plots/extended/weight_variance_vs_kl_divergence"
PLOT_DIR_ESS_VS_KLD = "results/plots/extended/ess_vs_kl_divergence"

results_df = pd.read_csv(
    "results/tables/extended/results.csv"
)

epsilon_grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50,]
dimension_grid = [2, 10, 50]

for epsilon in epsilon_grid:
    for dimension in dimension_grid:
        if epsilon==0.0:
            continue

        plot_test_mse_vs_kl_fixed_epsilon(results_df, PLOT_DIR_TEST_VS_KLD, epsilon=epsilon, dimension=dimension)
        plot_ess_vs_kl_fixed_epsilon(results_df, PLOT_DIR_ESS_VS_KLD, epsilon=epsilon, dimension=dimension)
        plot_weight_variance_vs_kl_fixed_epsilon(results_df, PLOT_DIR_WEIGHT_VARIANCE_VS_KLD, epsilon=epsilon, dimension=dimension)

# plot_test_mse_vs_kl_by_model(
#     results_df,
#     PLOT_DIR,
# )

# plot_ess_vs_kl_by_model(
#     results_df,
#     PLOT_DIR,
# )

# plot_weight_variance_vs_kl_by_model(
#     results_df,
#     PLOT_DIR,
# )

# plot_generalisation_gap_vs_kl_by_model(
#     results_df,
#     PLOT_DIR,
# )

# plot_test_mse_vs_dimension_by_model(
#     results_df,
#     PLOT_DIR,
# )

# print("[DONE] Individual-model plots generated.")