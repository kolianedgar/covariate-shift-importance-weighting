import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import (
    plot_test_mse_vs_kl_by_model,
    plot_ess_vs_kl_by_model,
    plot_weight_variance_vs_kl_by_model,
    plot_generalisation_gap_vs_kl_by_model,
    plot_test_mse_vs_dimension_by_model,
    plot_test_mse_vs_kl_fixed_epsilon,
    plot_weight_variance_vs_kl_fixed_epsilon
)

PLOT_DIR = "results/plots/debug"

results_df = pd.read_csv(
    "results/tables/extended/results.csv"
)

plot_test_mse_vs_kl_fixed_epsilon(results_df, PLOT_DIR, epsilon=0.05)
plot_test_mse_vs_kl_fixed_epsilon(results_df, PLOT_DIR, epsilon=0.50)

plot_weight_variance_vs_kl_fixed_epsilon(results_df, PLOT_DIR, epsilon=0.05)
plot_weight_variance_vs_kl_fixed_epsilon(results_df, PLOT_DIR, epsilon=0.50)

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