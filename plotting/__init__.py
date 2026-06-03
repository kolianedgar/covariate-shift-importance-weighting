# plotting/__init__.py

from .figures import (
    plot_test_mse_vs_kl,
    plot_ess_vs_kl,
    plot_weight_variance_vs_kl,
    plot_test_mse_vs_epsilon,
    plot_generalisation_gap_vs_kl,
    plot_test_mse_vs_dimension,
    plot_weight_histogram,
    plot_ess_heatmap,
    generate_all_plots,
    plot_test_mse_vs_kl_by_model,
    plot_ess_vs_kl_by_model,
    plot_weight_variance_vs_kl_by_model,
    plot_test_mse_vs_dimension_by_model,
    plot_generalisation_gap_vs_kl_by_model
)

__all__ = [
    "plot_test_mse_vs_kl",
    "plot_ess_vs_kl",
    "plot_weight_variance_vs_kl",
    "plot_test_mse_vs_epsilon",
    "plot_generalisation_gap_vs_kl",
    "plot_test_mse_vs_dimension",
    "plot_weight_histogram",
    "plot_ess_heatmap",
    "generate_all_plots",
    "plot_test_mse_vs_kl_by_model",
    "plot_ess_vs_kl_by_model",
    "plot_weight_variance_vs_kl_by_model",
    "plot_test_mse_vs_dimension_by_model",
    "plot_generalisation_gap_vs_kl_by_model",
]