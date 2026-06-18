import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

from .helpers import (
    save_figure,
)

from analysis import aggregate_results

# ============================================================
# CONSTANTS
# ============================================================

MODEL_PAIRS = [
    ("ols",          "weighted_ols",          "OLS / Weighted OLS"),
    ("linear_svr",   "weighted_linear_svr",   "Linear SVR / Weighted Linear SVR"),
    ("rbf_svr",      "weighted_rbf_svr",      "RBF SVR / Weighted RBF SVR"),
]

SHIFT_TYPES = ["mean", "covariance", "combined"]

SHIFT_LINESTYLES = {
    "mean":       "-",
    "covariance": "--",
    "combined":   ":",
}

SHIFT_LABELS = {
    "mean":       "Mean Shift",
    "covariance": "Covariance Shift",
    "combined":   "Combined Shift",
}

SHIFT_TYPES = ["mean", "covariance", "combined"]
 
SHIFT_COLOURS = {
    "mean":       "#4C9BE8",
    "covariance": "#E8734C",
    "combined":   "#6ABF69",
}

SHIFT_TITLES = {
    "mean":       "Mean Shift",
    "covariance": "Covariance Shift",
    "combined":   "Combined Shift",
}

COLOUR_UNWEIGHTED = "#4C9BE8"
COLOUR_WEIGHTED   = "#E8734C"

SHIFT_CMAPS = {
    "mean":       "Blues",
    "covariance": "Oranges",
    "combined":   "Greens",
}

DIMENSIONS = [2, 10, 50]
 
DIM_COLOURS = {
    2:  "#4C9BE8",
    10: "#E8734C",
    50: "#6ABF69",
}
 
DIM_LABELS = {
    2:  "d = 2",
    10: "d = 10",
    50: "d = 50",
}

SAMPLE_ESTIMATOR_EXPONENT_THRESHOLD = 10.0  # d * lambda^2 > 10 is unreliable


# ==================================================================
# 1. MSE vs ESS - Value of Epsilon Fixed
# ==================================================================

def plot_test_mse_vs_ess_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    ess_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot Test MSE vs ESS (Effective Sample Size), fixed at a single
    epsilon level.
 
    ESS is a property of the importance weight distribution and is
    identical across models for the same (lambda, alpha, epsilon,
    dimension, seed) combination. Test MSE is model-specific.
    The two are joined at the experiment level before aggregating.
 
    Layout
    ------
    3 panels side by side, one per model pair:
        OLS / Weighted OLS
        Linear SVR / Weighted Linear SVR
        RBF SVR / Weighted RBF SVR
 
    Encoding
    --------
    Colour    : blue = unweighted, orange = weighted
    Line style: solid = mean shift, dashed = covariance, dotted = combined
 
    Parameters
    ----------
    df          : pd.DataFrame
        Full results dataframe.
    plot_dir    : str
        Directory to save the figure.
    epsilon     : float
        Contamination level to fix (e.g. 0.05 or 0.50).
    dimension   : int
        Dimension to fix (default 10).
    target_mode : str or None
        If provided, filters to "linear" or "nonlinear".
    ess_n_bins  : int
        Number of equal-width bins for ESS aggregation.
        ESS is an outcome variable and not evenly spaced, so binning
        gives stable x-axis points for clean line plots.
    figsize     : tuple
        Figure size.
    """
 
    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------
 
    data = df.copy()
    data = data[data["epsilon"]   == epsilon]
    data = data[data["dimension"] == dimension]
 
    if target_mode is not None:
        data = data[data["target_mode"] == target_mode]
 
    if data.empty:
        print(f"[WARN] No data for epsilon={epsilon}, dimension={dimension}. Skipping.")
        return
 
    # --------------------------------------------------------
    # 2. BIN ESS
    #    ESS is an outcome variable driven by (lambda, alpha, epsilon).
    #    Binning gives stable x-axis points for aggregation.
    #    ESS is bounded in [0, n_train], here [0, 1000].
    # --------------------------------------------------------
 
    data = data.copy()
    data["ess_bin"] = pd.cut(
        data["ess"],
        bins=ess_n_bins,
        labels=False,
    )
 
    bin_edges = pd.cut(
        data["ess"],
        bins=ess_n_bins,
    ).cat.categories
 
    bin_centres = np.array([interval.mid for interval in bin_edges])
 
    # --------------------------------------------------------
    # 3. AGGREGATE over seeds (and lambda/alpha within each bin)
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["model_type", "shift_type", "ess_bin"],
        metric_cols=["test_mse"],
    )
 
    agg["ess_mid"] = agg["ess_bin"].apply(
        lambda b: bin_centres[int(b)] if not pd.isna(b) else np.nan
    )
    agg = agg.dropna(subset=["ess_mid"])
 
    # --------------------------------------------------------
    # 4. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, (model_uw, model_w, panel_title) in zip(axes, MODEL_PAIRS):
 
        for shift in SHIFT_TYPES:
 
            ls = SHIFT_LINESTYLES[shift]
 
            for model, colour in [
                (model_uw, COLOUR_UNWEIGHTED),
                (model_w,  COLOUR_WEIGHTED),
            ]:
 
                subset = agg[
                    (agg["model_type"] == model) &
                    (agg["shift_type"] == shift)
                ].sort_values("ess_mid")
 
                if subset.empty:
                    continue
 
                ax.plot(
                    subset["ess_mid"],
                    subset["test_mse_mean"],
                    color=colour,
                    linestyle=ls,
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                )
 
        ax.set_title(panel_title, fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("ESS", fontsize=10)
        ax.set_ylabel("Test MSE", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
 
    # --------------------------------------------------------
    # 5. SHARED LEGEND
    # --------------------------------------------------------
 
    colour_handles = [
        mlines.Line2D([], [], color=COLOUR_UNWEIGHTED, linewidth=2, label="unweighted"),
        mlines.Line2D([], [], color=COLOUR_WEIGHTED,   linewidth=2, label="weighted"),
    ]
 
    style_handles = [
        mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=2, label=SHIFT_LABELS[s])
        for s, ls in SHIFT_LINESTYLES.items()
    ]
 
    fig.legend(
        handles=colour_handles + style_handles,
        loc="lower center",
        ncol=len(colour_handles) + len(style_handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )
 
    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_str = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"Test MSE vs ESS  (d={dimension}, ε={epsilon}{target_str})",
        fontsize=12,
        y=1.02,
    )
 
    eps_str    = str(epsilon).replace(".", "p")
    target_tag = f"_{target_mode}" if target_mode else ""
    filename   = f"mse_vs_ess_eps{eps_str}_d{dimension}{target_tag}.png"
 
    save_figure(filename, plot_dir)

