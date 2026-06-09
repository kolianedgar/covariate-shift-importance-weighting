import matplotlib.pyplot as plt
from .helpers import save_figure
import numpy as np
from analysis import aggregate_results
import matplotlib.lines as mlines
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
 
SHIFT_COLORS = {
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

# ============================================================
# 1 - Test MSE vs Chi-Squared Divergence - Value of Epsilon Fixed
# ============================================================

def plot_test_mse_vs_chi_squared_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    chi_sq_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot Test MSE vs KL divergence, fixed at a single epsilon level.
 
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
    kl_n_bins   : int
        Number of equal-width bins for KL divergence aggregation.
        Binning prevents multiple KL values mapping to the same x point
        across different (lambda, alpha) combinations.
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
    # 2. BIN KL DIVERGENCE
    #    KL is a continuous value driven by (lambda, alpha, epsilon).
    #    Binning gives stable x-axis points for aggregation.
    # --------------------------------------------------------
 
    data = data.copy()
    data["chi_sq_bin"] = pd.cut(
        data["chi_squared_divergence"],
        bins=chi_sq_n_bins,
        labels=False,
    )
 
    # Bin centre for plotting
    bin_edges = pd.cut(
        data["chi_squared_divergence"],
        bins=chi_sq_n_bins,
    ).cat.categories
 
    bin_centres = np.array([interval.mid for interval in bin_edges])
 
    # --------------------------------------------------------
    # 3. AGGREGATE over seeds (and lambda/alpha within each bin)
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["model_type", "shift_type", "chi_sq_bin"],
        metric_cols=["test_mse"],
    )
 
    # Map bin index to bin centre
    agg["chi_sq_mid"] = agg["chi_sq_bin"].apply(
        lambda b: bin_centres[int(b)] if not pd.isna(b) else np.nan
    )
    agg = agg.dropna(subset=["chi_sq_mid"])
 
    # --------------------------------------------------------
    # 4. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, (model_uw, model_w, panel_title) in zip(axes, MODEL_PAIRS):
 
        for shift in SHIFT_TYPES:
 
            ls    = SHIFT_LINESTYLES[shift]
            label = SHIFT_LABELS[shift]
 
            for model, color, weight_label in [
                (model_uw, COLOUR_UNWEIGHTED, "unweighted"),
                (model_w,  COLOUR_WEIGHTED,   "weighted"),
            ]:
 
                subset = agg[
                    (agg["model_type"] == model) &
                    (agg["shift_type"] == shift)
                ].sort_values("chi_sq_mid")
 
                if subset.empty:
                    continue
 
                ax.plot(
                    subset["chi_sq_mid"],
                    subset["test_mse_mean"],
                    color=color,
                    linestyle=ls,
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                )
 
 
 
        ax.set_title(panel_title, fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("Chi-Squared Divergence", fontsize=10)
        ax.set_ylabel("Test MSE",      fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
 
    # --------------------------------------------------------
    # 5. SHARED LEGEND
    #    Two legend groups: colour (weighted/unweighted) + line style (shift type)
    # --------------------------------------------------------
 
    color_handles = [
        mlines.Line2D([], [], color=COLOUR_UNWEIGHTED, linewidth=2, label="unweighted"),
        mlines.Line2D([], [], color=COLOUR_WEIGHTED,   linewidth=2, label="weighted"),
    ]
 
    style_handles = [
        mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=2, label=SHIFT_LABELS[s])
        for s, ls in SHIFT_LINESTYLES.items()
    ]
 
    all_handles = color_handles + style_handles
 
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )
 
    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_str = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"Test MSE vs Chi-Squared Divergence  (d={dimension}, ε={epsilon}{target_str})",
        fontsize=12,
        y=1.02,
    )
 
    eps_str    = str(epsilon).replace(".", "p")
    target_str = f"_{target_mode}" if target_mode else ""
    filename   = f"test_mse_vs_chi_sq_eps{eps_str}_d{dimension}{target_str}.png"
 
    save_figure(filename, plot_dir)
 
# ==================================================================
# 2 - Variance of Weights vs KL Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_weight_variance_vs_chi_sq_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    kl_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot importance weight variance vs KL divergence, fixed at a single
    epsilon level.
 
    Since importance weights depend only on the distributions (not the
    model), we deduplicate by taking one representative model before
    aggregating, avoiding inflated averaging over identical weight values.
 
    Layout
    ------
    3 panels side by side, one per shift type:
        Mean Shift | Covariance Shift | Combined Shift
 
    Each panel has a single line: weight variance vs KL divergence,
    aggregated over seeds (and lambda/alpha within each KL bin).
 
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
    kl_n_bins   : int
        Number of equal-width bins for KL divergence aggregation.
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
    # 2. DEDUPLICATE BY MODEL
    #    Weight variance is identical across models for the same
    #    (lambda, alpha, epsilon, dimension, seed) combination.
    #    Keep one representative model to avoid duplicated rows.
    # --------------------------------------------------------
 
    data = data[data["model_type"] == "ols"].copy()
 
    # --------------------------------------------------------
    # 3. BIN KL DIVERGENCE
    # --------------------------------------------------------
 
    data["chi_sq_bin"] = pd.cut(
        data["chi_squared_divergence"],
        bins=kl_n_bins,
        labels=False,
    )
 
    bin_edges = pd.cut(
        data["chi_squared_divergence"],
        bins=kl_n_bins,
    ).cat.categories
 
    bin_centres = np.array([interval.mid for interval in bin_edges])
 
    # --------------------------------------------------------
    # 4. AGGREGATE over seeds (and lambda/alpha within each bin)
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["shift_type", "chi_sq_bin"],
        metric_cols=["weight_variance"],
    )
 
    agg["chi_sq_mid"] = agg["chi_sq_bin"].apply(
        lambda b: bin_centres[int(b)] if not pd.isna(b) else np.nan
    )
    agg = agg.dropna(subset=["chi_sq_mid"])
 
    # --------------------------------------------------------
    # 5. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, shift in zip(axes, SHIFT_TYPES):
 
        subset = agg[
            agg["shift_type"] == shift
        ].sort_values("chi_sq_mid")
 
        if subset.empty:
            ax.set_title(SHIFT_TITLES[shift], fontsize=11)
            ax.set_xlabel("Chi-Squared Divergence", fontsize=10)
            ax.set_ylabel("Weight Variance", fontsize=10)
            continue
 
        color = SHIFT_COLORS[shift]
 
        ax.plot(
            subset["chi_sq_mid"],
            subset["weight_variance_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=SHIFT_LABELS[shift],
        )
 
        ax.set_title(SHIFT_TITLES[shift], fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("Chi-Squared Divergence", fontsize=10)
        ax.set_ylabel("Weight Variance", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
 
    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_str = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"Importance Weight Variance vs Chi-Squared Divergence  (d={dimension}, ε={epsilon}{target_str})",
        fontsize=12,
        y=1.02,
    )
 
    eps_str    = str(epsilon).replace(".", "p")
    target_tag = f"_{target_mode}" if target_mode else ""
    filename   = f"weight_variance_vs_chi_sq_eps{eps_str}_d{dimension}{target_tag}.png"
 
    save_figure(filename, plot_dir)

# ==================================================================
# 3 - ESS vs KL Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_ess_vs_chi_sq_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    kl_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot Effective Sample Size (ESS) vs KL divergence, fixed at a single
    epsilon level.
 
    Since ESS depends only on the distributions (not the
    model), we deduplicate by taking one representative model before
    aggregating, avoiding inflated averaging over identical weight values.
 
    Layout
    ------
    3 panels side by side, one per shift type:
        Mean Shift | Covariance Shift | Combined Shift
 
    Each panel has a single line: weight variance vs KL divergence,
    aggregated over seeds (and lambda/alpha within each KL bin).
 
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
    kl_n_bins   : int
        Number of equal-width bins for KL divergence aggregation.
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
    # 2. DEDUPLICATE BY MODEL
    #    Weight variance is identical across models for the same
    #    (lambda, alpha, epsilon, dimension, seed) combination.
    #    Keep one representative model to avoid duplicated rows.
    # --------------------------------------------------------
 
    data = data[data["model_type"] == "ols"].copy()
 
    # --------------------------------------------------------
    # 3. BIN KL DIVERGENCE
    # --------------------------------------------------------
 
    data["chi_sq_bin"] = pd.cut(
        data["chi_squared_divergence"],
        bins=kl_n_bins,
        labels=False,
    )
 
    bin_edges = pd.cut(
        data["kl_divergence"],
        bins=kl_n_bins,
    ).cat.categories
 
    bin_centres = np.array([interval.mid for interval in bin_edges])
 
    # --------------------------------------------------------
    # 4. AGGREGATE over seeds (and lambda/alpha within each bin)
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["shift_type", "chi_sq_bin"],
        metric_cols=["ess"],
    )
 
    agg["chi_sq_mid"] = agg["chi_sq_bin"].apply(
        lambda b: bin_centres[int(b)] if not pd.isna(b) else np.nan
    )
    agg = agg.dropna(subset=["chi_sq_mid"])
 
    # --------------------------------------------------------
    # 5. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, shift in zip(axes, SHIFT_TYPES):
 
        subset = agg[
            agg["shift_type"] == shift
        ].sort_values("chi_sq_mid")
 
        if subset.empty:
            ax.set_title(SHIFT_TITLES[shift], fontsize=11)
            ax.set_xlabel("Chi-Squared Divergence", fontsize=10)
            ax.set_ylabel("ESS", fontsize=10)
            continue
 
        color = SHIFT_COLORS[shift]
 
        ax.plot(
            subset["chi_sq_mid"],
            subset["ess_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=SHIFT_LABELS[shift],
        )
 
        ax.set_title(SHIFT_TITLES[shift], fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("Chi-Squared Divergence", fontsize=10)
        ax.set_ylabel("ESS", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
 
    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_str = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"Effective Sample Size vs Chi-Squared Divergence  (d={dimension}, ε={epsilon}{target_str})",
        fontsize=12,
        y=1.02,
    )
 
    eps_str    = str(epsilon).replace(".", "p")
    target_tag = f"_{target_mode}" if target_mode else ""
    filename   = f"ess_vs_chi_sq_eps{eps_str}_d{dimension}{target_tag}.png"
 
    save_figure(filename, plot_dir)


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
 
            for model, color in [
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
                    color=color,
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
 
    color_handles = [
        mlines.Line2D([], [], color=COLOUR_UNWEIGHTED, linewidth=2, label="unweighted"),
        mlines.Line2D([], [], color=COLOUR_WEIGHTED,   linewidth=2, label="weighted"),
    ]
 
    style_handles = [
        mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=2, label=SHIFT_LABELS[s])
        for s, ls in SHIFT_LINESTYLES.items()
    ]
 
    fig.legend(
        handles=color_handles + style_handles,
        loc="lower center",
        ncol=len(color_handles) + len(style_handles),
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
    filename   = f"test_mse_vs_ess_eps{eps_str}_d{dimension}{target_tag}.png"
 
    save_figure(filename, plot_dir)