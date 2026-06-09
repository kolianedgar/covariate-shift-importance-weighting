import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from .helpers import save_figure
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

DIMENSIONS = [2, 10, 50]
 
DIM_COLORS = {
    2:  "#4C9BE8",
    10: "#E8734C",
    50: "#6ABF69",
}
 
DIM_LABELS = {
    2:  "d = 2",
    10: "d = 10",
    50: "d = 50",
}

def _nice_x_formatter(values):
    """
    Choose a clean tick formatter based on the magnitude of the x values.
    Returns a matplotlib FuncFormatter.
    """
    max_val = values.max() if len(values) > 0 else 1.0
 
    if max_val < 0.01:
        # Scientific notation for very small values
        return ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    elif max_val < 0.1:
        return ticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    elif max_val < 1.0:
        return ticker.FuncFormatter(lambda x, _: f"{x:.3f}")
    else:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    
def _nice_y_formatter(values):
    """
    Choose a clean tick formatter based on the magnitude of the y values.
    Returns a matplotlib FuncFormatter.
    """
    max_val = values.max() if len(values) > 0 else 1.0
 
    if max_val < 0.01:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    elif max_val < 0.1:
        return ticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    elif max_val < 1.0:
        return ticker.FuncFormatter(lambda x, _: f"{x:.3f}")
    else:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2f}")

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
    Plot Test MSE vs Chi-Squared divergence, fixed at a single epsilon level.
 
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
        Number of equal-width bins for Chi-Squared divergence aggregation.
        Binning prevents multiple Chi-Squared values mapping to the same x point
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
    # 2. BIN Chi-Squared DIVERGENCE
    #    Chi-Squared is a continuous value driven by (lambda, alpha, epsilon).
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
# 2 - Var(w(x)) vs Chi-Squared Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_weight_variance_vs_chi_sq_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    chi_sq_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot importance weight variance vs Chi-Squared divergence, fixed at a single
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
        Number of equal-width bins for Chi-Squared divergence aggregation.
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
    # 3. BIN Chi-Squared DIVERGENCE
    # --------------------------------------------------------
 
    data["chi_sq_bin"] = pd.cut(
        data["chi_squared_divergence"],
        bins=chi_sq_n_bins,
        labels=False,
    )
 
    bin_edges = pd.cut(
        data["chi_squared_divergence"],
        bins=chi_sq_n_bins,
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
# 3 - ESS vs Chi-Squared Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_ess_vs_chi_sq_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    chi_sq_n_bins=20,
    figsize=(16, 5),
):
    """
    Plot Effective Sample Size (ESS) vs Chi-Squared divergence, fixed at
    a single epsilon level.
 
    Since ESS depends only on the distributions (not the model), we
    deduplicate by taking one representative model before aggregating,
    avoiding inflated averaging over identical weight values.
 
    Layout
    ------
    3 panels side by side, one per shift type:
        Mean Shift | Covariance Shift | Combined Shift
 
    Each panel has a single line: ESS vs Chi-Squared divergence,
    aggregated over seeds (and lambda/alpha within each Chi-Squared bin).
 
    Parameters
    ----------
    df            : pd.DataFrame
        Full results dataframe.
    plot_dir      : str
        Directory to save the figure.
    epsilon       : float
        Contamination level to fix (e.g. 0.05 or 0.50).
    dimension     : int
        Dimension to fix (default 10).
    target_mode   : str or None
        If provided, filters to "linear" or "nonlinear".
    chi_sq_n_bins : int
        Number of equal-width bins for Chi-Squared divergence aggregation.
    figsize       : tuple
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
    # --------------------------------------------------------
 
    data = data[data["model_type"] == "ols"].copy()
 
    # --------------------------------------------------------
    # 3. BIN CHI-SQUARED DIVERGENCE PER SHIFT TYPE
    #    Bin separately per shift type so each panel uses the
    #    full range of its own chi-squared values.
    # --------------------------------------------------------
 
    binned_parts = []
 
    for shift in SHIFT_TYPES:
 
        part = data[data["shift_type"] == shift].copy()
 
        if part.empty:
            continue
 
        part["chi_sq_bin"] = pd.cut(
            part["chi_squared_divergence"],
            bins=chi_sq_n_bins,
            labels=False,
        )
 
        # Bin centres derived from chi_squared_divergence (not kl_divergence)
        bin_edges = pd.cut(
            part["chi_squared_divergence"],
            bins=chi_sq_n_bins,
        ).cat.categories
 
        bin_centres = np.array([interval.mid for interval in bin_edges])
 
        part["chi_sq_mid"] = part["chi_sq_bin"].apply(
            lambda b: bin_centres[int(b)] if not pd.isna(b) else np.nan
        )
 
        binned_parts.append(part)
 
    if not binned_parts:
        print("[WARN] No binned data produced. Skipping.")
        return
 
    data = pd.concat(binned_parts, ignore_index=True)
    data = data.dropna(subset=["chi_sq_mid"])
 
    # --------------------------------------------------------
    # 4. AGGREGATE over seeds (and lambda/alpha within each bin)
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["shift_type", "chi_sq_bin", "chi_sq_mid"],
        metric_cols=["ess"],
    )
 
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
            ax.set_xlabel("χ² Divergence", fontsize=10)
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
        )
 
        # Apply clean x-axis formatting based on value magnitude
        ax.xaxis.set_major_formatter(_nice_x_formatter(subset["chi_sq_mid"]))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
 
        ax.set_title(SHIFT_TITLES[shift], fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("χ² Divergence", fontsize=10)
        ax.set_ylabel("ESS", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
 
    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_str = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"Effective Sample Size vs χ² Divergence  (d={dimension}, ε={epsilon}{target_str})",
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

def plot_chi_squared_vs_lambda(
    df,
    plot_dir,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot Chi-Squared divergence vs lambda (mean shift magnitude).
 
    Fixes shift_type="mean" (alpha=1) and aggregates across all epsilon
    levels and seeds, since chi-squared depends only on the shift
    parameters and dimension, not on epsilon.
 
    Deduplicates to one representative model (OLS) before aggregating
    since chi-squared is identical across all models for the same
    experimental configuration.
 
    Layout
    ------
    3 panels side by side, one per dimension:
        d = 2 | d = 10 | d = 50
 
    Each panel has a single line: chi-squared vs lambda,
    aggregated over all seeds and epsilon levels.
 
    Parameters
    ----------
    df          : pd.DataFrame
        Full results dataframe.
    plot_dir    : str
        Directory to save the figure.
    target_mode : str or None
        If provided, filters to "linear" or "nonlinear".
        Since chi-squared doesn't depend on target_mode either,
        this only affects which rows are included before deduplication.
    figsize     : tuple
        Figure size.
    """
 
    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------
 
    data = df.copy()
 
    # Fix shift type to mean (alpha=1, lambda varies)
    data = data[data["shift_type"] == "mean"]
 
    if target_mode is not None:
        data = data[data["target_mode"] == target_mode]
 
    if data.empty:
        print("[WARN] No data for shift_type=mean. Skipping.")
        return
 
    # --------------------------------------------------------
    # 2. DEDUPLICATE BY MODEL
    #    Chi-squared is identical across models. Keep one
    #    representative to avoid inflating the aggregation.
    # --------------------------------------------------------
 
    data = data[data["model_type"] == "ols"].copy()
 
    # --------------------------------------------------------
    # 3. AGGREGATE over seeds and epsilon levels
    #    No binning needed — lambda has only 5 discrete values.
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["dimension", "lambda"],
        metric_cols=["chi_squared_divergence"],
    )
 
    # --------------------------------------------------------
    # 4. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, d in zip(axes, DIMENSIONS):
 
        subset = agg[
            agg["dimension"] == d
        ].sort_values("lambda")
 
        if subset.empty:
            ax.set_title(DIM_LABELS[d], fontsize=11)
            ax.set_xlabel("λ", fontsize=10)
            ax.set_ylabel("χ² Divergence", fontsize=10)
            continue
 
        color = DIM_COLORS[d]
 
        ax.plot(
            subset["lambda"],
            subset["chi_squared_divergence_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
 
        ax.set_title(DIM_LABELS[d], fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("λ (mean shift magnitude)", fontsize=10)
        ax.set_ylabel("χ² Divergence", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
 
    # --------------------------------------------------------
    # 5. TITLE AND SAVE
    # --------------------------------------------------------
 
    target_tag = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"χ² Divergence vs λ  (mean shift, aggregated over all ε{target_tag})",
        fontsize=12,
        y=1.02,
    )
 
    target_suffix = f"_{target_mode}" if target_mode else ""
    filename = f"chi_squared_vs_lambda{target_suffix}.png"
 
    save_figure(filename, plot_dir)

def plot_chi_squared_vs_alpha(
    df,
    plot_dir,
    target_mode=None,
    alpha_max=None,
    figsize=(16, 5),
):
    """
    Plot Chi-Squared divergence vs alpha (covariance scaling factor).
 
    Fixes shift_type="covariance" (lambda=0) and aggregates across all
    epsilon levels and seeds, since chi-squared depends only on the shift
    parameters and dimension, not on epsilon.
 
    Deduplicates to one representative model (OLS) before aggregating
    since chi-squared is identical across all models for the same
    experimental configuration.
 
    Under covariance shift, chi-squared grows exponentially with both
    alpha and dimension. At high d (e.g. d=50) and high alpha (e.g.
    alpha=2 or alpha=3), the values become extreme and compress the rest
    of the plot. Use alpha_max to exclude these if needed.
 
    Layout
    ------
    3 panels side by side, one per dimension:
        d = 2 | d = 10 | d = 50
 
    Each panel has a single line: chi-squared vs alpha,
    aggregated over all seeds and epsilon levels.
 
    Parameters
    ----------
    df          : pd.DataFrame
        Full results dataframe.
    plot_dir    : str
        Directory to save the figure.
    target_mode : str or None
        If provided, filters to "linear" or "nonlinear".
    alpha_max   : float or None
        If provided, excludes rows where alpha > alpha_max before
        plotting. Useful for preventing extreme chi-squared values at
        high alpha and high dimension from compressing the y-axis.
        E.g. alpha_max=1.5 excludes alpha=2.0 and alpha=3.0.
        Default None (no exclusion).
    figsize     : tuple
        Figure size.
    """
 
    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------
 
    data = df.copy()
 
    # Fix shift type to covariance (lambda=0, alpha varies)
    data = data[data["shift_type"] == "covariance"]
 
    if target_mode is not None:
        data = data[data["target_mode"] == target_mode]
 
    if alpha_max is not None:
        excluded = data[data["alpha"] > alpha_max]["alpha"].unique()
        if len(excluded) > 0:
            print(
                f"[INFO] Excluding alpha values > {alpha_max}: "
                f"{sorted(excluded)}"
            )
        data = data[data["alpha"] <= alpha_max]
 
    if data.empty:
        print("[WARN] No data for shift_type=covariance after filtering. Skipping.")
        return
 
    # --------------------------------------------------------
    # 2. DEDUPLICATE BY MODEL
    #    Chi-squared is identical across models. Keep one
    #    representative to avoid inflating the aggregation.
    # --------------------------------------------------------
 
    data = data[data["model_type"] == "ols"].copy()
 
    # --------------------------------------------------------
    # 3. AGGREGATE over seeds and epsilon levels
    #    No binning needed — alpha has only 5 discrete values
    #    (or fewer if alpha_max is set).
    # --------------------------------------------------------
 
    agg = aggregate_results(
        data,
        groupby_cols=["dimension", "alpha"],
        metric_cols=["chi_squared_divergence"],
    )
 
    # --------------------------------------------------------
    # 4. PLOT
    # --------------------------------------------------------
 
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
 
    for ax, d in zip(axes, DIMENSIONS):
 
        subset = agg[
            agg["dimension"] == d
        ].sort_values("alpha")
 
        if subset.empty:
            ax.set_title(DIM_LABELS[d], fontsize=11)
            ax.set_xlabel("α", fontsize=10)
            ax.set_ylabel("χ² Divergence", fontsize=10)
            continue
 
        color = DIM_COLORS[d]
 
        ax.plot(
            subset["alpha"],
            subset["chi_squared_divergence_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
 
        # Clean y-axis formatting based on value magnitude
        ax.yaxis.set_major_formatter(
            _nice_y_formatter(subset["chi_squared_divergence_mean"])
        )
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))
 
        ax.set_title(DIM_LABELS[d], fontsize=11, fontweight="normal", pad=8)
        ax.set_xlabel("α (covariance scale factor)", fontsize=10)
        ax.set_ylabel("χ² Divergence", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(left=subset["alpha"].min() - 0.05)
        ax.set_ylim(bottom=0)
 
    # --------------------------------------------------------
    # 5. TITLE AND SAVE
    # --------------------------------------------------------
 
    alpha_str  = f", α ≤ {alpha_max}" if alpha_max is not None else ""
    target_tag = f", target={target_mode}" if target_mode else ""
    fig.suptitle(
        f"χ² Divergence vs α  (covariance shift, aggregated over all ε{alpha_str}{target_tag})",
        fontsize=12,
        y=1.02,
    )
 
    alpha_suffix  = f"_alphamax{str(alpha_max).replace('.', 'p')}" if alpha_max is not None else ""
    target_suffix = f"_{target_mode}" if target_mode else ""
    filename      = f"chi_squared_vs_alpha{alpha_suffix}{target_suffix}.png"
 
    save_figure(filename, plot_dir)