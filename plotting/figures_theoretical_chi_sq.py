import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

from .helpers import (
    save_figure,
    check_estimator_reliability,
    check_non_monotone,
    nice_x_formatter,
    nice_y_formatter
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

# ============================================================
# 1. Generalisation Gap vs χ² Divergence - Value of Epsilon Fixed
# ============================================================

def plot_generalisation_gap_vs_chi_squared_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot Generalisation Gap vs theoretical χ² divergence.

    Produces ONE FIGURE PER SHIFT TYPE.

    Example output files:

        gen_gap_vs_true_chi_sq_eps0p1_d2_mean.png
        gen_gap_vs_true_chi_sq_eps0p1_d2_covariance.png
        gen_gap_vs_true_chi_sq_eps0p1_d2_combined.png

    Layout per figure
    -----------------
    Panel 1:
        OLS vs Weighted OLS

    Panel 2:
        Linear SVR vs Weighted Linear SVR

    Panel 3:
        RBF SVR vs Weighted RBF SVR

    Colours
    --------
    blue = unweighted
    red  = weighted
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["epsilon"] == epsilon
    ]

    data = data[
        data["dimension"] == dimension
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if data.empty:

        print(
            f"[WARN] No data for "
            f"epsilon={epsilon}, "
            f"dimension={dimension}."
        )

        return

    # --------------------------------------------------------
    # 2. REMOVE INFINITE χ²
    # --------------------------------------------------------

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    if data.empty:

        print(
            f"[WARN] No finite χ² values."
        )

        return

    # --------------------------------------------------------
    # 3. AGGREGATE OVER SEEDS
    # --------------------------------------------------------

    agg = aggregate_results(
        data,
        groupby_cols=[
            "shift_type",
            "model_type",
            "chi_squared_divergence_theoretical",
        ],
        metric_cols=[
            "generalisation_gap",
        ],
    )

    # --------------------------------------------------------
    # 4. ONE FIGURE PER SHIFT TYPE
    # --------------------------------------------------------

    for shift in SHIFT_TYPES:

        shift_data = agg[
            agg["shift_type"] == shift
        ]

        if shift_data.empty:
            continue

        # ----------------------------------------------------
        # Determine which model pairs are present
        # ----------------------------------------------------

        available_pairs = []

        for model_uw, model_w, panel_title in MODEL_PAIRS:

            has_uw = (
                shift_data["model_type"] == model_uw
            ).any()

            has_w = (
                shift_data["model_type"] == model_w
            ).any()

            if has_uw or has_w:

                available_pairs.append(
                    (
                        model_uw,
                        model_w,
                        panel_title,
                    )
                )

        if len(available_pairs) == 0:
            continue

        fig, axes = plt.subplots(
            1,
            len(available_pairs),
            figsize=(5 * len(available_pairs), 5),
            sharey=False,
        )

        if len(available_pairs) == 1:
            axes = [axes]

        # ----------------------------------------------------
        # Plot each available model pair
        # ----------------------------------------------------

        for ax, (
            model_uw,
            model_w,
            panel_title,
        ) in zip(
            axes,
            available_pairs,
        ):

            subset_uw = shift_data[
                shift_data["model_type"] == model_uw
            ].sort_values(
                "chi_squared_divergence_theoretical"
            )

            if not subset_uw.empty:

                ax.plot(
                    subset_uw[
                        "chi_squared_divergence_theoretical"
                    ],
                    subset_uw[
                        "generalisation_gap_mean"
                    ],
                    color="blue",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label="unweighted",
                )

            subset_w = shift_data[
                shift_data["model_type"] == model_w
            ].sort_values(
                "chi_squared_divergence_theoretical"
            )

            if not subset_w.empty:

                ax.plot(
                    subset_w[
                        "chi_squared_divergence_theoretical"
                    ],
                    subset_w[
                        "generalisation_gap_mean"
                    ],
                    color="red",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label="weighted",
                )

            ax.set_title(
                panel_title,
                fontsize=11,
                fontweight="normal",
                pad=8,
            )

            ax.set_xlabel(
                "True χ² Divergence",
                fontsize=10,
            )

            ax.set_ylabel(
                "Generalisation Gap",
                fontsize=10,
            )

            ax.tick_params(
                labelsize=9,
            )

            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
            )

            ax.grid(
                True,
                linewidth=0.4,
                alpha=0.5,
            )

            ax.set_xscale("log")

            ax.legend(
                fontsize=8,
                frameon=True,
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        target_str = (
            f", target={target_mode}"
            if target_mode
            else ""
        )

        fig.suptitle(
            f"Generalisation Gap vs True χ² Divergence\n"
            f"{SHIFT_TITLES[shift]}"
            f" (d={dimension}, ε={epsilon}"
            f"{target_str})",
            fontsize=12,
            y=1.03,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        eps_str = str(epsilon).replace(".", "")

        target_tag = (
            f"_{target_mode}"
            if target_mode
            else ""
        )

        filename = (
            f"gap_vs_div"
            f"_eps{eps_str}"
            f"_d{dimension}"
            f"_{shift}"
            f"{target_tag}.png"
        )

        save_figure(
            filename,
            plot_dir,
        )
                
# ==================================================================
# 2. Var(w(x)) vs True χ² Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_true_w_var_vs_chi_sq_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    dimension=10,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot importance-weight variance against theoretical χ² divergence.

    Fixed:
        epsilon
        dimension

    One panel per shift type.

    Aggregates only across seeds since χ² is deterministic for a
    given (dimension, lambda, alpha, shift_type).
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["epsilon"] == epsilon
    ]

    data = data[
        data["dimension"] == dimension
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if data.empty:

        print(
            f"[WARN] No data for "
            f"epsilon={epsilon}, "
            f"dimension={dimension}."
        )

        return

    # --------------------------------------------------------
    # 2. DEDUPLICATE BY MODEL
    # --------------------------------------------------------

    data = data[
        data["model_type"] == "ols"
    ].copy()

    # --------------------------------------------------------
    # 3a. REMOVE α >= 2
    # --------------------------------------------------------

    n_alpha_ge_2 = (
        data["alpha"] >= 2.0
    ).sum()

    if n_alpha_ge_2 > 0:

        print(
            f"[INFO] Ignoring "
            f"{n_alpha_ge_2} rows with "
            f"alpha >= 2 "
            f"(theoretical χ² diverges)."
        )

    data = data[
        data["alpha"] < 2.0
    ].copy()

    # --------------------------------------------------------
    # 3b. REMOVE INFINITE χ² VALUES
    # --------------------------------------------------------

    n_total = len(data)

    n_inf = (
        ~np.isfinite(
            data["chi_squared_divergence_theoretical"]
        )
    ).sum()

    if n_inf > 0:

        print(
            f"[INFO] Found {n_inf}/{n_total} rows "
            f"with infinite theoretical χ² divergence."
        )

    data = data[
        np.isfinite(
            data["chi_squared_divergence_theoretical"]
        )
    ].copy()

    n_remaining = len(data)

    print(
        f"[INFO] Remaining finite rows: "
        f"{n_remaining}/{n_total}"
    )

    if n_remaining == 0:

        print(
            f"[WARN] All rows for "
            f"epsilon={epsilon}, "
            f"dimension={dimension} "
            f"have infinite χ² divergence."
        )

        return
    
    # --------------------------------------------------------
    # 3c. DIAGNOSTICS
    # --------------------------------------------------------

    for shift in SHIFT_TYPES:

        n_shift = len(
            data[
                data["shift_type"] == shift
            ]
        )

        print(
            f"[INFO] {shift}: "
            f"{n_shift} finite rows"
        )

    # --------------------------------------------------------
    # 4. AGGREGATE OVER SEEDS
    # --------------------------------------------------------

    agg = aggregate_results(
        data,
        groupby_cols=[
            "shift_type",
            "chi_squared_divergence_theoretical",
        ],
        metric_cols=[
            "weight_var_true",
        ],
    )

    # --------------------------------------------------------
    # 5. PLOT
    # --------------------------------------------------------

    available_shifts = [
        shift
        for shift in SHIFT_TYPES
        if not agg[
            agg["shift_type"] == shift
        ].empty
    ]

    if len(available_shifts) == 0:

        print("[WARN] No data to plot.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_shifts),
        figsize=(
            figsize[0] * len(available_shifts) / 3,
            figsize[1],
        ),
        sharey=False,
    )

    if len(available_shifts) == 1:
        axes = [axes]

    for ax, shift in zip(
        axes,
        available_shifts,
    ):

        subset = agg[
            agg["shift_type"] == shift
        ].sort_values(
            "chi_squared_divergence_theoretical"
        )

        colour = SHIFT_COLOURS[shift]

        ax.plot(
            subset[
                "chi_squared_divergence_theoretical"
            ],
            subset[
                "weight_var_true_mean"
            ],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.set_title(
            SHIFT_TITLES[shift],
            fontsize=11,
            fontweight="normal",
            pad=8,
        )

        ax.set_xlabel(
            "True χ² Divergence",
            fontsize=10,
        )

        ax.set_ylabel(
            "True Weight Variance",
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

    # --------------------------------------------------------
    # 6. TITLE AND SAVE
    # --------------------------------------------------------

    target_str = (
        f", target={target_mode}"
        if target_mode
        else ""
    )

    fig.suptitle(
        f"True Importance Weight Variance vs "
        f"True χ² Divergence "
        f"(d={dimension}, ε={epsilon}"
        f"{target_str})",
        fontsize=12,
        y=1.02,
    )

    eps_str = str(epsilon).replace(
        ".",
        "p",
    )

    target_tag = (
        f"_{target_mode}"
        if target_mode
        else ""
    )

    filename = (
        f"true_var_w_vs_true_chi_sq"
        f"_eps{eps_str}"
        f"_d{dimension}"
        f"{target_tag}.png"
    )

    save_figure(
        filename,
        plot_dir,
    )

# ==================================================================
# 3. Empirical ESS vs True χ² Divergence - Value of Epsilon Fixed
# ==================================================================

def plot_empirical_ess_vs_chi_squared_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    n_train,
    dimension=10,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot theoretical ESS vs theoretical χ² divergence.

    Uses

        ESS_true =
            n_train /
            (1 + epsilon^2 * chi_squared_divergence)

    Fixed:
        epsilon
        dimension

    One panel per shift type.

    Since both ESS_true and χ² are deterministic functions of
    (dimension, lambda, alpha, shift_type), we aggregate only
    over duplicate rows arising from seeds/models.
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["epsilon"] == epsilon
    ]

    data = data[
        data["dimension"] == dimension
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if data.empty:

        print(
            f"[WARN] No data for "
            f"epsilon={epsilon}, "
            f"dimension={dimension}. Skipping."
        )

        return

    # --------------------------------------------------------
    # 2. DEDUPLICATE BY MODEL
    # --------------------------------------------------------

    data = data[
        data["model_type"] == "ols"
    ].copy()

    # --------------------------------------------------------
    # 3. REMOVE INFINITE χ² VALUES
    # --------------------------------------------------------

    n_total = len(data)

    n_inf = (
        ~np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ).sum()

    if n_inf > 0:

        print(
            f"[INFO] Found {n_inf}/{n_total} rows "
            f"with infinite theoretical χ² divergence."
        )

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    n_remaining = len(data)

    print(
        f"[INFO] Remaining finite rows: "
        f"{n_remaining}/{n_total}"
    )

    if n_remaining == 0:

        print(
            f"[WARN] All rows for "
            f"epsilon={epsilon}, "
            f"dimension={dimension} "
            f"have infinite χ² divergence."
        )

        return

    # --------------------------------------------------------
    # 5. DIAGNOSTICS
    # --------------------------------------------------------

    for shift in SHIFT_TYPES:

        n_shift = len(
            data[
                data["shift_type"] == shift
            ]
        )

        print(
            f"[INFO] {shift}: "
            f"{n_shift} finite rows"
        )

    # --------------------------------------------------------
    # 6. AGGREGATE
    # --------------------------------------------------------

    agg = aggregate_results(
        data,
        groupby_cols=[
            "shift_type",
            "chi_squared_divergence_theoretical",
        ],
        metric_cols=[
            "ess_empirical",
        ],
    )

    # --------------------------------------------------------
    # 7. PLOT
    # --------------------------------------------------------

    available_shifts = [
        shift
        for shift in SHIFT_TYPES
        if not agg[agg["shift_type"] == shift].empty
    ]

    if len(available_shifts) == 0:

        print("[WARN] No finite data to plot.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_shifts),
        figsize=(5 * len(available_shifts), figsize[1]),
        sharey=False,
    )

    if len(available_shifts) == 1:
        axes = [axes]

    for ax, shift in zip(axes, available_shifts):

        subset = (
            agg[
                agg["shift_type"] == shift
            ]
            .sort_values(
                "chi_squared_divergence_theoretical"
            )
        )

        colour = SHIFT_COLOURS[shift]

        ax.plot(
            subset[
                "chi_squared_divergence_theoretical"
            ],
            subset[
                "ess_empirical_mean"
            ],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.ticklabel_format(
            axis="x",
            style="sci",
            scilimits=(0, 0),
        )

        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )

        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
                prune="both",
            )
        )

        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=6,
            )
        )

        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=30,
            ha="right",
        )

        ax.set_title(
            SHIFT_TITLES[shift],
            fontsize=11,
            fontweight="normal",
            pad=8,
        )

        ax.set_xlabel(
            "True χ² Divergence",
            fontsize=10,
        )

        ax.set_ylabel(
            "Empirical ESS",
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_xscale("log")
        ax.set_ylim(bottom=0)

    # --------------------------------------------------------
    # 8. TITLE
    # --------------------------------------------------------

    target_str = (
        f", target={target_mode}"
        if target_mode
        else ""
    )

    fig.suptitle(
        f"Empirical ESS vs "
        f"True χ² Divergence "
        f"(d={dimension}, "
        f"ε={epsilon}, "
        f"n={n_train}"
        f"{target_str})",
        fontsize=12,
        y=1.02,
    )

    # --------------------------------------------------------
    # 9. SAVE
    # --------------------------------------------------------

    eps_str = str(epsilon).replace(
        ".",
        "p",
    )

    target_tag = (
        f"_{target_mode}"
        if target_mode
        else ""
    )

    filename = (
        f"empirical_ess_vs_chi_sq"
        f"_eps{eps_str}"
        f"_d{dimension}"
        f"_n{n_train}"
        f"{target_tag}.png"
    )

    save_figure(
        filename,
        plot_dir,
    )

def plot_true_ess_vs_chi_squared_fixed_epsilon(
    df,
    plot_dir,
    epsilon,
    n_train,
    dimension=10,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot theoretical ESS vs theoretical χ² divergence.

    Uses

        ESS_true =
            n_train /
            (1 + epsilon^2 * chi_squared_divergence)

    Fixed:
        epsilon
        dimension

    One panel per shift type.

    Since both ESS_true and χ² are deterministic functions of
    (dimension, lambda, alpha, shift_type), we aggregate only
    over duplicate rows arising from seeds/models.
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["epsilon"] == epsilon
    ]

    data = data[
        data["dimension"] == dimension
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if data.empty:

        print(
            f"[WARN] No data for "
            f"epsilon={epsilon}, "
            f"dimension={dimension}. Skipping."
        )

        return

    # --------------------------------------------------------
    # 2. DEDUPLICATE BY MODEL
    # --------------------------------------------------------

    data = data[
        data["model_type"] == "ols"
    ].copy()

    # --------------------------------------------------------
    # 3. REMOVE INFINITE χ² VALUES
    # --------------------------------------------------------

    n_total = len(data)

    n_inf = (
        ~np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ).sum()

    if n_inf > 0:

        print(
            f"[INFO] Found {n_inf}/{n_total} rows "
            f"with infinite theoretical χ² divergence."
        )

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    n_remaining = len(data)

    print(
        f"[INFO] Remaining finite rows: "
        f"{n_remaining}/{n_total}"
    )

    if n_remaining == 0:

        print(
            f"[WARN] All rows for "
            f"epsilon={epsilon}, "
            f"dimension={dimension} "
            f"have infinite χ² divergence."
        )

        return

    # --------------------------------------------------------
    # 5. DIAGNOSTICS
    # --------------------------------------------------------

    for shift in SHIFT_TYPES:

        n_shift = len(
            data[
                data["shift_type"] == shift
            ]
        )

        print(
            f"[INFO] {shift}: "
            f"{n_shift} finite rows"
        )

    # --------------------------------------------------------
    # 6. AGGREGATE
    # --------------------------------------------------------

    agg = aggregate_results(
        data,
        groupby_cols=[
            "shift_type",
            "chi_squared_divergence_theoretical",
        ],
        metric_cols=[
            "ess_theoretical",
        ],
    )

    # --------------------------------------------------------
    # 7. PLOT
    # --------------------------------------------------------

    available_shifts = [
        shift
        for shift in SHIFT_TYPES
        if not agg[
            agg["shift_type"] == shift
        ].empty
    ]

    if len(available_shifts) == 0:

        print("[WARN] No shift types available for plotting.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_shifts),
        figsize=(
            figsize[0] * len(available_shifts) / 3,
            figsize[1],
        ),
        sharey=False,
    )

    if len(available_shifts) == 1:
        axes = [axes]

    for ax, shift in zip(axes, available_shifts):

        subset = agg[
            agg["shift_type"] == shift
        ].sort_values(
            "chi_squared_divergence_theoretical"
        )

        colour = SHIFT_COLOURS[shift]

        ax.plot(
            subset[
                "chi_squared_divergence_theoretical"
            ],
            subset[
                "ess_theoretical_mean"
            ],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.ticklabel_format(
            axis="x",
            style="sci",
            scilimits=(0, 0),
        )

        ax.ticklabel_format(
            axis="y",
            style="plain",
        )

        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
                prune="both",
            )
        )

        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=6,
            )
        )

        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=30,
            ha="right",
        )

        ax.set_title(
            SHIFT_TITLES[shift],
            fontsize=11,
            fontweight="normal",
            pad=8,
        )

        ax.set_xlabel(
            "True χ² Divergence",
            fontsize=10,
        )

        ax.set_ylabel(
            "True ESS",
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_ylim(bottom=0)

    # --------------------------------------------------------
    # 8. TITLE
    # --------------------------------------------------------

    target_str = (
        f", target={target_mode}"
        if target_mode
        else ""
    )

    fig.suptitle(
        f"True ESS vs "
        f"True χ² Divergence "
        f"(d={dimension}, "
        f"ε={epsilon}, "
        f"n={n_train}"
        f"{target_str})",
        fontsize=12,
        y=1.02,
    )

    # --------------------------------------------------------
    # 9. SAVE
    # --------------------------------------------------------

    eps_str = str(epsilon).replace(
        ".",
        "p",
    )

    target_tag = (
        f"_{target_mode}"
        if target_mode
        else ""
    )

    filename = (
        f"true_ess_vs_chi_sq"
        f"_eps{eps_str}"
        f"_d{dimension}"
        f"_n{n_train}"
        f"{target_tag}.png"
    )

    save_figure(
        filename,
        plot_dir,
    )

# ==================================================================
# 4. True χ² Divergence vs λ - Value of Epsilon Aggregated
# ==================================================================

def plot_chi_squared_vs_lambda(
    df,
    plot_dir,
    target_mode=None,
    figsize=(16, 5),
):
    """
    Plot theoretical χ² divergence vs λ.

    Fixes shift_type="mean" and plots

        χ²(P₁ || P₀) = exp(d λ²) - 1

    for each dimension.

    Only dimensions with available finite χ² values
    are shown.
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["shift_type"] == "mean"
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if data.empty:

        print(
            "[WARN] No data for shift_type=mean."
        )

        return

    # --------------------------------------------------------
    # 2. REMOVE NON-FINITE χ² VALUES
    # --------------------------------------------------------

    n_total = len(data)

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    n_remaining = len(data)

    if n_remaining < n_total:

        print(
            f"[INFO] Removed "
            f"{n_total - n_remaining} rows "
            f"with non-finite χ² divergence."
        )

    if data.empty:

        print(
            "[WARN] No finite theoretical χ² values."
        )

        return

    # --------------------------------------------------------
    # 3. KEEP UNIQUE THEORETICAL VALUES
    # --------------------------------------------------------

    agg = (
        data[
            [
                "dimension",
                "lambda",
                "chi_squared_divergence_theoretical",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["dimension", "lambda"]
        )
    )

    # --------------------------------------------------------
    # 4. DETERMINE AVAILABLE DIMENSIONS
    # --------------------------------------------------------

    available_dims = sorted(
        agg["dimension"].unique().tolist()
    )

    if len(available_dims) == 0:

        print(
            "[WARN] No dimensions contain finite χ² values."
        )

        return

    print(
        f"[INFO] Plotting dimensions: "
        f"{available_dims}"
    )

    # --------------------------------------------------------
    # 5. CREATE FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(available_dims),
        figsize=(
            max(5 * len(available_dims), 6),
            figsize[1],
        ),
        sharey=False,
    )

    if len(available_dims) == 1:
        axes = [axes]

    # --------------------------------------------------------
    # 6. PLOT EACH DIMENSION
    # --------------------------------------------------------

    for ax, d in zip(
        axes,
        available_dims,
    ):

        subset = agg[
            agg["dimension"] == d
        ].sort_values(
            "lambda"
        )

        if subset.empty:
            continue

        colour = DIM_COLOURS[d]

        ax.plot(
            subset["lambda"],
            subset[
                "chi_squared_divergence_theoretical"
            ],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # scientific notation for large χ² values

        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )

        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=6,
                prune="both",
            )
        )

        ax.set_title(
            DIM_LABELS[d],
            fontsize=11,
            fontweight="normal",
            pad=8,
        )

        ax.set_xlabel(
            "λ (mean shift magnitude)",
            fontsize=10,
        )

        ax.set_ylabel(
            "True χ² Divergence",
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # --------------------------------------------------------
    # 7. TITLE
    # --------------------------------------------------------

    target_tag = (
        f", target={target_mode}"
        if target_mode
        else ""
    )

    fig.suptitle(
        f"True χ² Divergence vs λ "
        f"(mean shift{target_tag})",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------

    target_suffix = (
        f"_{target_mode}"
        if target_mode
        else ""
    )

    filename = (
        f"true_chi_squared_vs_lambda"
        f"{target_suffix}.png"
    )

    save_figure(
        filename,
        plot_dir,
    )

# ==================================================================
# 5. True χ² Divergence vs α - Value of Epsilon Aggregated
# ==================================================================

def plot_chi_squared_vs_alpha(
    df,
    plot_dir,
    target_mode=None,
    alpha_max=None,
    figsize=(16, 5),
):
    """
    Plot theoretical χ² divergence vs α.

    Covariance shift only.

    χ²(P₁ || P₀) is finite only for α < 2.
    Infinite values are removed automatically.

    Only dimensions that actually contain finite values
    are plotted.
    """

    # --------------------------------------------------------
    # 1. FILTER
    # --------------------------------------------------------

    data = df.copy()

    data = data[
        data["shift_type"] == "covariance"
    ]

    if target_mode is not None:

        data = data[
            data["target_mode"] == target_mode
        ]

    if alpha_max is not None:

        data = data[
            data["alpha"] <= alpha_max
        ]

    if data.empty:

        print(
            "[WARN] No covariance-shift data "
            "after filtering."
        )

        return

    # --------------------------------------------------------
    # 2. REMOVE inf / NaN χ²
    # --------------------------------------------------------

    n_before = len(data)

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    n_removed = n_before - len(data)

    if n_removed > 0:

        print(
            f"[INFO] Removed {n_removed} rows "
            f"with infinite χ² divergence."
        )

    if data.empty:

        print(
            "[WARN] No finite χ² values remain."
        )

        return

    # --------------------------------------------------------
    # 3. KEEP UNIQUE THEORETICAL VALUES
    # --------------------------------------------------------

    agg = (
        data[
            [
                "dimension",
                "alpha",
                "chi_squared_divergence_theoretical",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["dimension", "alpha"]
        )
    )

    # --------------------------------------------------------
    # 4. DETERMINE DIMENSIONS PRESENT
    # --------------------------------------------------------

    available_dims = sorted(
        agg["dimension"].unique()
    )

    if len(available_dims) == 0:

        print(
            "[WARN] No dimensions remain "
            "after filtering."
        )

        return

    print(
        "[INFO] Plotting dimensions:",
        available_dims,
    )

    # --------------------------------------------------------
    # 5. CREATE ONLY REQUIRED PANELS
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(available_dims),
        figsize=(
            max(5 * len(available_dims), 6),
            figsize[1],
        ),
        sharey=False,
    )

    if len(available_dims) == 1:
        axes = [axes]

    # --------------------------------------------------------
    # 6. PLOT
    # --------------------------------------------------------

    for ax, d in zip(
        axes,
        available_dims,
    ):

        subset = agg[
            agg["dimension"] == d
        ].sort_values("alpha")

        if subset.empty:
            continue

        colour = DIM_COLOURS.get(
            d,
            "tab:blue",
        )

        ax.plot(
            subset["alpha"],
            subset[
                "chi_squared_divergence_theoretical"
            ],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # scientific notation
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )

        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=6,
            )
        )

        ax.set_title(
            DIM_LABELS.get(
                d,
                f"d={d}",
            ),
            fontsize=11,
            fontweight="normal",
            pad=8,
        )

        ax.set_xlabel(
            "α (covariance scale factor)",
            fontsize=10,
        )

        ax.set_ylabel(
            "True χ² Divergence",
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_xlim(
            left=subset["alpha"].min() - 0.05
        )

        ax.set_ylim(bottom=0)

    # --------------------------------------------------------
    # 7. TITLE
    # --------------------------------------------------------

    alpha_str = (
        f", α ≤ {alpha_max}"
        if alpha_max is not None
        else ""
    )

    target_str = (
        f", target={target_mode}"
        if target_mode
        else ""
    )

    fig.suptitle(
        f"True χ² Divergence vs α "
        f"(covariance shift"
        f"{alpha_str}"
        f"{target_str})",
        fontsize=12,
        y=1.02,
    )

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------

    alpha_suffix = (
        f"_alphamax{str(alpha_max).replace('.', 'p')}"
        if alpha_max is not None
        else ""
    )

    target_suffix = (
        f"_{target_mode}"
        if target_mode
        else ""
    )

    filename = (
        f"true_chi_squared_vs_alpha"
        f"{alpha_suffix}"
        f"{target_suffix}.png"
    )

    save_figure(
        filename,
        plot_dir,
    )

def export_mc_vs_true_chi_squared_summary(
    df,
    output_csv,
):
    """
    Export a summary table comparing Monte-Carlo and theoretical
    χ² divergence.

    Aggregates over:
        - random seeds
        - epsilon
        - lambda / alpha values

    Groups by:
        - shift type
        - dimensionality

    Output columns
    --------------
    shift_type
    dimension

    true_chi_squared_mean
    true_chi_squared_std

    mc_chi_squared_mean
    mc_chi_squared_std

    relative_error_mean
    relative_error_std
    """

    data = df.copy()

    # --------------------------------------------------------
    # keep one model only
    # --------------------------------------------------------

    data = data[
        data["model_type"] == "ols"
    ].copy()

    # --------------------------------------------------------
    # finite theoretical χ² only
    # --------------------------------------------------------

    data = data[
        np.isfinite(
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
    ].copy()

    if data.empty:

        print(
            "[WARN] No finite theoretical χ² values."
        )

        return

    # --------------------------------------------------------
    # relative error (%)
    # --------------------------------------------------------

    data["relative_error"] = (
        100
        * np.abs(
            data[
                "chi_squared_divergence"
            ]
            -
            data[
                "chi_squared_divergence_theoretical"
            ]
        )
        /
        np.maximum(
            data[
                "chi_squared_divergence_theoretical"
            ],
            1e-12,
        )
    )

    # --------------------------------------------------------
    # aggregate
    # --------------------------------------------------------

    summary = (
        data
        .groupby(
            [
                "shift_type",
                "dimension",
            ]
        )
        .agg(
            true_chi_squared_mean=(
                "chi_squared_divergence_theoretical",
                "mean",
            ),
            true_chi_squared_std=(
                "chi_squared_divergence_theoretical",
                "std",
            ),
            mc_chi_squared_mean=(
                "chi_squared_divergence",
                "mean",
            ),
            mc_chi_squared_std=(
                "chi_squared_divergence",
                "std",
            ),
            relative_error_mean=(
                "relative_error",
                "mean",
            ),
            relative_error_std=(
                "relative_error",
                "std",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # ordering
    # --------------------------------------------------------

    summary["shift_type"] = pd.Categorical(
        summary["shift_type"],
        categories=SHIFT_TYPES,
        ordered=True,
    )

    summary = summary.sort_values(
        [
            "shift_type",
            "dimension",
        ]
    )

    # --------------------------------------------------------
    # export
    # --------------------------------------------------------

    summary.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"[INFO] Saved summary to {output_csv}"
    )

    return summary