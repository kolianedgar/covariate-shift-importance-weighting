import matplotlib.pyplot as plt
from .helpers import save_figure
import numpy as np
from analysis import aggregate_results

# ============================================================
# 1. TEST MSE VS KL DIVERGENCE
# ============================================================

def plot_test_mse_vs_kl(
    df,
    plot_dir,
    shift_type=None,
    target_mode=None,
    dimension=None,
):
    """
    Plot:
        Test MSE vs KL divergence
    """

    data = df.copy()

    if shift_type is not None:
        data = data[
            data["shift_type"] == shift_type
        ]

    if target_mode is not None:
        data = data[
            data["target_mode"] == target_mode
        ]

    if dimension is not None:
        data = data[
            data["dimension"] == dimension
        ]

    agg = aggregate_results(

        data,

        groupby_cols=[
            "model_type",
            "kl_divergence",
        ],

        metric_cols=[
            "test_mse",
        ]
    )

    plt.figure(figsize=(8, 6))

    for model in agg["model_type"].unique():

        subset = agg[
            agg["model_type"] == model
        ].sort_values("kl_divergence")

        plt.plot(
            subset["kl_divergence"],
            subset["test_mse_mean"],
            label=model
        )

    plt.xlabel("KL Divergence")

    plt.ylabel("Test MSE")

    plt.title(
        "Test MSE vs KL Divergence"
    )

    plt.legend()

    save_figure(
        "test_mse_vs_kl.png",
        plot_dir
    )


# ============================================================
# 2. ESS VS KL DIVERGENCE
# ============================================================

def plot_ess_vs_kl(df, plot_dir):

    agg = aggregate_results(

        df,

        groupby_cols=[
            "model_type",
            "kl_divergence",
        ],

        metric_cols=[
            "ess",
        ]
    )

    plt.figure(figsize=(8, 6))

    for model in agg["model_type"].unique():

        subset = agg[
            agg["model_type"] == model
        ].sort_values("kl_divergence")

        plt.plot(
            subset["kl_divergence"],
            subset["ess_mean"],
            label=model
        )

    plt.xlabel("KL Divergence")

    plt.ylabel("Effective Sample Size")

    plt.title(
        "ESS vs KL Divergence"
    )

    plt.legend()

    save_figure(
        "ess_vs_kl.png",
        plot_dir,
    )


# ============================================================
# 3. WEIGHT VARIANCE VS KL
# ============================================================

def plot_weight_variance_vs_kl(df, plot_dir):

    agg = aggregate_results(

        df,

        groupby_cols=[
            "model_type",
            "kl_divergence",
        ],

        metric_cols=[
            "weight_variance",
        ]
    )

    plt.figure(figsize=(8, 6))

    for model in agg["model_type"].unique():

        subset = agg[
            agg["model_type"] == model
        ].sort_values("kl_divergence")

        plt.plot(
            subset["kl_divergence"],
            subset["weight_variance_mean"],
            label=model
        )

    plt.xlabel("KL Divergence")

    plt.ylabel("Weight Variance")

    plt.title(
        "Weight Variance vs KL Divergence"
    )

    plt.legend()

    save_figure(
        "weight_variance_vs_kl.png",
        plot_dir
    )


# ============================================================
# 4. TEST MSE VS EPSILON
# ============================================================

def plot_test_mse_vs_epsilon(
    df,
    plot_dir,
    model_type="ols",
):
    """
    Robustness against contamination.
    """

    data = df[
        df["model_type"] == model_type
    ]

    agg = aggregate_results(

        data,

        groupby_cols=[
            "epsilon",
        ],

        metric_cols=[
            "test_mse",
        ]
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        agg["epsilon"],
        agg["test_mse_mean"]
    )

    plt.xlabel("Epsilon")

    plt.ylabel("Test MSE")

    plt.title(
        f"Test MSE vs Epsilon ({model_type})",
    )

    save_figure(
        f"test_mse_vs_epsilon_{model_type}.png",
        plot_dir,
    )


# ============================================================
# 5. GENERALISATION GAP VS KL
# ============================================================

def plot_generalisation_gap_vs_kl(df, plot_dir):

    agg = aggregate_results(

        df,

        groupby_cols=[
            "model_type",
            "kl_divergence",
        ],

        metric_cols=[
            "generalisation_gap",
        ]
    )

    plt.figure(figsize=(8, 6))

    for model in agg["model_type"].unique():

        subset = agg[
            agg["model_type"] == model
        ].sort_values("kl_divergence")

        plt.plot(
            subset["kl_divergence"],
            subset["generalisation_gap_mean"],
            label=model
        )

    plt.xlabel("KL Divergence")

    plt.ylabel("Generalisation Gap")

    plt.title(
        "Generalisation Gap vs KL Divergence"
    )

    plt.legend()

    save_figure(
        "generalisation_gap_vs_kl.png",
        plot_dir,
    )


# ============================================================
# 6. TEST MSE VS DIMENSION
# ============================================================

def plot_test_mse_vs_dimension(df, plot_dir):

    agg = aggregate_results(

        df,

        groupby_cols=[
            "model_type",
            "dimension",
        ],

        metric_cols=[
            "test_mse",
        ]
    )

    plt.figure(figsize=(8, 6))

    for model in agg["model_type"].unique():

        subset = agg[
            agg["model_type"] == model
        ].sort_values("dimension")

        plt.plot(
            subset["dimension"],
            subset["test_mse_mean"],
            label=model
        )

    plt.xlabel("Dimension")

    plt.ylabel("Test MSE")

    plt.title(
        "Test MSE vs Dimension"
    )

    plt.legend()

    save_figure(
        "test_mse_vs_dimension.png",
        plot_dir,
    )


# ============================================================
# 7. HISTOGRAM OF IMPORTANCE WEIGHTS
# ============================================================

def plot_weight_histogram(
    weights,
    plot_dir,
    filename="weight_histogram.png",
):

    plt.figure(figsize=(8, 6))

    plt.hist(
        weights,
        bins=50
    )

    plt.xlabel("Importance Weight")

    plt.ylabel("Frequency")

    plt.title(
        "Importance Weight Distribution"
    )

    save_figure(filename, plot_dir)


# ============================================================
# 8. ESS HEATMAP
# ============================================================

def plot_ess_heatmap(
    df,
    plot_dir,
    model_type="weighted_ols",
):
    """
    ESS as function of:
        lambda x epsilon
    """

    data = df[
        df["model_type"] == model_type
    ]

    pivot = data.pivot_table(
        index="lambda",
        columns="epsilon",
        values="ess",
        aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        pivot,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(label="ESS")

    plt.xticks(
        range(len(pivot.columns)),
        labels=pivot.columns
    )

    plt.yticks(
        range(len(pivot.index)),
        labels=pivot.index
    )

    plt.xlabel("Epsilon")

    plt.ylabel("Lambda")

    plt.title(
        f"ESS Heatmap ({model_type})"
    )

    save_figure(
        f"ess_heatmap_{model_type}.png",
        plot_dir
    )


# ============================================================
# 9. MASTER PLOTTING FUNCTION
# ============================================================

def generate_all_plots(results_df, plot_dir):

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_test_mse_vs_kl(results_df, plot_dir)

    plot_ess_vs_kl(results_df, plot_dir)

    plot_weight_variance_vs_kl(results_df, plot_dir)

    plot_generalisation_gap_vs_kl(results_df, plot_dir)

    plot_test_mse_vs_dimension(results_df, plot_dir)

    for model in results_df["model_type"].unique():

        plot_test_mse_vs_epsilon(
            results_df,
            plot_dir,
            model_type=model,
        )

        plot_ess_heatmap(
            results_df,
            plot_dir,
            model_type=model,
        )

    print("=" * 60)
    print("ALL PLOTS GENERATED")
    print("=" * 60)
