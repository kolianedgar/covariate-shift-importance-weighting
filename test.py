from utils import *
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# GLOBAL PLOTTING CONFIG
# ============================================================

PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# HELPER: SAVE FIGURE
# ============================================================

def save_figure(filename):

    path = os.path.join(PLOT_DIR, filename)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")


# ============================================================
# HELPER: AGGREGATE OVER SEEDS
# ============================================================

def aggregate_results(
    df,
    groupby_cols,
    metric_cols
):
    """
    Aggregate results over seeds.

    Returns:
        mean/std dataframe.
    """

    grouped = df.groupby(groupby_cols)

    mean_df = grouped[metric_cols].mean()

    std_df = grouped[metric_cols].std()

    mean_df.columns = [
        f"{c}_mean"
        for c in mean_df.columns
    ]

    std_df.columns = [
        f"{c}_std"
        for c in std_df.columns
    ]

    result = pd.concat(
        [mean_df, std_df],
        axis=1
    ).reset_index()

    return result


# ============================================================
# 1. TEST MSE VS KL DIVERGENCE
# ============================================================

def plot_test_mse_vs_kl(
    df,
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
        "test_mse_vs_kl.png"
    )


# ============================================================
# 2. ESS VS KL DIVERGENCE
# ============================================================

def plot_ess_vs_kl(df):

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
        "ess_vs_kl.png"
    )


# ============================================================
# 3. WEIGHT VARIANCE VS KL
# ============================================================

def plot_weight_variance_vs_kl(df):

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
        "weight_variance_vs_kl.png"
    )


# ============================================================
# 4. TEST MSE VS EPSILON
# ============================================================

def plot_test_mse_vs_epsilon(
    df,
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
        f"Test MSE vs Epsilon ({model_type})"
    )

    save_figure(
        f"test_mse_vs_epsilon_{model_type}.png"
    )


# ============================================================
# 5. GENERALISATION GAP VS KL
# ============================================================

def plot_generalisation_gap_vs_kl(df):

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
        "generalisation_gap_vs_kl.png"
    )


# ============================================================
# 6. TEST MSE VS DIMENSION
# ============================================================

def plot_test_mse_vs_dimension(df):

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
        "test_mse_vs_dimension.png"
    )


# ============================================================
# 7. HISTOGRAM OF IMPORTANCE WEIGHTS
# ============================================================

def plot_weight_histogram(
    weights,
    filename="weight_histogram.png"
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

    save_figure(filename)


# ============================================================
# 8. ESS HEATMAP
# ============================================================

def plot_ess_heatmap(
    df,
    model_type="weighted_ols"
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
        aggfunc=np.mean
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
        f"ess_heatmap_{model_type}.png"
    )


# ============================================================
# 9. MASTER PLOTTING FUNCTION
# ============================================================

def generate_all_plots(results_df):

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_test_mse_vs_kl(results_df)

    plot_ess_vs_kl(results_df)

    plot_weight_variance_vs_kl(results_df)

    plot_generalisation_gap_vs_kl(results_df)

    plot_test_mse_vs_dimension(results_df)

    for model in results_df["model_type"].unique():

        plot_test_mse_vs_epsilon(
            results_df,
            model_type=model
        )

        plot_ess_heatmap(
            results_df,
            model_type=model
        )

    print("=" * 60)
    print("ALL PLOTS GENERATED")
    print("=" * 60)

def run_experiment_grid(
    config,
    save_path="covariate_shift_results.csv",
    preview_rows=10,
):
    """
    Run full experiment grid and save results.

    Parameters
    ----------
    config : dict
        Experiment grid configuration.

    save_path : str
        CSV output filepath.

    preview_rows : int
        Number of dataframe rows to preview.

    Returns
    -------
    results_df : pd.DataFrame
        Structured experiment dataframe.
    """

    # ============================================================
    # 1. INITIALISE RESULTS CONTAINER
    # ============================================================

    results = []

    # ============================================================
    # 2. CREATE GRID ITERATOR
    # ============================================================

    experiment_iterator = itertools.product(

        config["dimensions"],
        config["lambda_grid"],
        config["alpha_grid"],
        config["epsilon_grid"],
        config["model_types"],
        config["target_modes"],
        config["shift_types"],
        config["seeds"],
    )

    total_experiments = (

        len(config["dimensions"])

        * len(config["lambda_grid"])

        * len(config["alpha_grid"])

        * len(config["epsilon_grid"])

        * len(config["model_types"])

        * len(config["target_modes"])

        * len(config["shift_types"])

        * len(config["seeds"])
    )

    print("=" * 80)
    print(f"TOTAL EXPERIMENTS: {total_experiments}")
    print("=" * 80)

    # ============================================================
    # 3. MAIN LOOP
    # ============================================================

    for idx, (
        d,
        lambda_scalar,
        alpha,
        epsilon,
        model_type,
        target_mode,
        shift_type,
        seed,
    ) in enumerate(experiment_iterator):

        print(
            f"[{idx+1}/{total_experiments}] "
            f"d={d}, "
            f"lambda={lambda_scalar}, "
            f"alpha={alpha}, "
            f"epsilon={epsilon}, "
            f"model={model_type}, "
            f"target={target_mode}, "
            f"shift={shift_type}, "
            f"seed={seed}"
        )

        beta = torch.ones(d)

        try:

            result = run_single_experiment(

                d=d,

                lambda_scalar=lambda_scalar,

                alpha=alpha,

                epsilon=epsilon,

                n_train=config["n_train"],

                n_test=config["n_test"],

                sigma=config["sigma"],

                beta=beta,

                model_type=model_type,

                target_mode=target_mode,

                shift_type=shift_type,

                seed=seed,
            )

            results.append(result)

        except Exception as e:

            print(f"FAILED: {e}")

    # ============================================================
    # 4. CONVERT TO DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(results)

    # ============================================================
    # 5. SAVE TO CSV
    # ============================================================

    results_df.to_csv(
        save_path,
        index=False
    )

    print("\n" + "=" * 80)
    print(f"RESULTS SAVED TO: {save_path}")
    print("=" * 80)

    # ============================================================
    # 6. SHOW DATAFRAME SUMMARY
    # ============================================================

    print("\nDATAFRAME SHAPE:")
    print(results_df.shape)

    print("\nCOLUMNS:")
    print(results_df.columns.tolist())

    print("\nFIRST ROWS:")
    print(results_df.head(preview_rows))

    return results_df

# ============================================================
# EXPERIMENT GRID
# ============================================================

GRID_CONFIG = {

    # --------------------------------------------------------
    # dimensionality
    # --------------------------------------------------------

    "dimensions": [
        2,
        10,
        50,
    ],

    # --------------------------------------------------------
    # mean-shift magnitude
    # --------------------------------------------------------

    "lambda_grid": [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
    ],

    # --------------------------------------------------------
    # covariance scaling
    # --------------------------------------------------------

    "alpha_grid": [
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
    ],

    # --------------------------------------------------------
    # contamination levels
    # --------------------------------------------------------

    "epsilon_grid": [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ],

    # --------------------------------------------------------
    # models
    # --------------------------------------------------------

    "model_types": [

        # unweighted
        "ols",
        "linear_svr",
        "rbf_svr",

        # weighted
        "weighted_ols",
        "weighted_linear_svr",
        "weighted_rbf_svr",
    ],

    # --------------------------------------------------------
    # target structures
    # --------------------------------------------------------

    "target_modes": [
        "linear",
        "nonlinear",
    ],

    # --------------------------------------------------------
    # shift categories
    # --------------------------------------------------------

    "shift_types": [
        "mean",
        "covariance",
        "combined",
    ],

    # --------------------------------------------------------
    # random seeds
    # --------------------------------------------------------

    "seeds": [
        0,
        1,
        2,
        3,
        4,
    ],

    # --------------------------------------------------------
    # fixed experimental parameters
    # --------------------------------------------------------

    "n_train": 1000,
    "n_test": 1000,

    "sigma": 0.1,
}

results_df = run_experiment_grid(
    config=GRID_CONFIG,
    save_path="covariate_shift_results.csv",
    preview_rows=5,
)

results_df.to_csv(
    "covariate_shift_results.csv",
    index=False
)
print("[SAVED] covariate_shift_results.csv")

generate_all_plots(results_df)

print("[DONE] Full experiment pipeline completed.")