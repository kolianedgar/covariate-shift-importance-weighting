import pandas as pd

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