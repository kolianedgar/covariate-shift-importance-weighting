import itertools
import traceback
from experiments.run_experiment import (
    run_single_synthetic_experiment, 
    run_single_external_experiment
)
from joblib import Parallel, delayed
import torch
import pandas as pd

def run_synthetic_worker(
    d,
    lambda_scalar,
    alpha,
    epsilon,
    model_type,
    target_mode,
    shift_type,
    seed,
    n_train,
    n_test,
    sigma,
):
    """
    Worker executed by a separate process for one synthetic experiment.
    """

    beta = torch.ones(d)

    try:

        return run_single_synthetic_experiment(
            d=d,
            lambda_scalar=lambda_scalar,
            alpha=alpha,
            epsilon=epsilon,
            n_train=n_train,
            n_test=n_test,
            sigma=sigma,
            beta=beta,
            model_type=model_type,
            target_mode=target_mode,
            shift_type=shift_type,
            seed=seed,
        )

    except Exception as e:

        print(
            f"FAILED: "
            f"d={d}, "
            f"lambda={lambda_scalar}, "
            f"alpha={alpha}, "
            f"epsilon={epsilon}, "
            f"model={model_type}, "
            f"target={target_mode}, "
            f"shift={shift_type}, "
            f"seed={seed}\n"
            f"{e}"
        )

        return None

def run_external_worker(
    dataset,
    lambda_scalar,
    alpha,
    epsilon,
    model_type,
    target_mode,
    shift_type,
    seed,
    sigma,
):
    """
    Worker executed by a separate process.
    """

    try:

        return run_single_external_experiment(

            dataset=dataset,
            lambda_scalar=lambda_scalar,
            alpha=alpha,
            epsilon=epsilon,
            sigma=sigma,
            model_type=model_type,
            target_mode=target_mode,
            shift_type=shift_type,
            seed=seed,
        )

    except Exception as e:

        if dataset.get("data_id") is not None:
            dataset_identifier = f"data_id={dataset['data_id']}"
        else:
            dataset_identifier = (
                f"{dataset['dataset_name']}"
                + (
                    f" (version={dataset['version']})"
                    if dataset.get("version") is not None
                    else ""
                )
            )

        print(
            f"FAILED: dataset={dataset_identifier}, "
            f"lambda={lambda_scalar}, "
            f"alpha={alpha}, "
            f"epsilon={epsilon}, "
            f"model={model_type}, "
            f"target={target_mode}, "
            f"shift={shift_type}, "
            f"seed={seed}\n"
            f"{e}"
        )

        traceback.print_exc()
        return None
    
def run_synthetic_experiment_grid(
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

    results = Parallel(
        n_jobs=-1,
        verbose=10,
    )(
        delayed(run_synthetic_worker)(
            d,
            lambda_scalar,
            alpha,
            epsilon,
            model_type,
            target_mode,
            shift_type,
            seed,
            config["n_train"],
            config["n_test"],
            config["sigma"],
        )
        for (
            d,
            lambda_scalar,
            alpha,
            epsilon,
            model_type,
            target_mode,
            shift_type,
            seed,
        ) in experiment_iterator
    )

    results = [result for result in results if result is not None]

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

def run_external_experiment_grid(
    config,
    save_path="external_covariate_shift_results.csv",
    preview_rows=10,
):
    """
    Run full experiment grid on external datasets and save results.

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

        config["datasets"],
        config["lambda_grid"],
        config["alpha_grid"],
        config["epsilon_grid"],
        config["model_types"],
        config["target_modes"],
        config["shift_types"],
        config["seeds"],
    )

    total_experiments = (

        len(config["datasets"])
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

    results = Parallel(

        n_jobs=-1,
        verbose=10,
        batch_size="auto",

    )(

        delayed(run_external_worker)(

            dataset,
            lambda_scalar,
            alpha,
            epsilon,
            model_type,
            target_mode,
            shift_type,
            seed,
            config["sigma"],

        )

        for (
            dataset,
            lambda_scalar,
            alpha,
            epsilon,
            model_type,
            target_mode,
            shift_type,
            seed,
        ) in experiment_iterator

    )

    results = [result for result in results if result is not None]
    
    # ============================================================
    # 4. CONVERT TO DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(results)

    # ============================================================
    # 5. SAVE TO CSV
    # ============================================================

    results_df.to_csv(
        save_path,
        index=False,
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