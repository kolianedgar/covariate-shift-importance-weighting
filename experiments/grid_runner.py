import itertools
from experiments.run_experiment import (
    run_single_synthetic_experiment, 
    run_single_external_experiment
)
import torch
import pandas as pd

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

            result = run_single_synthetic_experiment(

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

    for idx, (
        dataset,
        lambda_scalar,
        alpha,
        epsilon,
        model_type,
        target_mode,
        shift_type,
        seed,
    ) in enumerate(experiment_iterator):

        print(
            f"[{idx + 1}/{total_experiments}] "
            f"dataset={dataset}, "
            f"lambda={lambda_scalar}, "
            f"alpha={alpha}, "
            f"epsilon={epsilon}, "
            f"model={model_type}, "
            f"target={target_mode}, "
            f"shift={shift_type}, "
            f"seed={seed}"
        )

        try:

            result = run_single_external_experiment(

                dataset=dataset,

                lambda_scalar=lambda_scalar,

                alpha=alpha,

                epsilon=epsilon,

                sigma=config["sigma"],

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