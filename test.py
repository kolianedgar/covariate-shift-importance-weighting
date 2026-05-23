from utils import *
import torch

def smoke_test_experiment():
    """
    Minimal end-to-end sanity test for the
    covariate-shift experiment pipeline.
    """

    print("=" * 60)
    print("RUNNING SMOKE TEST")
    print("=" * 60)

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    d = 5

    lambda_scalar = 1.0

    alpha = 1.5

    epsilon = 0.2

    n_train = 1000

    n_test = 1000

    sigma = 0.1

    beta = torch.ones(d)

    model_type = "weighted_rbf_svr"

    target_mode = "linear"

    shift_type = "combined"

    seed = 42

    # ============================================================
    # 2. RUN EXPERIMENT
    # ============================================================

    results = run_single_experiment(
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

    # ============================================================
    # 3. PRINT RESULTS
    # ============================================================

    print("\nEXPERIMENT RESULTS\n")

    for key, value in results.items():

        if isinstance(value, float):

            print(f"{key:30s}: {value:.6f}")

        else:

            print(f"{key:30s}: {value}")

    # ============================================================
    # 4. BASIC SANITY CHECKS
    # ============================================================

    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    assert results["ess"] > 0, \
        "ESS must be positive."

    assert results["train_mse"] >= 0, \
        "Train MSE must be nonnegative."

    assert results["test_mse"] >= 0, \
        "Test MSE must be nonnegative."

    assert results["weight_variance"] >= 0, \
        "Weight variance must be nonnegative."

    assert not torch.isnan(
        torch.tensor(results["kl_divergence"])
    ), "KL divergence is NaN."

    print("All sanity checks passed.")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return results

smoke_test_experiment()