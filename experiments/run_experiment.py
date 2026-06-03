import math
import torch
import numpy as np
from utils import generate_targets
from utils import (
    compute_importance_weights,
    weight_statistics,
    effective_sample_size
)

from utils import (
    make_P1_combined,
    make_P1_cov_shift,
    make_P1_mean_shift,
    make_training_distribution,
    sample_distribution,
    sample_contaminated_distribution
)

from utils import (
    train_linear_regression,
    train_linear_svr, train_rbf_svr,
    train_weighted_linear_regression,
    train_weighted_linear_svr,
    train_weighted_rbf_svr,
    predict_model
)

from utils import (
    mse,
    rmse
)

def run_single_experiment(
    d,
    lambda_scalar,
    alpha,
    epsilon,
    n_train,
    n_test,
    sigma,
    beta,
    model_type="ols",
    target_mode="linear",
    shift_type="mean",
    seed=42,
):
    """
    Run one complete controlled covariate-shift experiment.

    Parameters
    ----------
    d : int
        Feature-space dimensionality.

    lambda_scalar : float
        Mean-shift magnitude.

    alpha : float
        Covariance scaling factor.

    epsilon : float
        Contamination probability.

    n_train : int
        Number of training samples.

    n_test : int
        Number of test samples.

    sigma : float
        Noise standard deviation.

    beta : torch.Tensor
        Regression coefficient vector.

    model_type : str
        One of:
            - "ols"
            - "weighted_ols"
            - "linear_svr"
            - "weighted_linear_svr"
            - "rbf_svr"
            - "weighted_rbf_svr"

    target_mode : str
        One of:
            - "linear"
            - "nonlinear"

    shift_type : str
        One of:
            - "mean"
            - "covariance"
            - "combined"

    seed : int
        Random seed.

    Returns
    -------
    dict
        Experiment results.
    """

    # ============================================================
    # 1. REPRODUCIBILITY
    # ============================================================

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============================================================
    # 2. DEFINE SHIFT PARAMETERS
    # ============================================================

    mu = torch.zeros(d)

    lambda_vec = lambda_scalar * torch.ones(d)

    # ============================================================
    # 3. CONSTRUCT DISTRIBUTIONS
    # ============================================================

    P0 = make_training_distribution(mu, d)

    if shift_type == "mean":

        P1 = make_P1_mean_shift(
            mu=mu,
            lambda_vec=lambda_vec,
            d=d
        )

    elif shift_type == "covariance":

        P1 = make_P1_cov_shift(
            mu=mu,
            alpha=alpha,
            d=d
        )

    elif shift_type == "combined":

        P1 = make_P1_combined(
            mu=mu,
            lambda_vec=lambda_vec,
            alpha=alpha,
            d=d
        )

    else:
        raise ValueError(
            f"Unknown shift_type: {shift_type}"
        )

    # ============================================================
    # 4. SAMPLE TRAINING DATA
    # ============================================================

    X_train = sample_distribution(
        dist=P0,
        n=n_train
    )

    y_train = generate_targets(
        X=X_train,
        beta=beta,
        sigma=sigma,
        mode=target_mode
    )

    # ============================================================
    # 5. SAMPLE CONTAMINATED TEST DATA
    # ============================================================

    X_test, contamination_mask = sample_contaminated_distribution(
        P0=P0,
        P1=P1,
        epsilon=epsilon,
        n=n_test
    )

    y_test = generate_targets(
        X=X_test,
        beta=beta,
        sigma=sigma,
        mode=target_mode
    )

    # ============================================================
    # 6. IMPORTANCE WEIGHTS
    # ============================================================

    weights = compute_importance_weights(
        X=X_train,
        P0=P0,
        P1=P1,
        epsilon=epsilon
    )

    ess = effective_sample_size(weights)

    weight_stats = weight_statistics(weights)

    # ============================================================
    # 7. KL DIVERGENCE (MONTE-CARLO ESTIMATION)
    # ============================================================

    sample_P0 = sample_distribution(
        dist=P0,
        n=n_test
    )

    log_p0 = P0.log_prob(sample_P0)

    log_p1 = P1.log_prob(sample_P0)

    if epsilon == 0.0:
        log_mix = log_p0

    elif epsilon == 1.0:
        log_mix = log_p1

    else:
        log_mix = torch.logsumexp(
            torch.stack([
                math.log(1.0 - epsilon) + log_p0,
                math.log(epsilon) + log_p1
            ]),
            dim=0
        )
        
    kl_p0_to_test = torch.mean(
        log_p0 - log_mix
    ).item()

    # ============================================================
    # 8. CONVERT TO NUMPY
    # ============================================================

    X_train_np = X_train.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()

    X_test_np = X_test.detach().cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy()

    weights_np = weights.detach().cpu().numpy()

    # ============================================================
    # 9. TRAIN MODEL
    # ============================================================

    if model_type == "ols":

        model = train_linear_regression(
            X_train_np,
            y_train_np
        )

    elif model_type == "weighted_ols":

        model = train_weighted_linear_regression(
            X_train_np,
            y_train_np,
            weights_np
        )

    elif model_type == "linear_svr":

        model = train_linear_svr(
            X_train_np,
            y_train_np
        )

    elif model_type == "weighted_linear_svr":

        model = train_weighted_linear_svr(
            X_train_np,
            y_train_np,
            weights_np
        )

    elif model_type == "rbf_svr":

        model = train_rbf_svr(
            X_train_np,
            y_train_np
        )

    elif model_type == "weighted_rbf_svr":

        model = train_weighted_rbf_svr(
            X_train_np,
            y_train_np,
            weights_np
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}"
        )

    # ============================================================
    # 10. PREDICTIONS
    # ============================================================

    y_train_pred = predict_model(
        model,
        X_train_np
    )

    y_test_pred = predict_model(
        model,
        X_test_np
    )

    # ============================================================
    # 11. LOSSES
    # ============================================================

    train_mse = mse(
        y_train_np,
        y_train_pred
    )

    test_mse = mse(
        y_test_np,
        y_test_pred
    )

    train_rmse = rmse(
        y_train_np,
        y_train_pred
    )

    test_rmse = rmse(
        y_test_np,
        y_test_pred
    )

    generalisation_gap = (
        test_mse - train_mse
    )

    # ============================================================
    # 12. RESULTS
    # ============================================================

    results = {

        # --------------------------------------------------------
        # experiment configuration
        # --------------------------------------------------------

        "dimension": d,
        "lambda": lambda_scalar,
        "alpha": alpha,
        "epsilon": epsilon,
        "shift_type": shift_type,
        "model_type": model_type,
        "target_mode": target_mode,
        "n_train": n_train,
        "n_test": n_test,
        "sigma": sigma,
        "seed": seed,

        # --------------------------------------------------------
        # divergence metrics
        # --------------------------------------------------------

        "kl_divergence": kl_p0_to_test,

        # --------------------------------------------------------
        # importance-weight diagnostics
        # --------------------------------------------------------

        "ess": ess,

        "weight_mean":
            weight_stats["mean"],

        "weight_variance":
            weight_stats["variance"],

        "weight_max":
            weight_stats["maximum"],

        "weight_min":
            weight_stats["minimum"],

        "weight_q1":
            weight_stats["lower_quartile"],

        "weight_median":
            weight_stats["median"],

        "weight_q3":
            weight_stats["upper_quartile"],

        # --------------------------------------------------------
        # predictive metrics
        # --------------------------------------------------------

        "train_mse": train_mse,
        "test_mse": test_mse,

        "train_rmse": train_rmse,
        "test_rmse": test_rmse,

        "generalisation_gap":
            generalisation_gap,
    }

    return results