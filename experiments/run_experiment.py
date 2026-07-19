import math
import torch
import numpy as np

from sklearn.model_selection import train_test_split

from utils import generate_targets
from utils import (
    compute_importance_weights,
    compute_empirical_weight_variance,
    compute_effective_sample_size_theoretical,
    compute_effective_sample_size_empirical,
    calculate_weight_variance
)

from utils import (
    make_P1_combined,
    make_P1_cov_shift,
    make_P1_mean_shift,
    make_training_distribution,
    sample_distribution,
    sample_contaminated_distribution,
    estimate_gaussian_parameters
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
    monte_carlo_chi_squared_divergence,
    chi_squared_divergence_theoretical_synthetic,
    chi_squared_divergence_theoretical_external
)

from utils import (
    mse,
    rmse
)

from utils import load_dataset

def run_single_synthetic_experiment(
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
    Sigma = torch.eye(d)

    lambda_vec = lambda_scalar * torch.ones(d)

    # ============================================================
    # 3. CONSTRUCT DISTRIBUTIONS
    # ============================================================

    P0 = make_training_distribution(mu, Sigma)

    if shift_type == "mean":

        P1 = make_P1_mean_shift(
            mu=mu,
            covariance_matrix=Sigma,
            lambda_vec=lambda_vec
        )

    elif shift_type == "covariance":

        P1 = make_P1_cov_shift(
            mu=mu,
            covariance_matrix=Sigma,
            alpha=alpha
        )

    elif shift_type == "combined":

        P1 = make_P1_combined(
            mu=mu,
            covariance_matrix=Sigma,
            alpha=alpha,
            lambda_vec=lambda_vec
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

    empirical_weight_variance = compute_empirical_weight_variance(weights)

    # ============================================================
    # 7. CHI-SQUARED DIVERGENCE (MONTE-CARLO ESTIMATION)
    # ============================================================

    chi2_divergence, chi2_weight_mean, chi2_weight_variance = monte_carlo_chi_squared_divergence(P0, P1, n_test)

    weight_variance_mc = calculate_weight_variance(epsilon, chi2_divergence)
    
    # ============================================================
    # 8. ESTIMATED EFFECTIVE SAMPLE SIZE
    # ============================================================

    ess_mc = compute_effective_sample_size_theoretical(chi2_divergence, n_train, epsilon)

    ess_emp = compute_effective_sample_size_empirical(weights)

    # ============================================================
    # 9. CHI-SQUARED DIVERGENCE (THEORETICAL)
    # ============================================================

    chi2_divergence_theoretical_value = (
        chi_squared_divergence_theoretical_synthetic(
            d=d,
            lambda_scalar=lambda_scalar,
            alpha=alpha,
            shift_type=shift_type,
        )
    )

    if math.isfinite(
        chi2_divergence_theoretical_value
    ):

        chi2_relative_error = abs(
            chi2_divergence
            -
            chi2_divergence_theoretical_value
        ) / max(
            abs(
                chi2_divergence_theoretical_value
            ),
            1e-12
        )

    else:
        chi2_relative_error = np.inf

    ess_true = compute_effective_sample_size_theoretical(chi2_divergence_theoretical_value, n_train, epsilon)

    weight_variance_true = calculate_weight_variance(epsilon, chi2_divergence_theoretical_value)

    # ============================================================
    # 9. CONVERT TO NUMPY
    # ============================================================

    X_train_np = X_train.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()

    X_test_np = X_test.detach().cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy()

    weights_np = weights.detach().cpu().numpy()

    # ============================================================
    # 10. TRAIN MODEL
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
    # 11. PREDICTIONS
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
    # 12. LOSSES
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
    # 13. RESULTS
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

        "chi_squared_divergence":
            chi2_divergence,

        "chi_squared_divergence_theoretical":
            chi2_divergence_theoretical_value,

        "chi_squared_relative_error":
            chi2_relative_error,

        "chi2_weight_mean":
            chi2_weight_mean,

        "chi2_weight_variance":
            chi2_weight_variance,
        
        # --------------------------------------------------------
        # importance-weight diagnostics
        # --------------------------------------------------------

        "ess_theoretical": ess_true,
        
        "ess_empirical": ess_emp,

        "ess_monte_carlo": ess_mc,

        "weight_var_empirical": empirical_weight_variance,

        "weight_var_monte_carlo": weight_variance_mc,

        "weight_var_true": weight_variance_true,
        
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

def run_single_external_experiment(
    dataset,
    lambda_scalar,
    alpha,
    epsilon,
    sigma,
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
    # 2. LOAD DATASET
    # ============================================================

    X, y = load_dataset(
        source=dataset["source"],
        dataset=dataset
    )

    # ============================================================
    # 3. TRAIN-TEST SPLITTING
    # ============================================================

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    n_train = X_train_real.shape[0]
    n_test = X_test_real.shape[0]

    d = X_train_real.shape[1]
    beta = torch.ones(d)

    # ============================================================
    # 4. DEFINE SHIFT PARAMETERS
    # ============================================================

    mu_hat, Sigma_hat = estimate_gaussian_parameters(
        torch.from_numpy(X_train_real)
    )

    lambda_vec = lambda_scalar * torch.ones(d)

    # ============================================================
    # 5. CONSTRUCT DISTRIBUTIONS (TRAINING + SHIFTED)
    # ============================================================

    P0 = make_training_distribution(mu_hat, Sigma_hat)

    if shift_type == "mean":

        P1 = make_P1_mean_shift(
            mu=mu_hat,
            covariance_matrix=Sigma_hat,
            lambda_vec=lambda_vec
        )

    elif shift_type == "covariance":

        P1 = make_P1_cov_shift(
            mu=mu_hat,
            covariance_matrix=Sigma_hat,
            alpha=alpha
        )

    elif shift_type == "combined":

        P1 = make_P1_combined(
            mu=mu_hat,
            covariance_matrix=Sigma_hat,
            alpha=alpha,
            lambda_vec=lambda_vec
        )

    else:
        raise ValueError(
            f"Unknown shift_type: {shift_type}"
        )

    # ============================================================
    # 6. SAMPLE TRAINING DATA
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
    # 7. SAMPLE CONTAMINATED TEST DATA
    # ============================================================

    X_test = sample_contaminated_distribution(
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
    # 8. IMPORTANCE WEIGHTS
    # ============================================================

    weights = compute_importance_weights(
        X=X_train,
        P0=P0,
        P1=P1,
        epsilon=epsilon
    )

    empirical_weight_variance = compute_empirical_weight_variance(weights)

    # ============================================================
    # 9. CHI-SQUARED DIVERGENCE (MONTE-CARLO ESTIMATION)
    # ============================================================

    chi2_divergence, chi2_weight_mean, chi2_weight_variance = monte_carlo_chi_squared_divergence(
        P0, 
        P1, 
        n_test
    )

    weight_variance_mc = calculate_weight_variance(
        epsilon, 
        chi2_divergence
    )
    
    # ============================================================
    # 10. ESTIMATED EFFECTIVE SAMPLE SIZE
    # ============================================================

    ess_mc = compute_effective_sample_size_theoretical(
        chi2_divergence,
        n_train,
        epsilon
    )

    ess_emp = compute_effective_sample_size_empirical(
        weights
    )

    # ============================================================
    # 11. CHI-SQUARED DIVERGENCE (THEORETICAL)
    # ============================================================

    chi2_divergence_theoretical_value = (
        chi_squared_divergence_theoretical_external(
            d=d,
            lambda_vec=lambda_vec,
            covariance_matrix=Sigma_hat,
            alpha = alpha,
            shift_type=shift_type
        )
    )

    if math.isfinite(
        chi2_divergence_theoretical_value
    ):

        chi2_relative_error = abs(
            chi2_divergence
            -
            chi2_divergence_theoretical_value
        ) / max(
            abs(
                chi2_divergence_theoretical_value
            ),
            1e-12
        )

    else:
        chi2_relative_error = np.inf

    ess_true = compute_effective_sample_size_theoretical(
        chi2_divergence_theoretical_value,
        n_train,
        epsilon
    )

    weight_variance_true = calculate_weight_variance(
        epsilon,
        chi2_divergence_theoretical_value
    )

    # ============================================================
    # 12. CONVERT TO NUMPY
    # ============================================================

    X_train_np = X_train.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()

    X_test_np = X_test.detach().cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy()

    weights_np = weights.detach().cpu().numpy()

    # ============================================================
    # 13. TRAIN MODEL
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
    # 14. PREDICTIONS
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
    # 15. LOSSES
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
    # 16. RESULTS
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

        "chi_squared_divergence":
            chi2_divergence,

        "chi_squared_divergence_theoretical":
            chi2_divergence_theoretical_value,

        "chi_squared_relative_error":
            chi2_relative_error,

        "chi2_weight_mean":
            chi2_weight_mean,

        "chi2_weight_variance":
            chi2_weight_variance,
        
        # --------------------------------------------------------
        # importance-weight diagnostics
        # --------------------------------------------------------

        "ess_theoretical": ess_true,
        
        "ess_empirical": ess_emp,

        "ess_monte_carlo": ess_mc,

        "weight_var_empirical": empirical_weight_variance,

        "weight_var_monte_carlo": weight_variance_mc,

        "weight_var_true": weight_variance_true,
        
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
