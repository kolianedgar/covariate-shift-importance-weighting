from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
import torch
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

torch.set_default_dtype(torch.float64)

def make_training_distribution(mu, d):
    covariance_matrix = torch.eye(d)

    training_dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
    return training_dist

def make_P1_mean_shift(mu, lambda_vec, d):

    covariance_matrix = torch.eye(d)

    mu_1 = mu + lambda_vec

    P1_mean_shift_dist = MultivariateNormal(loc=mu_1, covariance_matrix=covariance_matrix)
    return P1_mean_shift_dist

def make_P1_cov_shift(mu, alpha, d):

    covariance_matrix = alpha * torch.eye(d)

    P1_cov_shift_dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
    return P1_cov_shift_dist

def make_P1_combined(mu, alpha, lambda_vec, d):

    mu_p1 = mu + lambda_vec

    covariance_matrix = alpha * torch.eye(d)

    P1_combined_dist = MultivariateNormal(loc=mu_p1, covariance_matrix=covariance_matrix)
    return P1_combined_dist

def sample_distribution(dist, n):

    X = dist.sample((n, ))

    return X

def sample_contaminated_distribution(P0, P1, epsilon, n):

    mask = Bernoulli(torch.tensor([epsilon]))

    sample_P0 = sample_distribution(P0, n)
    sample_P1 = sample_distribution(P1, n)

    sample_mask = mask.sample((n, )).bool()
    
    sample_P_epsilon = torch.where(
        sample_mask,
        sample_P1,
        sample_P0
    )
    
    return sample_P_epsilon, sample_mask

def log_density(dist, X):
    log_density_dist = dist.log_prob(X)
    
    return log_density_dist

def log_mixture_density(P0, P1, X, epsilon):
    
    log_density_P0 = log_density(P0, X)
    log_density_P1 = log_density(P1, X)

    log_density_epsilon = torch.logsumexp(
        torch.stack([
            math.log(1 - epsilon) + log_density_P0,
            math.log(epsilon) + log_density_P1
        ]),
        dim=0
    )

    return log_density_epsilon

def compute_importance_weights(P0, P1, X, epsilon):

    log_p0 = log_density(P0, X)
    log_p1 = log_density(P1, X)

    weights = (1-epsilon) + epsilon * torch.exp(log_p1 - log_p0)

    return weights

def effective_sample_size(weights):
    weights = torch.as_tensor(weights)

    ess = (weights.sum() ** 2) / (weights.pow(2).sum())

    return ess.item()

def weight_statistics(weights):

    return {
        "mean": torch.mean(weights).item(),
        "variance": torch.var(weights).item(),
        "maximum": torch.max(weights).item(),
        "minimum": torch.min(weights).item(),
        "lower_quartile": torch.quantile(weights, 0.25).item(),
        "median": torch.quantile(weights, 0.5).item(),
        "upper_quartile": torch.quantile(weights, 0.75).item()
    }

import torch


def generate_targets(
    X,
    beta,
    sigma,
    mode="linear"
):
    """
    Generate regression targets.

    Model:
        Y = f(X) + tau

    where:
        tau ~ N(0, sigma^2)

    Parameters
    ----------
    X : torch.Tensor
        Input data of shape (n_samples, d)

    beta : torch.Tensor
        Coefficient vector of shape (d,)

    sigma : float
        Noise standard deviation

    mode : str
        One of:
            - "linear"
            - "nonlinear"

    Returns
    -------
    y : torch.Tensor
        Target vector of shape (n_samples,)
    """

    # ============================================================
    # 1. COMPUTE SIGNAL
    # ============================================================

    linear_response = X @ beta

    if mode == "linear":

        signal = linear_response

    elif mode == "nonlinear":

        # smooth nonlinear target
        signal = torch.sin(linear_response)

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    # ============================================================
    # 2. ADD GAUSSIAN NOISE
    # ============================================================

    noise = sigma * torch.randn(
        X.shape[0],
        dtype=X.dtype,
        device=X.device
    )

    y = signal + noise

    return y


def train_linear_regression(
    X_train,
    y_train
):
    """
    Train Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted OLS model
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_linear_regression(
    X_train,
    y_train,
    weights
):
    """
    Train importance-weighted linear regression.

    Minimises:

        sum_i w_i (y_i - x_i^T beta)^2

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted weighted linear regression model
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def train_linear_svr(
    X_train,
    y_train,
    C=1.0,
    epsilon=0.1
):
    """
    Train Linear Support Vector Regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    Returns
    -------
    model : sklearn.svm.SVR
        Trained linear SVR model
    """

    model = SVR(
        kernel="linear",
        C=C,
        epsilon=epsilon
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_linear_svr(
    X_train,
    y_train,
    weights,
    C=1.0,
    epsilon=0.1,
    max_iter=-1
):
    """
    Train importance-weighted Linear SVR.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained weighted Linear SVR model
    """

    model = SVR(
        kernel="linear",
        C=C,
        epsilon=epsilon,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def train_rbf_svr(
    X_train,
    y_train,
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    max_iter=-1
):
    """
    Train Gaussian RBF-kernel Support Vector Regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    gamma : str or float
        RBF kernel coefficient

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained RBF-SVR model
    """

    model = SVR(
        kernel="rbf",
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_rbf_svr(
    X_train,
    y_train,
    weights,
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    max_iter=-1
):
    """
    Train importance-weighted Gaussian RBF-kernel SVR.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    gamma : str or float
        RBF kernel bandwidth parameter

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained weighted RBF-SVR model
    """

    model = SVR(
        kernel="rbf",
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def predict_model(model, X):
    """
    Generate predictions using a fitted model.

    Parameters
    ----------
    model :
        Trained sklearn model.

    X : np.ndarray
        Input features of shape (n_samples, d)

    Returns
    -------
    y_pred : np.ndarray
        Predicted targets of shape (n_samples,)
    """

    y_pred = model.predict(X)

    return y_pred

def mse(y_true, y_pred):
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets

    y_pred : np.ndarray
        Predicted targets

    Returns
    -------
    float
        Mean Squared Error
    """

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets

    y_pred : np.ndarray
        Predicted targets

    Returns
    -------
    float
        Root Mean Squared Error
    """

    return np.sqrt(mse(y_true, y_pred))

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