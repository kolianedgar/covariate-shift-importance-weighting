import torch
from .density import *

def compute_importance_weights(P0, P1, X, epsilon):

    """
    Compute importance weights for a contaminated target distribution.

    Parameters
    ----------
    P0 :
        Source distribution.

    P1 :
        Shifted distribution.

    X : np.ndarray
        Input features of shape (n_samples, d).

    epsilon :
        Contamination factor determining the contribution of the shifted
        distribution.

    Returns
    -------
    weights : torch.Tensor
        Importance weights of shape (n_samples,).
    """

    log_p0 = log_density(P0, X)
    log_p1 = log_density(P1, X)

    weights = (1-epsilon) + epsilon * torch.exp(log_p1 - log_p0)
    return weights

def compute_density_ratio(P0, P1, X):

    """
    Compute density ratios between a shifted and source distribution.

    Parameters
    ----------
    P0 :
        Source distribution.

    P1 :
        Shifted distribution.

    X : np.ndarray
        Input features of shape (n_samples, d).

    Returns
    -------
    density_ratios : torch.Tensor
        Density ratios of shape (n_samples,).
    """

    log_p0 = log_density(P0, X)
    log_p1 = log_density(P1, X)

    density_ratios = torch.exp(log_p1 - log_p0)
    return density_ratios

def compute_effective_sample_size_theoretical(chi2_divergence, n_train, epsilon):
    """
    Compute the effective sample size,
    using the derived formula showing its relation to variance of importance weights.

    Parameters
    ----------
    chi2_divergence :
        Chi-squared divergence between the source and shifted distributions.

    n_train :
        Number of training samples.

    epsilon :
        Contamination factor determining the contribution of the shifted
        distribution.

    Returns
    -------
    estimated_effective_sample_size :
        Theoretical effective sample size.
    """

    estimated_effective_sample_size = n_train / (1+calculate_weight_variance(epsilon, chi2_divergence))

    return estimated_effective_sample_size

def compute_effective_sample_size_empirical(weights):

    """
    Compute the empirical effective sample size from importance weights.

    Parameters
    ----------
    weights : torch.Tensor
        Importance weights of shape (n_samples,).

    Returns
    -------
    ess_empirical :
        Empirical effective sample size.
    """

    sum_of_weights = torch.sum(weights)
    sum_of_squared_weights = torch.sum(weights**2)
    ess_empirical = (sum_of_weights ** 2) / sum_of_squared_weights

    return ess_empirical.item()

def compute_empirical_weight_variance(weights):

    """
    Compute the empirical variance of importance weights.

    Parameters
    ----------
    weights : torch.Tensor
        Importance weights of shape (n_samples,).

    Returns
    -------
    :
        Empirical variance of the importance weights.
    """
     
    return torch.var(weights).item()

def calculate_weight_variance(epsilon, chi2_divergence):

    """
    Compute the variance of importance weights
    using its derived definition under contamination model.

    Parameters
    ----------
    epsilon :
        Contamination factor determining the contribution of the shifted
        distribution.

    chi2_divergence :
        Chi-squared divergence between the source and shifted distributions.

    Returns
    -------
    :
        Theoretical variance of the importance weights.
    """

    if epsilon == 0.0 or chi2_divergence == 0.0:
        return 0.0

    elif chi2_divergence == float('inf') and epsilon != 0:
        return float('inf')

    else:
        return (epsilon**2)*chi2_divergence