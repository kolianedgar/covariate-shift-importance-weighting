import torch
import math
from utils import sample_distribution, compute_density_ratio

def monte_carlo_chi_squared_divergence(
    P0,
    P1,
    n_test
):

    """
    Estimate chi-squared divergence and density-ratio statistics using Monte Carlo sampling.

    Parameters
    ----------
    P0 :
        Source probability distribution.

    P1 :
        Shifted probability distribution.

    n_test : int
        Number of samples drawn from the source distribution.

    Returns
    -------
    chi2_divergence : float
        Monte Carlo estimate of the chi-squared divergence.

    chi2_weight_mean : float
        Mean of the estimated density ratios.

    chi2_weight_variance : float
        Variance of the estimated density ratios.
    """

    sample_P0 = sample_distribution(
        dist=P0,
        n=n_test
    )

    density_ratios = compute_density_ratio(
        P0,
        P1,
        sample_P0
    )

    chi2_divergence = max(
        0.0,
        (torch.mean(density_ratios ** 2) - 1.0).item()
    )

    chi2_weight_mean = torch.mean(density_ratios).item()

    chi2_weight_variance = torch.var(
        density_ratios,
        unbiased=False
    ).item()

    return chi2_divergence, chi2_weight_mean, chi2_weight_variance

def chi_squared_divergence_theoretical_synthetic(
    d,
    lambda_scalar,
    alpha,
    shift_type,
):

    """
    Compute the theoretical chi-squared divergence for a synthetic Gaussian shift.

    Parameters
    ----------
    d : int
        Dimensionality of the distributions.

    lambda_scalar : float
        Scalar amount of mean shift applied to each feature.

    alpha : float
        Covariance inflation parameter.

    shift_type : str
        Type of distributional shift: ``"mean"``, ``"covariance"``,
        or ``"combined"``.

    Returns
    -------
    float
        Theoretical chi-squared divergence between the source and shifted
        distributions.
    """

    if shift_type == "mean":
        try:
            return math.exp(
                d * lambda_scalar**2
            ) - 1.0
        
        except OverflowError:
            return float('inf')
    
    elif shift_type == "covariance":

        if alpha >= 2.0:
            return float('inf')
        
        try:
            return (
                alpha * (2.0 - alpha)
            ) ** (-d / 2.0) - 1.0
        
        except OverflowError:
            return float('inf')

    elif shift_type == "combined":

        if alpha >= 2.0:
            return float("inf")

        try:
            return (
                math.exp(
                    d * lambda_scalar**2 / (2.0 - alpha)
                )
                *
                (
                    alpha * (2.0 - alpha)
                ) ** (-d / 2.0)
                - 1.0
            )
        
        except OverflowError:
            return float('inf')

    else:
        raise ValueError(
            f"Unknown shift_type: {shift_type}"
        )
    
def chi_squared_divergence_theoretical_external(
    d, 
    lambda_vec, 
    covariance_matrix, 
    alpha, 
    shift_type
):

    """
    Compute the theoretical chi-squared divergence for an external dataset.

    Parameters
    ----------
    d : int
        Dimensionality of the distributions.

    lambda_vec : torch.Tensor
        Mean-shift vector.

    covariance_matrix : torch.Tensor
        Covariance matrix of the source distribution.

    alpha : float
        Covariance inflation parameter.

    shift_type : str
        Type of distributional shift: ``"mean"``, ``"covariance"``,
        or ``"combined"``.

    Returns
    -------
    float
        Theoretical chi-squared divergence between the source and shifted
        distributions.
    """
    
    cov_matr_inv = torch.linalg.inv(covariance_matrix)
    quad_form = lambda_vec @ cov_matr_inv @ lambda_vec


    if shift_type == "mean":
        try:
            return torch.exp(quad_form).item() - 1

        except OverflowError:
            return float('inf')
        
    elif shift_type == "covariance":

        if alpha >=2:
            return float("inf")

        try:
            return (
                alpha * (2.0 - alpha)
            ) ** (-d / 2.0) - 1.0
        
        except OverflowError:
            return float('inf')
    
    elif shift_type == "combined":
        
        if alpha >= 2:
            return float("inf")

        try:
            return (
                (alpha*(2-alpha))**(-d / 2)
                * torch.exp(quad_form / (2 - alpha)).item()
                - 1
            )

        except OverflowError:
            return float('inf')
        
    else:
        raise ValueError (
            f"Unknown shift_type: {shift_type}"
        )