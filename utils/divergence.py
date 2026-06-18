import torch
import math
from utils import sample_distribution, compute_density_ratio

def monte_carlo_chi_squared_divergence(P0, P1, n_test):
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

def chi_squared_divergence_theoretical(
    d,
    lambda_scalar,
    alpha,
    shift_type,
):
    """
    Theoretical χ²(P1 || P0).

    P0 = N(0, I)

    mean shift:
        P1 = N(lambda, I)

    covariance shift:
        P1 = N(0, alpha I)

    combined:
        P1 = N(lambda, alpha I)
    """

    if shift_type == "mean":

        return math.exp(
            d * lambda_scalar**2
        ) - 1.0

    elif shift_type == "covariance":

        if alpha >= 2.0:
            return float("inf")

        return (
            alpha * (2.0 - alpha)
        ) ** (-d / 2.0) - 1.0

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

            return float("inf")

    else:

        raise ValueError(
            f"Unknown shift_type: {shift_type}"
        )