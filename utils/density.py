import torch
import math

def log_density(dist, X):

    """
    Compute the log-density of a distribution at given observations.

    Parameters
    ----------
    dist :
        Probability distribution used to evaluate the log-density.

    X : np.ndarray
        Input observations of shape (n_samples, d).

    Returns
    -------
    log_density_dist : torch.Tensor
        Log-density values for each observation.
    """

    log_density_dist = dist.log_prob(X)
    
    return log_density_dist

def log_mixture_density(P0, P1, X, epsilon):

    """
    Compute the log-density of a contaminated mixture distribution.

    Parameters
    ----------
    P0 :
        Source probability distribution.

    P1 :
        Shifted probability distribution.

    X : np.ndarray
        Input observations of shape (n_samples, d).

    epsilon : float
        Contamination proportion of the shifted distribution.

    Returns
    -------
    log_density_epsilon : torch.Tensor
        Log-density values of the mixture distribution for each observation.
    """
    
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