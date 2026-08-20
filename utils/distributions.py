from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
import torch

torch.set_default_dtype(torch.float64)

def make_training_distribution(mu, covariance_matrix):

    """
    Create the source multivariate normal distribution.

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the distribution.

    covariance_matrix : torch.Tensor
        Covariance matrix of the distribution.

    Returns
    -------
    MultivariateNormal
        Source multivariate normal distribution.
    """

    return MultivariateNormal(
        loc=mu,
        covariance_matrix=covariance_matrix
    )

def make_P1_mean_shift(mu, covariance_matrix, lambda_vec):

    """
    Create a distribution with a mean shift relative to the source distribution.

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the source distribution.

    covariance_matrix : torch.Tensor
        Covariance matrix of the source distribution.

    lambda_vec : torch.Tensor
        Mean-shift vector.

    Returns
    -------
    MultivariateNormal
        Multivariate normal distribution with the shifted mean.
    """

    mu_1 = mu + lambda_vec

    return MultivariateNormal(
        loc=mu_1,
        covariance_matrix=covariance_matrix
    )

def make_P1_cov_shift(mu, covariance_matrix, alpha):

    """
    Create a distribution with covariance inflation relative to the source distribution.

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the source distribution.

    covariance_matrix : torch.Tensor
        Covariance matrix of the source distribution.

    alpha : float
        Covariance inflation parameter.

    Returns
    -------
    MultivariateNormal
        Multivariate normal distribution with the inflated covariance matrix.
    """

    Sigma_1 = alpha * covariance_matrix

    return MultivariateNormal(
        loc=mu,
        covariance_matrix=Sigma_1
    )

def make_P1_combined(mu, covariance_matrix, alpha, lambda_vec):

    """
    Create a distribution with both mean and covariance shifts.

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the source distribution.

    covariance_matrix : torch.Tensor
        Covariance matrix of the source distribution.

    alpha : float
        Covariance inflation parameter.

    lambda_vec : torch.Tensor
        Mean-shift vector.

    Returns
    -------
    MultivariateNormal
        Multivariate normal distribution with shifted mean and covariance.
    """

    mu_1 = mu + lambda_vec

    Sigma_1 = alpha * covariance_matrix

    return MultivariateNormal(
        loc=mu_1,
        covariance_matrix=Sigma_1
    )

def sample_distribution(dist, n):

    """
    Generate samples from a probability distribution.

    Parameters
    ----------
    dist :
        Probability distribution to sample from.

    n : int
        Number of samples to generate.

    Returns
    -------
    X : torch.Tensor
        Samples of shape (n, d).
    """

    X = dist.sample((n, ))

    return X

def sample_contaminated_distribution(P0, P1, epsilon, n):

    """
    Generate samples from a contaminated mixture of two distributions.

    Parameters
    ----------
    P0 :
        Source probability distribution.

    P1 :
        Shifted probability distribution.

    epsilon : float
        Proportion of samples drawn from the shifted distribution.

    n : int
        Number of samples to generate.

    Returns
    -------
    sample_P_epsilon : torch.Tensor
        Samples from the contaminated distribution of shape (n, d).
    """

    mask = Bernoulli(torch.tensor([epsilon]))

    sample_P0 = sample_distribution(P0, n)
    sample_P1 = sample_distribution(P1, n)

    sample_mask = mask.sample((n, )).bool()
    
    sample_P_epsilon = torch.where(
        sample_mask,
        sample_P1,
        sample_P0
    )
    
    return sample_P_epsilon

def estimate_gaussian_parameters(X):

    """
    Estimate the mean vector and covariance matrix from observations.

    Parameters
    ----------
    X : torch.Tensor
        Input observations of shape (n_samples, d).

    Returns
    -------
    mu_hat : torch.Tensor
        Estimated mean vector.

    Sigma_hat : torch.Tensor
        Estimated covariance matrix.
    """
    
    mu_hat = X.mean(dim=0)
    X_centered = X - mu_hat
    n_samples = X.shape[0]

    eps = 1e-6

    Sigma_hat = (
        X_centered.T @ X_centered
    ) / (n_samples - 1)

    Sigma_hat += + eps * torch.eye(
        Sigma_hat.shape[0],
        dtype = Sigma_hat.dtype,
        device = Sigma_hat.device
    )

    return mu_hat, Sigma_hat