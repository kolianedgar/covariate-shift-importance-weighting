from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
import torch

torch.set_default_dtype(torch.float64)

def make_training_distribution(mu, covariance_matrix):

    return MultivariateNormal(
        loc=mu,
        covariance_matrix=covariance_matrix
    )

def make_P1_mean_shift(mu, covariance_matrix, lambda_vec):

    mu_1 = mu + lambda_vec

    return MultivariateNormal(
        loc=mu_1,
        covariance_matrix=covariance_matrix
    )

def make_P1_cov_shift(mu, covariance_matrix, alpha):

    Sigma_1 = alpha * covariance_matrix

    return MultivariateNormal(
        loc=mu,
        covariance_matrix=Sigma_1
    )

def make_P1_combined(mu, covariance_matrix, alpha, lambda_vec):

    mu_1 = mu + lambda_vec

    Sigma_1 = alpha * covariance_matrix

    return MultivariateNormal(
        loc=mu_1,
        covariance_matrix=Sigma_1
    )

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
    
    return sample_P_epsilon

def estimate_gaussian_parameters(X):

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