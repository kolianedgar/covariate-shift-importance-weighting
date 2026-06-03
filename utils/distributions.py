from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
import torch

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