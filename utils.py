from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
import torch
import math

def make_gaussian(mean, covariance):
    return MultivariateNormal(
            loc=mean,
            covariance_matrix=covariance
        )

def make_mask(epsilon):
    return Bernoulli(torch.Tensor([epsilon]))

def sample_dist(P, n_samples):
    return P.sample((n_samples, ))

def sample_mixture(P0, P1, P_mask, n_samples):
    return {
        "sample_p0": P0.sample((n_samples, )), 
        "sample_p1": P1.sample((n_samples, )), 
        "mask": P_mask.sample((n_samples, )).bool()
    }

def make_mixture_dist(sample_mask, sample_P0, sample_P1):
    return torch.where(
        sample_mask,
        sample_P1,
        sample_P0
    )

def log_mixture_density(x, P0, P1, epsilon):
    log_p0 = P0.log_prob(x)
    log_p1 = P1.log_prob(x)
    epsilon = max(epsilon, 1e-12)

    log_mix = torch.logsumexp(
                torch.stack([
                    math.log(1-epsilon) + log_p0,
                    math.log(epsilon) + log_p1
                ]),
                dim=0
            )
    
    return log_mix

def estimate_kl(P_ref, P0, P1, epsilon, n_samples):
    x = P_ref.sample((n_samples, ))

    log_pref = P_ref.log_prob(x)
    log_mix = log_mixture_density(x, P0, P1, epsilon)

    return torch.mean(log_pref - log_mix)