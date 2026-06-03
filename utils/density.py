import torch
import math

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