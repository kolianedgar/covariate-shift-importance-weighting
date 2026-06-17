import torch
from .density import *

def compute_importance_weights(P0, P1, X, epsilon):

    log_p0 = log_density(P0, X)
    log_p1 = log_density(P1, X)

    weights = (1-epsilon) + epsilon * torch.exp(log_p1 - log_p0)
    return weights

def compute_density_ratio(P0, P1, X):
    log_p0 = log_density(P0, X)
    log_p1 = log_density(P1, X)

    density_ratios = torch.exp(log_p1 - log_p0)
    return density_ratios

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