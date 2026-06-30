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

def compute_effective_sample_size_theoretical(chi2_divergence, n_train, epsilon):
    estimated_effective_sample_size = n_train / (1+calculate_weight_variance(epsilon, chi2_divergence))

    return estimated_effective_sample_size

def compute_effective_sample_size_empirical(weights):
    sum_of_weights = torch.sum(weights)
    sum_of_squared_weights = torch.sum(weights**2)
    ess_empirical = (sum_of_weights ** 2) / sum_of_squared_weights

    return ess_empirical.item()

def compute_empirical_weight_variance(weights):
    return torch.var(weights).item()

def calculate_weight_variance(epsilon, chi2_divergence):
    return (epsilon**2)*chi2_divergence