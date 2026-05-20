from utils import *
import torch

dim = 5
n_samples = 100
epsilon = 0.3
mean_p0 = torch.zeros(dim)
cov_matrix = torch.eye(dim)
mean_lambda = torch.ones(dim)

P_0 = make_gaussian(mean_p0, cov_matrix)
P_lambda = make_gaussian(mean_lambda, cov_matrix)
mask = make_mask(epsilon)

sample_p0 = sample_dist(P_0, n_samples)
sample_p_lambda = sample_dist(P_lambda, n_samples)
samples = sample_mixture(P_0, P_lambda, mask, n_samples)
sample_mask = samples["mask"]

sample_p_epsilon = make_mixture_samples(sample_mask, sample_p0, sample_p_lambda)

log_mix = log_mixture_density(sample_p0, P_0, P_lambda, epsilon)

kl_est = estimate_kl(P_0, P_0, P_lambda, epsilon, n_samples)
print(f"Estimated KL-Divergence: {kl_est}")