"""
Shared statistical utilities.

This module contains functions that were previously duplicated across
task_a_mcmc.py and task_b_mcmc.py, including:
- Gelman-Rubin convergence diagnostic
- Truncated normal sampling
- Covariate generation
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.special import ndtr
from typing import List


def sample_truncated_normal(mean: float, std: float,
                           lower: float, upper: float) -> float:
    """Sample from truncated normal distribution.

    Args:
        mean: Mean of the underlying normal distribution
        std: Standard deviation of the underlying normal distribution
        lower: Lower truncation bound
        upper: Upper truncation bound

    Returns:
        A single sample from the truncated normal distribution
    """
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)


def selection_prob_at_least_one_exceeds_cutoff(
    cutoff: float,
    bid_mu: float,
    bid_sigma: float,
    n_bidders: int,
    *,
    eps: float = 1e-12,
) -> float:
    """Selection probability Pr(S=1 | cutoff) in the baseline debugging case.

    Baseline case: informal bids equal valuations,
        b^I_ij = v_ij,  v_ij ~ Normal(bid_mu, bid_sigma^2).

    With n_bidders bidders, the auction is observed (reaches the formal stage)
    if at least one bidder is admitted:
        S=1  <=>  exists j: b^I_ij >= cutoff.

    Thus:
        Pr(S=1 | cutoff) = 1 - Pr(all bids < cutoff)
                         = 1 - Phi((cutoff - bid_mu) / bid_sigma) ^ n_bidders.
    """
    if n_bidders <= 0:
        raise ValueError("n_bidders must be positive")
    if bid_sigma <= 0:
        raise ValueError("bid_sigma must be positive")

    z = (float(cutoff) - float(bid_mu)) / float(bid_sigma)
    p_below = float(ndtr(z))
    p_select = 1.0 - (p_below ** int(n_bidders))

    # Avoid division-by-zero / numerical issues in MH ratios.
    if p_select < eps:
        return float(eps)
    if p_select > 1.0:
        return 1.0
    return float(p_select)


def gelman_rubin(chains: List[np.ndarray]) -> float:
    """Compute Gelman-Rubin R-hat convergence diagnostic.

    For univariate chains, returns the R-hat statistic.
    For multivariate chains, returns the maximum R-hat across parameters.

    Args:
        chains: List of MCMC chain arrays (each chain is an array of samples)

    Returns:
        R-hat statistic (values < 1.1 indicate convergence)
    """
    if chains[0].ndim == 1:
        m = len(chains)
        n = len(chains[0])

        chain_means = np.array([np.mean(chain) for chain in chains])
        B = n * np.var(chain_means, ddof=1)

        chain_vars = np.array([np.var(chain, ddof=1) for chain in chains])
        W = np.mean(chain_vars)

        var_plus = ((n - 1) / n) * W + (1 / n) * B
        rhat = np.sqrt(var_plus / W) if W > 0 else 1.0
        return rhat

    # For multivariate chains, report the max R-hat across parameters
    n_params = chains[0].shape[1]
    rhats = []
    for k in range(n_params):
        rhats.append(gelman_rubin([chain[:, k] for chain in chains]))
    return float(np.max(rhats))


def draw_covariates(k: int, x_mean: float = 0.0, x_std: float = 1.0) -> np.ndarray:
    """Draw auction covariates with intercept.

    Generates a covariate vector of length k where the first element is
    always 1.0 (intercept) and remaining elements are drawn from N(x_mean, x_std).

    Args:
        k: Total number of covariates (including intercept)
        x_mean: Mean for non-intercept covariates (can be scalar or array)
        x_std: Std dev for non-intercept covariates (can be scalar or array)

    Returns:
        Array of shape (k,) with covariates [1.0, x_1, x_2, ...]
    """
    if k == 1:
        return np.array([1.0])

    mean = np.atleast_1d(x_mean).astype(float)
    std = np.atleast_1d(x_std).astype(float)

    if mean.size == 1:
        mean = np.full(k - 1, mean.item())
    if std.size == 1:
        std = np.full(k - 1, std.item())

    if mean.size != k - 1 or std.size != k - 1:
        raise ValueError(f"x_mean/x_std must have length {k - 1}, got {mean.size}/{std.size}")

    z = np.random.normal(mean, std)
    return np.concatenate(([1.0], z))


def compute_mean_x(k: int, x_mean: float = 0.0) -> np.ndarray:
    """Compute the mean covariate vector.

    Args:
        k: Total number of covariates (including intercept)
        x_mean: Mean for non-intercept covariates

    Returns:
        Array of shape (k,) with mean covariates [1.0, x_mean, x_mean, ...]
    """
    if k == 1:
        return np.array([1.0])

    mean = np.atleast_1d(x_mean).astype(float)
    if mean.size == 1:
        mean = np.full(k - 1, mean.item())
    if mean.size != k - 1:
        raise ValueError(f"x_mean must have length {k - 1}, got {mean.size}")

    return np.concatenate(([1.0], mean))
