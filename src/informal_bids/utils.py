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
from typing import List, Literal, Tuple


MisreportingMode = Literal["scale", "shift"]
CutoffSpec = Literal["intercept", "moments_k4", "depth_k2", "depth_k2_ratio"]


def lambda_f_first_price(n_bidders: int) -> float:
    """First-price shading factor lambda_f = 1 - 1/n."""
    if n_bidders <= 1:
        raise ValueError("n_bidders must be >= 2")
    return float(1.0 - 1.0 / float(n_bidders))


def informal_bid_multiplier(
    n_bidders: int,
    kappa: float,
    *,
    mode: MisreportingMode = "scale",
) -> float:
    """Informal-stage bid multiplier lambda_I(n, kappa) under a chosen parameterization.

    - mode='scale' (analysis report / README): lambda_I = (1 - 1/n) * exp(kappa).
    - mode='shift' (meeting notes): lambda_I = exp(kappa), which corresponds to
      the additive meeting-notes parameter alpha via:
          alpha = exp(kappa) - (1 - 1/n).
    """
    if mode not in ("scale", "shift"):
        raise ValueError("mode must be 'scale' or 'shift'")
    if mode == "scale":
        return float(lambda_f_first_price(n_bidders) * np.exp(float(kappa)))
    return float(np.exp(float(kappa)))


def misreporting_measures(
    n_bidders: int,
    kappa: float,
    *,
    mode: MisreportingMode = "scale",
) -> Tuple[float, float, float, float]:
    """Return (lambda_f, lambda_i, tilde_alpha, alpha_additive).

    Definitions:
    - lambda_f = 1 - 1/n
    - lambda_i = informal bid multiplier
    - tilde_alpha = lambda_i / lambda_f - 1 (analysis report)
    - alpha_additive = lambda_i - lambda_f (meeting notes)
    """
    lam_f = float(lambda_f_first_price(n_bidders))
    lam_i = float(informal_bid_multiplier(n_bidders, kappa, mode=mode))
    tilde_alpha = float(lam_i / lam_f - 1.0)
    alpha_additive = float(lam_i - lam_f)
    return lam_f, lam_i, tilde_alpha, alpha_additive


def cutoff_feature_names(spec: CutoffSpec) -> List[str]:
    """Return feature names for a cutoff specification (including intercept)."""
    if spec == "intercept":
        return ["c"]
    if spec == "moments_k4":
        return ["c", "m1_top1", "m2_top2_avg", "m3_top3_avg"]
    if spec == "depth_k2":
        return ["c", "depth_mean_23", "depth_gap_23"]
    if spec == "depth_k2_ratio":
        return ["c", "depth_mean_23", "depth_gap_23_ratio"]
    raise ValueError(f"Unknown cutoff spec: {spec}")


def cutoff_expected_k(spec: CutoffSpec) -> int:
    """Return expected number of coefficients (including intercept) for a cutoff spec."""
    if spec == "intercept":
        return 1
    if spec == "moments_k4":
        return 4
    if spec in ("depth_k2", "depth_k2_ratio"):
        return 3
    raise ValueError(f"Unknown cutoff spec: {spec}")


def compute_cutoff_features(informal_bids: np.ndarray, spec: CutoffSpec) -> np.ndarray:
    """Compute cutoff covariates X_i from informal bids for a chosen specification."""
    bids_sorted = np.sort(informal_bids)[::-1]
    if spec == "intercept":
        return np.array([1.0], dtype=float)

    if bids_sorted.size < 3:
        raise ValueError("cutoff feature construction requires at least 3 bidders")

    if spec == "moments_k4":
        m1 = float(bids_sorted[0])
        m2 = float(np.mean(bids_sorted[:2]))
        m3 = float(np.mean(bids_sorted[:3]))
        return np.array([1.0, m1, m2, m3], dtype=float)

    if spec in ("depth_k2", "depth_k2_ratio"):
        b2 = float(bids_sorted[1])
        b3 = float(bids_sorted[2])
        m1 = 0.5 * (b2 + b3)
        gap = b2 - b3
        if spec == "depth_k2_ratio":
            denom = abs(m1) + 1e-6
            gap = gap / denom
        return np.array([1.0, m1, gap], dtype=float)

    raise ValueError(f"Unknown cutoff spec: {spec}")


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


def informal_bid_multiplier_option_c(n_bidders: int, kappa: float) -> float:
    """Informal-stage bid multiplier under option C (kappa reparam).

    Option C in the Jan 14, 2026 notes:
        b^I = (1 - 1/n) (1 + tilde_alpha) v
    with (1 + tilde_alpha) = exp(kappa), so:
        lambda_I(n, kappa) = (1 - 1/n) * exp(kappa).
    """
    if n_bidders <= 1:
        raise ValueError("n_bidders must be >= 2")
    return float((1.0 - 1.0 / float(n_bidders)) * np.exp(float(kappa)))


def selection_prob_reaches_formal_stage(
    cutoff: float,
    gamma: float,
    sigma_nu: float,
    n_bidders: int,
    kappa: float,
    *,
    mode: MisreportingMode = "scale",
    eps: float = 1e-12,
) -> float:
    """Selection probability Pr(S=1 | cutoff) for Task B two-stage DGP.

    Under option C:
        v_ij ~ Normal(gamma, sigma_nu^2)
        b^I_ij = lambda_I(n_i, kappa) * v_ij
        S_i = 1  <=>  exists j: b^I_ij >= cutoff
                  <=> exists j: v_ij >= cutoff / lambda_I.

    Thus:
        Pr(S=1 | cutoff) = 1 - Phi((cutoff / lambda_I - gamma) / sigma_nu) ^ n_bidders.
    """
    if n_bidders <= 1:
        raise ValueError("n_bidders must be >= 2")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")

    lambda_i = informal_bid_multiplier(n_bidders, kappa, mode=mode)
    z = (float(cutoff) / lambda_i - float(gamma)) / float(sigma_nu)
    p_below = float(ndtr(z))
    p_select = 1.0 - (p_below ** int(n_bidders))

    if p_select < eps:
        return float(eps)
    if p_select > 1.0:
        return 1.0
    return float(p_select)


def calibrate_cutoff_intercept_for_target_mean(
    *,
    target_mean_cutoff: float,
    theta: np.ndarray,
    n_bidders: int,
    gamma: float,
    sigma_nu: float,
    kappa: float,
    misreporting_mode: MisreportingMode = "scale",
    cutoff_spec: CutoffSpec = "moments_k4",
    n_sim: int = 20000,
    seed: int = 0,
) -> float:
    """Calibrate the cutoff intercept so E[b*_i] matches a target in the moments model.

    For the Task B moments cutoff with X_i = [1, m1, m2, m3] and
        b*_i = c + theta1*m1 + theta2*m2 + theta3*m3 + omega_i,
    we choose c such that E[b*_i] ≈ target_mean_cutoff (using Monte Carlo).

    Notes:
    - This uses the unconditional moments of informal bids (initiated auctions).
    - Requires n_bidders >= 3 to define the top-3 average.
    """
    theta = np.atleast_1d(theta).astype(float)
    expected_k = cutoff_expected_k(cutoff_spec)
    if expected_k == 1:
        if theta.size != 0:
            raise ValueError("theta must be empty for intercept-only cutoff")
    else:
        if theta.size != expected_k - 1:
            raise ValueError(f"theta must have length {expected_k - 1} for spec '{cutoff_spec}'")
    if n_bidders < 3:
        raise ValueError("n_bidders must be >= 3 to compute cutoff moments")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")

    rng = np.random.default_rng(int(seed))
    v = float(gamma) + rng.normal(0.0, float(sigma_nu), size=(int(n_sim), int(n_bidders)))
    lambda_i = informal_bid_multiplier(int(n_bidders), float(kappa), mode=misreporting_mode)
    bids = lambda_i * v

    bids_sorted = np.sort(bids, axis=1)[:, ::-1]

    if cutoff_spec == "moments_k4":
        m1 = bids_sorted[:, 0]
        m2 = np.mean(bids_sorted[:, :2], axis=1)
        m3 = np.mean(bids_sorted[:, :3], axis=1)
        implied = theta[0] * m1 + theta[1] * m2 + theta[2] * m3
    elif cutoff_spec in ("depth_k2", "depth_k2_ratio"):
        b2 = bids_sorted[:, 1]
        b3 = bids_sorted[:, 2]
        m1 = 0.5 * (b2 + b3)
        gap = b2 - b3
        if cutoff_spec == "depth_k2_ratio":
            gap = gap / (np.abs(m1) + 1e-6)
        implied = theta[0] * m1 + theta[1] * gap
    elif cutoff_spec == "intercept":
        implied = np.zeros(int(n_sim), dtype=float)
    else:
        raise ValueError(f"Unknown cutoff spec: {cutoff_spec}")

    c = float(target_mean_cutoff) - float(np.mean(implied))
    return float(c)


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
