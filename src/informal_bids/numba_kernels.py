"""Numba-accelerated kernels for MCMC.

These functions are intentionally written for the simulation exercise where
covariates are typically intercept-only (k=1). For k>1, the main scripts fall
back to the pure-Python/Numpy implementation.

We implement:
- Truncated standard normal sampling via inverse-CDF with an inlined normal PPF.
- Inverse-gamma sampling via reciprocal-gamma.

All kernels are `njit(cache=True)` so repeated sensitivity runs amortize compile
cost.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator


_SQRT2 = math.sqrt(2.0)


@njit(cache=True)
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@njit(cache=True)
def norm_ppf(p: float) -> float:
    """Approximate inverse standard normal CDF (Acklam-style rational approximation)."""
    # Coefficients
    a0 = -3.969683028665376e01
    a1 = 2.209460984245205e02
    a2 = -2.759285104469687e02
    a3 = 1.383577518672690e02
    a4 = -3.066479806614716e01
    a5 = 2.506628277459239e00

    b0 = -5.447609879822406e01
    b1 = 1.615858368580409e02
    b2 = -1.556989798598866e02
    b3 = 6.680131188771972e01
    b4 = -1.328068155288572e01

    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e00
    c3 = -2.549732539343734e00
    c4 = 4.374664141464968e00
    c5 = 2.938163982698783e00

    d0 = 7.784695709041462e-03
    d1 = 3.224671290700398e-01
    d2 = 2.445134137142996e00
    d3 = 3.754408661907416e00

    plow = 0.02425
    phigh = 1.0 - plow

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
        den = ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
        return num / den

    if p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5)
        den = (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
        return (num / den) * q

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
    den = ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
    return -(num / den)


@njit(cache=True)
def truncnorm_std_rvs(a: float, b: float) -> float:
    """Sample Z ~ N(0,1) truncated to [a,b], allowing +/-inf bounds."""
    eps = 1e-12

    lo = 0.0 if math.isinf(a) and a < 0 else norm_cdf(a)
    hi = 1.0 if math.isinf(b) and b > 0 else norm_cdf(b)

    # Clip away from 0/1 to avoid infinities in ppf.
    if lo < eps:
        lo = eps
    if lo > 1.0 - eps:
        lo = 1.0 - eps

    if hi < eps:
        hi = eps
    if hi > 1.0 - eps:
        hi = 1.0 - eps

    # In pathological cases, fall back to midpoint.
    width = hi - lo
    if width <= 0.0:
        u = 0.5 * (lo + hi)
    else:
        u = lo + width * np.random.random()

    return norm_ppf(u)


@njit(cache=True)
def invgamma_rvs(shape: float, scale: float) -> float:
    """Sample X ~ InvGamma(shape, scale) using reciprocal-gamma."""
    # If X ~ InvGamma(a, b) with density b^a / Gamma(a) x^{-a-1} exp(-b/x),
    # then 1/X ~ Gamma(a, scale=1/b).
    y = np.random.gamma(shape, 1.0 / scale)
    return 1.0 / y


@njit(cache=True)
def selection_prob_at_least_one_exceeds_cutoff(
    cutoff: float,
    bid_mu: float,
    bid_sigma: float,
    n_bidders: int,
    eps: float,
) -> float:
    z = (cutoff - bid_mu) / bid_sigma
    p_below = norm_cdf(z)
    p_select = 1.0 - (p_below ** n_bidders)
    if p_select < eps:
        return eps
    if p_select > 1.0:
        return 1.0
    return p_select


@njit(cache=True)
def task_a_run_chain_intercept(
    L: np.ndarray,
    U: np.ndarray,
    n_bidders: np.ndarray,
    n_iterations: int,
    mu_prior_mean: float,
    mu_prior_std: float,
    sigma_prior_a: float,
    sigma_prior_b: float,
    mu_init: float,
    sigma_init: float,
    bid_mu: float,
    bid_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Task A sampler for intercept-only cutoff model (selection-aware MH within Gibbs)."""
    np.random.seed(seed)

    N = L.shape[0]
    mu = float(mu_init)
    sigma = float(sigma_init)

    mu_chain = np.zeros(n_iterations, dtype=np.float64)
    sigma_chain = np.zeros(n_iterations, dtype=np.float64)

    tau_prior_sq = float(mu_prior_std) ** 2
    a_prior = float(sigma_prior_a)
    b_prior = float(sigma_prior_b)

    eps_select = 1e-12

    b_star = np.zeros(N, dtype=np.float64)
    for i in range(N):
        a = (L[i] - mu) / sigma
        b = (U[i] - mu) / sigma
        b_star[i] = mu + truncnorm_std_rvs(a, b) * sigma

    for t in range(n_iterations):
        b_star_sum = 0.0
        # Step 1: update b*_i via independence MH
        for i in range(N):
            a = (L[i] - mu) / sigma
            b = (U[i] - mu) / sigma
            b_prop = mu + truncnorm_std_rvs(a, b) * sigma

            p_old = selection_prob_at_least_one_exceeds_cutoff(
                b_star[i], bid_mu, bid_sigma, int(n_bidders[i]), eps_select
            )
            p_prop = selection_prob_at_least_one_exceeds_cutoff(
                b_prop, bid_mu, bid_sigma, int(n_bidders[i]), eps_select
            )

            alpha = p_old / p_prop
            if alpha >= 1.0 or np.random.random() < alpha:
                b_star[i] = b_prop

            b_star_sum += b_star[i]

        # Step 2: update mu
        tau_post_sq = 1.0 / (1.0 / tau_prior_sq + N / (sigma * sigma))
        mu_post = tau_post_sq * (mu_prior_mean / tau_prior_sq + b_star_sum / (sigma * sigma))
        mu = mu_post + np.random.normal() * math.sqrt(tau_post_sq)

        # Step 3: update sigma^2
        a_post = a_prior + 0.5 * N
        ss = 0.0
        for i in range(N):
            diff = b_star[i] - mu
            ss += diff * diff
        b_post = b_prior + 0.5 * ss
        sigma_sq = invgamma_rvs(a_post, b_post)
        sigma = math.sqrt(sigma_sq)

        mu_chain[t] = mu
        sigma_chain[t] = sigma

    return mu_chain, sigma_chain


@njit(cache=True)
def task_b_run_chain_intercept(
    L_S: np.ndarray,
    U_S: np.ndarray,
    L_F: np.ndarray,
    U_F: np.ndarray,
    n_iterations: int,
    mu_S_prior_mean: float,
    mu_S_prior_std: float,
    mu_F_prior_mean: float,
    mu_F_prior_std: float,
    sigma_prior_a: float,
    sigma_prior_b: float,
    mu_S_init: float,
    mu_F_init: float,
    sigma_S_init: float,
    sigma_F_init: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Task B Gibbs sampler for intercept-only (S and F) cutoff model."""
    np.random.seed(seed)

    N_S = L_S.shape[0]
    N_F = L_F.shape[0]

    mu_S = float(mu_S_init)
    mu_F = float(mu_F_init)
    sigma_S = float(sigma_S_init)
    sigma_F = float(sigma_F_init)

    mu_S_chain = np.zeros(n_iterations, dtype=np.float64)
    mu_F_chain = np.zeros(n_iterations, dtype=np.float64)
    sigma_S_chain = np.zeros(n_iterations, dtype=np.float64)
    sigma_F_chain = np.zeros(n_iterations, dtype=np.float64)

    tau_S_sq = float(mu_S_prior_std) ** 2
    tau_F_sq = float(mu_F_prior_std) ** 2

    a_prior = float(sigma_prior_a)
    b_prior = float(sigma_prior_b)

    nu_S = np.zeros(N_S, dtype=np.float64)
    nu_F = np.zeros(N_F, dtype=np.float64)

    for t in range(n_iterations):
        # --- Type S ---
        b_star_S_sum = 0.0
        for i in range(N_S):
            a = (L_S[i] - mu_S) / sigma_S
            b = (U_S[i] - mu_S) / sigma_S
            nu_i = truncnorm_std_rvs(a, b) * sigma_S
            nu_S[i] = nu_i
            b_star_S_sum += mu_S + nu_i

        tau_S_post_sq = 1.0 / (1.0 / tau_S_sq + N_S / (sigma_S * sigma_S))
        mu_S_post = tau_S_post_sq * (mu_S_prior_mean / tau_S_sq + b_star_S_sum / (sigma_S * sigma_S))
        mu_S = mu_S_post + np.random.normal() * math.sqrt(tau_S_post_sq)

        a_S_post = a_prior + 0.5 * N_S
        ss_S = 0.0
        for i in range(N_S):
            ss_S += nu_S[i] * nu_S[i]
        b_S_post = b_prior + 0.5 * ss_S
        sigma_S_sq = invgamma_rvs(a_S_post, b_S_post)
        sigma_S = math.sqrt(sigma_S_sq)

        # --- Type F ---
        b_star_F_sum = 0.0
        for i in range(N_F):
            a = (L_F[i] - mu_F) / sigma_F
            b = (U_F[i] - mu_F) / sigma_F
            nu_i = truncnorm_std_rvs(a, b) * sigma_F
            nu_F[i] = nu_i
            b_star_F_sum += mu_F + nu_i

        tau_F_post_sq = 1.0 / (1.0 / tau_F_sq + N_F / (sigma_F * sigma_F))
        mu_F_post = tau_F_post_sq * (mu_F_prior_mean / tau_F_sq + b_star_F_sum / (sigma_F * sigma_F))
        mu_F = mu_F_post + np.random.normal() * math.sqrt(tau_F_post_sq)

        a_F_post = a_prior + 0.5 * N_F
        ss_F = 0.0
        for i in range(N_F):
            ss_F += nu_F[i] * nu_F[i]
        b_F_post = b_prior + 0.5 * ss_F
        sigma_F_sq = invgamma_rvs(a_F_post, b_F_post)
        sigma_F = math.sqrt(sigma_F_sq)

        mu_S_chain[t] = mu_S
        mu_F_chain[t] = mu_F
        sigma_S_chain[t] = sigma_S
        sigma_F_chain[t] = sigma_F

    return mu_S_chain, mu_F_chain, sigma_S_chain, sigma_F_chain
