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
def seed_numba_rng(seed: int) -> None:
    """Seed Numba RNG once per chain."""
    np.random.seed(seed)


@njit(cache=True)
def informal_bid_multiplier_numba(n_bidders: int, kappa: float, mode_flag: int) -> float:
    """Numba version of informal_bid_multiplier with mode flag (0=scale, 1=shift)."""
    if mode_flag == 0:
        return (1.0 - 1.0 / float(n_bidders)) * math.exp(kappa)
    return math.exp(kappa)


@njit(cache=True)
def selection_prob_reaches_formal_stage_numba(
    cutoff: float,
    gamma: float,
    sigma_nu: float,
    n_bidders: int,
    kappa: float,
    mode_flag: int,
    eps: float,
) -> float:
    lambda_i = informal_bid_multiplier_numba(n_bidders, kappa, mode_flag)
    z = (cutoff / lambda_i - gamma) / sigma_nu
    p_below = norm_cdf(z)
    p_select = 1.0 - (p_below ** n_bidders)
    if p_select < eps:
        return eps
    if p_select > 1.0:
        return 1.0
    return p_select


@njit(cache=True)
def task_b_update_b_star(
    b_star: np.ndarray,
    X: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    n_bidders: np.ndarray,
    beta: np.ndarray,
    sigma_omega: float,
    gamma: float,
    sigma_nu: float,
    kappa: float,
    mode_flag: int,
) -> tuple[int, int]:
    """Update b_star via independence MH (Task B). Returns (accept_count, total_count)."""
    N = b_star.shape[0]
    k = beta.shape[0]
    accept = 0
    total = 0
    eps = 1e-12

    for i in range(N):
        xb = 0.0
        for j in range(k):
            xb += X[i, j] * beta[j]
        a = (L[i] - xb) / sigma_omega
        b = (U[i] - xb) / sigma_omega
        b_prop = xb + truncnorm_std_rvs(a, b) * sigma_omega

        n = int(n_bidders[i])
        p_old = selection_prob_reaches_formal_stage_numba(b_star[i], gamma, sigma_nu, n, kappa, mode_flag, eps)
        p_prop = selection_prob_reaches_formal_stage_numba(b_prop, gamma, sigma_nu, n, kappa, mode_flag, eps)
        alpha = p_old / p_prop

        total += 1
        if alpha >= 1.0 or np.random.random() < alpha:
            b_star[i] = b_prop
            accept += 1

    return accept, total


@njit(cache=True)
def task_b_logpost_gamma(
    gamma: float,
    gamma0: float,
    s_gamma: float,
    b_star: np.ndarray,
    bI: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    sigma_nu: float,
    kappa: float,
    mode_flag: int,
) -> float:
    """Log-posterior for gamma (Task B) with selection penalty."""
    logp = -0.5 * ((gamma - gamma0) ** 2) / (s_gamma * s_gamma)
    eps = 1e-12

    n_auctions = n_bidders.shape[0]
    for i in range(n_auctions):
        n = int(n_bidders[i])
        lambda_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        start = int(offsets[i])
        end = start + n
        for idx in range(start, end):
            v = bI[idx] / lambda_i
            diff = (v - gamma) / sigma_nu
            logp += -0.5 * diff * diff
        logp += -float(n) * math.log(sigma_nu)
        p_select = selection_prob_reaches_formal_stage_numba(
            b_star[i], gamma, sigma_nu, n, kappa, mode_flag, eps
        )
        logp += -math.log(p_select)
    return logp


@njit(cache=True)
def task_b_sum_sq_v(
    bI: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    gamma: float,
    kappa: float,
    mode_flag: int,
) -> tuple[float, int]:
    """Return sum of squared residuals for v = bI/lambda_i around gamma."""
    ss = 0.0
    n_v = 0
    n_auctions = n_bidders.shape[0]
    for i in range(n_auctions):
        n = int(n_bidders[i])
        lambda_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        start = int(offsets[i])
        end = start + n
        for idx in range(start, end):
            v = bI[idx] / lambda_i
            diff = v - gamma
            ss += diff * diff
            n_v += 1
    return ss, n_v


@njit(cache=True)
def task_b_log_selection_sum(
    b_star: np.ndarray,
    n_bidders: np.ndarray,
    gamma: float,
    sigma_nu: float,
    kappa: float,
    mode_flag: int,
) -> float:
    """Return sum of log selection probabilities across auctions."""
    eps = 1e-12
    total = 0.0
    n_auctions = n_bidders.shape[0]
    for i in range(n_auctions):
        n = int(n_bidders[i])
        p = selection_prob_reaches_formal_stage_numba(
            b_star[i], gamma, sigma_nu, n, kappa, mode_flag, eps
        )
        total += math.log(p)
    return total


@njit(cache=True)
def task_b_logpost_kappa(
    kappa: float,
    kappa0: float,
    s_kappa: float,
    gamma: float,
    sigma_nu: float,
    sigma_eta: float,
    b_star: np.ndarray,
    bI: np.ndarray,
    bF: np.ndarray,
    adm: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    lambda_f: np.ndarray,
    mode_flag: int,
) -> float:
    """Log-posterior for kappa (Task B) with selection penalty."""
    logp = -0.5 * ((kappa - kappa0) ** 2) / (s_kappa * s_kappa) - math.log(s_kappa)
    eps = 1e-12
    log2pi = 1.8378770664093453

    n_auctions = n_bidders.shape[0]
    for i in range(n_auctions):
        n = int(n_bidders[i])
        lambda_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        start = int(offsets[i])
        end = start + n

        # Informal bids likelihood (bI)
        for idx in range(start, end):
            v = bI[idx] / lambda_i
            z = (v - gamma) / sigma_nu
            logp += -0.5 * z * z
        logp += -float(n) * (math.log(sigma_nu) + math.log(lambda_i) + 0.5 * log2pi)

        # Formal bids likelihood for admitted bidders
        lf = float(lambda_f[i])
        for idx in range(start, end):
            if adm[idx]:
                v = bI[idx] / lambda_i
                resid = bF[idx] / lf - v
                logp += -0.5 * (resid / sigma_eta) * (resid / sigma_eta)
                logp += -(math.log(sigma_eta) + math.log(lf) + 0.5 * log2pi)

        # Selection penalty
        p_select = selection_prob_reaches_formal_stage_numba(
            b_star[i], gamma, sigma_nu, n, kappa, mode_flag, eps
        )
        logp += -math.log(p_select)

    return logp
