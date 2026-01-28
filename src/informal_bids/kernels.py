from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    from numba import config as numba_config
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Numba is required for this branch (performance-critical). "
        "Install numba or use branch `two_stage` for the legacy implementation."
    ) from exc


HAS_NUMBA = True
NUMBA_JIT_ENABLED = not bool(getattr(numba_config, "DISABLE_JIT", 0))


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def norm_cdf(x: float) -> float:
    # 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865475))


@njit(cache=True)
def informal_bid_multiplier_numba(n_bidders: int, kappa: float, mode_flag: int) -> float:
    """Numba version of lambda_I(J,kappa).

    mode_flag: 0 = scale  (lambda_I = (1-1/J)*exp(kappa))
               1 = shift  (lambda_I = exp(kappa))
    """
    exp_k = math.exp(kappa)
    if mode_flag == 1:
        return exp_k
    return (1.0 - 1.0 / float(n_bidders)) * exp_k


@njit(cache=True)
def selection_prob_baseline_numba(
    b_star: float,
    gamma: float,
    sigma_nu: float,
    n_bidders: int,
    kappa: float,
    mode_flag: int,
    eps: float,
) -> float:
    lam_i = informal_bid_multiplier_numba(n_bidders, kappa, mode_flag)
    z = (b_star / lam_i - gamma) / sigma_nu
    eps_p = 1e-15
    p_below = norm_cdf(z)
    if p_below < eps_p:
        p_below = eps_p
    if p_below > 1.0 - eps_p:
        p_below = 1.0 - eps_p
    log_pnone = float(n_bidders) * math.log(p_below)
    p_select = -math.expm1(log_pnone)
    if p_select < eps:
        return eps
    if p_select > 1.0:
        return 1.0
    return p_select


@njit(cache=True)
def selection_prob_type_shift_numba(
    b_star: float,
    gamma: float,
    sigma_nu: float,
    nS: int,
    nF: int,
    kappa: float,
    delta: float,
    mode_flag: int,
    eps: float,
) -> float:
    n = nS + nF
    lam_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
    zF = (b_star / lam_i - gamma) / sigma_nu
    zS = ((b_star + delta) / lam_i - gamma) / sigma_nu
    eps_p = 1e-15
    p_below_F = norm_cdf(zF)
    if p_below_F < eps_p:
        p_below_F = eps_p
    if p_below_F > 1.0 - eps_p:
        p_below_F = 1.0 - eps_p
    p_below_S = norm_cdf(zS)
    if p_below_S < eps_p:
        p_below_S = eps_p
    if p_below_S > 1.0 - eps_p:
        p_below_S = 1.0 - eps_p
    log_pnone = float(nS) * math.log(p_below_S) + float(nF) * math.log(p_below_F)
    p_select = -math.expm1(log_pnone)
    if p_select < eps:
        return eps
    if p_select > 1.0:
        return 1.0
    return p_select


@njit(cache=True)
def task_b_bounds_type_shift(
    bI: np.ndarray,
    admitted: np.ndarray,
    T: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Packed (L,U) bounds for type-shift admission: bI >= b* + delta*T.

    Uses adjusted bids bI_adj = bI - delta*T, so bounds are computed over bI_adj.

    This is called every MCMC iteration for Candidate 3, so it must avoid Python
    loops and allocations.
    """
    N = n_bidders.shape[0]
    L = np.empty(N, dtype=np.float64)
    U = np.empty(N, dtype=np.float64)

    for i in range(N):
        n = int(n_bidders[i])
        start = int(offsets[i])
        end = start + n

        n_adm = 0
        min_adm = math.inf
        max_rej = -math.inf

        for idx in range(start, end):
            val = float(bI[idx]) - float(delta) * float(T[idx])
            if admitted[idx]:
                n_adm += 1
                if val < min_adm:
                    min_adm = val
            else:
                if val > max_rej:
                    max_rej = val

        if n_adm == 0:
            L[i] = math.inf
            U[i] = -math.inf
        elif n_adm == n:
            L[i] = -math.inf
            U[i] = min_adm
        else:
            L[i] = max_rej
            U[i] = min_adm

    return L, U


@njit(cache=True)
def task_b_log_selection_sum(
    b_star: np.ndarray,
    n_bidders: np.ndarray,
    nS: np.ndarray,
    nF: np.ndarray,
    gamma: float,
    sigma_nu: float,
    kappa: float,
    delta: float,
    mode_flag: int,
    use_type_shift: int,
) -> float:
    eps = 1e-12
    total = 0.0
    N = n_bidders.shape[0]
    for i in range(N):
        if use_type_shift == 1:
            p = selection_prob_type_shift_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                int(nS[i]),
                int(nF[i]),
                kappa,
                delta,
                mode_flag,
                eps,
            )
        else:
            p = selection_prob_baseline_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                int(n_bidders[i]),
                kappa,
                mode_flag,
                eps,
            )
        total += math.log(p)
    return total


@njit(cache=True)
def task_b_logpost_gamma(
    gamma: float,
    gamma0: float,
    s_gamma: float,
    b_star: np.ndarray,
    bI: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    nS: np.ndarray,
    nF: np.ndarray,
    sigma_nu: float,
    kappa: float,
    delta: float,
    mode_flag: int,
    use_type_shift: int,
) -> float:
    """Log-posterior for gamma (Task B) with selection penalty.

    Constants in the v-density cancel in MH ratios, so we omit log(2π) terms.
    """
    logp = -0.5 * ((gamma - gamma0) ** 2) / (s_gamma * s_gamma)
    eps = 1e-12

    N = n_bidders.shape[0]
    for i in range(N):
        n = int(n_bidders[i])
        lam_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        start = int(offsets[i])
        end = start + n

        for idx in range(start, end):
            v = bI[idx] / lam_i
            diff = (v - gamma) / sigma_nu
            logp += -0.5 * diff * diff
        logp += -float(n) * math.log(sigma_nu)

        if use_type_shift == 1:
            p_select = selection_prob_type_shift_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                int(nS[i]),
                int(nF[i]),
                kappa,
                delta,
                mode_flag,
                eps,
            )
        else:
            p_select = selection_prob_baseline_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                n,
                kappa,
                mode_flag,
                eps,
            )
        logp += -math.log(p_select)

    return logp


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
    admitted: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    lambda_f: np.ndarray,
    nS: np.ndarray,
    nF: np.ndarray,
    delta: float,
    mode_flag: int,
    use_type_shift: int,
) -> float:
    """Log-posterior for kappa (Task B) with selection penalty."""
    logp = -0.5 * ((kappa - kappa0) ** 2) / (s_kappa * s_kappa) - math.log(s_kappa)
    eps = 1e-12
    log2pi = 1.8378770664093453  # log(2*pi)

    N = n_bidders.shape[0]
    for i in range(N):
        n = int(n_bidders[i])
        lam_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        lf = float(lambda_f[i])
        start = int(offsets[i])
        end = start + n

        # Informal bids likelihood (bI) and formal bids likelihood for admitted bidders.
        for idx in range(start, end):
            v = bI[idx] / lam_i
            z = (v - gamma) / sigma_nu
            logp += -0.5 * z * z
            if admitted[idx]:
                resid = bF[idx] / lf - v
                logp += -0.5 * (resid / sigma_eta) * (resid / sigma_eta)
                logp += -(math.log(sigma_eta) + math.log(lf) + 0.5 * log2pi)
        logp += -float(n) * (math.log(sigma_nu) + math.log(lam_i) + 0.5 * log2pi)

        # Selection penalty
        if use_type_shift == 1:
            p_select = selection_prob_type_shift_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                int(nS[i]),
                int(nF[i]),
                kappa,
                delta,
                mode_flag,
                eps,
            )
        else:
            p_select = selection_prob_baseline_numba(
                float(b_star[i]),
                gamma,
                sigma_nu,
                n,
                kappa,
                mode_flag,
                eps,
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
) -> Tuple[float, int]:
    """Return sum of squared residuals for v = bI/lambda_i around gamma."""
    ss = 0.0
    n_v = 0
    N = n_bidders.shape[0]
    for i in range(N):
        n = int(n_bidders[i])
        lam_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        start = int(offsets[i])
        end = start + n
        for idx in range(start, end):
            v = bI[idx] / lam_i
            diff = v - gamma
            ss += diff * diff
            n_v += 1
    return ss, n_v


@njit(cache=True)
def task_b_sum_sq_eta(
    bI: np.ndarray,
    bF: np.ndarray,
    admitted: np.ndarray,
    offsets: np.ndarray,
    n_bidders: np.ndarray,
    lambda_f: np.ndarray,
    kappa: float,
    mode_flag: int,
) -> Tuple[float, int]:
    """Return sum of squared eta residuals and count of admitted bidders."""
    ss = 0.0
    n_eta = 0
    N = n_bidders.shape[0]
    for i in range(N):
        n = int(n_bidders[i])
        lam_i = informal_bid_multiplier_numba(n, kappa, mode_flag)
        lf = float(lambda_f[i])
        start = int(offsets[i])
        end = start + n
        for idx in range(start, end):
            if admitted[idx]:
                v = bI[idx] / lam_i
                resid = bF[idx] / lf - v
                ss += resid * resid
                n_eta += 1
    return ss, n_eta
