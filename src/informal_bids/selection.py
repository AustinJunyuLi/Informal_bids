from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import ndtr

from .config import EPS_SELECT
from .misreporting import MisreportingMode, informal_bid_multiplier


def _lambda_i_vec(n_bidders: np.ndarray, *, kappa: float, misreporting_mode: MisreportingMode) -> np.ndarray:
    """Vectorized lambda_I over auctions."""
    n_bidders = np.asarray(n_bidders, dtype=float)
    exp_k = float(np.exp(float(kappa)))
    if misreporting_mode == "shift":
        return np.full_like(n_bidders, exp_k, dtype=float)
    if misreporting_mode == "scale":
        return (1.0 - 1.0 / n_bidders) * exp_k
    raise ValueError("misreporting_mode must be 'scale' or 'shift'")


def p_select_baseline_scalar(
    b_star: float,
    *,
    gamma: float,
    sigma_nu: float,
    n_bidders: int,
    kappa: float,
    misreporting_mode: MisreportingMode,
    eps: float = EPS_SELECT,
) -> float:
    if n_bidders <= 1:
        raise ValueError("n_bidders must be >= 2")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")

    lam_i = informal_bid_multiplier(n_bidders, kappa, mode=misreporting_mode)
    z = (float(b_star) / lam_i - float(gamma)) / float(sigma_nu)
    eps_p = 1e-15
    p_below = float(np.clip(float(ndtr(z)), eps_p, 1.0 - eps_p))
    log_pnone = float(n_bidders) * float(np.log(p_below))
    p_select = -float(np.expm1(log_pnone))
    return float(np.clip(p_select, eps, 1.0))


def p_select_type_shift_scalar(
    b_star: float,
    *,
    gamma: float,
    sigma_nu: float,
    nS: int,
    nF: int,
    kappa: float,
    delta: float,
    misreporting_mode: MisreportingMode,
    eps: float = EPS_SELECT,
) -> float:
    if nS < 0 or nF < 0:
        raise ValueError("nS and nF must be nonnegative")
    n = int(nS) + int(nF)
    if n <= 1:
        raise ValueError("total bidders must be >= 2")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")

    lam_i = informal_bid_multiplier(n, kappa, mode=misreporting_mode)
    zF = (float(b_star) / lam_i - float(gamma)) / float(sigma_nu)
    zS = ((float(b_star) + float(delta)) / lam_i - float(gamma)) / float(sigma_nu)

    eps_p = 1e-15
    p_below_F = float(np.clip(float(ndtr(zF)), eps_p, 1.0 - eps_p))
    p_below_S = float(np.clip(float(ndtr(zS)), eps_p, 1.0 - eps_p))
    log_pnone = float(nS) * float(np.log(p_below_S)) + float(nF) * float(np.log(p_below_F))
    p_select = -float(np.expm1(log_pnone))
    return float(np.clip(p_select, eps, 1.0))


def p_select_baseline(
    b_star: np.ndarray,
    *,
    gamma: float,
    sigma_nu: float,
    n_bidders: Iterable[int],
    kappa: float,
    misreporting_mode: MisreportingMode,
    eps: float = EPS_SELECT,
) -> np.ndarray:
    b_star = np.asarray(b_star, dtype=float)
    n_bidders_arr = np.asarray(n_bidders, dtype=int)
    if np.any(n_bidders_arr <= 1):
        raise ValueError("n_bidders must be >= 2")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")

    lam_i = _lambda_i_vec(n_bidders_arr, kappa=float(kappa), misreporting_mode=misreporting_mode)
    z = (b_star / lam_i - float(gamma)) / float(sigma_nu)
    eps_p = 1e-15
    p_below = np.clip(ndtr(z), eps_p, 1.0 - eps_p)
    log_pnone = n_bidders_arr.astype(float) * np.log(p_below)
    p_select = -np.expm1(log_pnone)
    return np.clip(p_select, float(eps), 1.0)


def p_select_type_shift(
    b_star: np.ndarray,
    *,
    gamma: float,
    sigma_nu: float,
    nS: Iterable[int],
    nF: Iterable[int],
    kappa: float,
    delta: float,
    misreporting_mode: MisreportingMode,
    eps: float = EPS_SELECT,
) -> np.ndarray:
    b_star = np.asarray(b_star, dtype=float)
    nS_arr = np.asarray(nS, dtype=int)
    nF_arr = np.asarray(nF, dtype=int)
    n_tot = nS_arr + nF_arr
    if np.any(n_tot <= 1):
        raise ValueError("total bidders must be >= 2")
    if sigma_nu <= 0:
        raise ValueError("sigma_nu must be positive")

    lam_i = _lambda_i_vec(n_tot, kappa=float(kappa), misreporting_mode=misreporting_mode)
    zF = (b_star / lam_i - float(gamma)) / float(sigma_nu)
    zS = ((b_star + float(delta)) / lam_i - float(gamma)) / float(sigma_nu)
    eps_p = 1e-15
    p_below_F = np.clip(ndtr(zF), eps_p, 1.0 - eps_p)
    p_below_S = np.clip(ndtr(zS), eps_p, 1.0 - eps_p)
    log_pnone = nS_arr.astype(float) * np.log(p_below_S) + nF_arr.astype(float) * np.log(p_below_F)
    p_select = -np.expm1(log_pnone)
    return np.clip(p_select, float(eps), 1.0)
