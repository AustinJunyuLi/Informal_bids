from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EPS_DIV
from .misreporting import MisreportingMode, informal_bid_multiplier, lambda_f_first_price
from .specs import (
    TASKB_SPECS,
    feature_cv,
    feature_depth_exmax,
    feature_gap23_ratio,
)
from .types import TaskBAuctionObs


def _round_to_tick(x: np.ndarray, tick: np.ndarray) -> np.ndarray:
    tick = np.asarray(tick, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.round(x / tick) * tick


def mean_log_uniform(lo: float, hi: float) -> float:
    """Return E[log U(lo,hi)] for a uniform random variable on (lo,hi)."""
    lo = float(lo)
    hi = float(hi)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("Uniform bounds must be positive for log moments.")
    if hi <= lo:
        raise ValueError("Uniform upper bound must exceed lower bound.")
    return float((hi * np.log(hi) - lo * np.log(lo) - (hi - lo)) / (hi - lo))


def _compute_X_i(
    spec_name: str,
    *,
    bI_obs: np.ndarray,
    T: np.ndarray,
    Z: float,
    tick: np.ndarray,
    tick_alpha_F: Optional[float] = None,
    tick_alpha_S: Optional[float] = None,
) -> np.ndarray:
    """Compute a single-auction cutoff feature row X_i without packing a dataset.

    Must stay consistent with `specs.py`'s `compute_X(...)` implementations.
    """
    bI_obs = np.asarray(bI_obs, dtype=float)
    T = np.asarray(T, dtype=np.int8)
    tick = np.asarray(tick, dtype=float)

    if spec_name == "legacy_intercept":
        return np.array([1.0], dtype=float)

    if spec_name == "legacy_moments_k4":
        b = np.sort(bI_obs)[::-1]
        return np.array([1.0, float(b[0]), float(np.mean(b[:2])), float(np.mean(b[:3]))], dtype=float)

    if spec_name == "legacy_depth_k2":
        b = np.sort(bI_obs)[::-1]
        return np.array([1.0, 0.5 * float(b[1] + b[2]), float(b[1] - b[2])], dtype=float)

    if spec_name == "legacy_depth_k2_ratio":
        b = np.sort(bI_obs)[::-1]
        m1 = 0.5 * float(b[1] + b[2])
        gap = float(b[1] - b[2])
        return np.array([1.0, m1, gap / (abs(m1) + EPS_DIV)], dtype=float)

    # Candidate 1 / 3: type share + gap23 ratio (scale-free)
    if spec_name in ("cand1_type_spr", "cand3_type_shift_admission"):
        g_shareS = float(np.mean(T == 1))
        spr = feature_gap23_ratio(bI_obs)
        return np.array([1.0, g_shareS, spr], dtype=float)

    # Candidate 2: type share + CV spread + depth (excluding max) + Z
    if spec_name == "cand2_type_spr_depth_z":
        g_shareS = float(np.mean(T == 1))
        spr = feature_cv(bI_obs)
        depth = feature_depth_exmax(bI_obs)
        return np.array([1.0, g_shareS, spr, depth, float(Z)], dtype=float)

    # Candidate 4: type share + gap23 ratio + precision proxy + Z
    if spec_name == "cand4_type_spr_prec_z":
        g_shareS = float(np.mean(T == 1))
        spr = feature_gap23_ratio(bI_obs)
        if tick_alpha_F is None or tick_alpha_S is None:
            p_prec = float(np.median(tick))
        else:
            alpha = np.where(T == 1, float(tick_alpha_S), float(tick_alpha_F)).astype(float)
            p_prec = -10.0 * float(np.mean(np.log(tick) - alpha))
        return np.array([1.0, g_shareS, spr, p_prec, float(Z)], dtype=float)

    raise ValueError(f"Unknown spec_name '{spec_name}'")


DEFAULT_BETA_BY_SPEC = {
    # New candidates
    "cand1_type_spr": np.array([1.19, 0.25, 0.60], dtype=float),
    "cand2_type_spr_depth_z": np.array([0.41, 0.25, 1.00, 0.80, 0.05], dtype=float),
    "cand3_type_shift_admission": np.array([1.19, 0.25, 0.60], dtype=float),
    "cand4_type_spr_prec_z": np.array([1.05, 0.25, 0.60, 0.40, 0.05], dtype=float),
    # Legacy baselines
    "legacy_intercept": np.array([1.40], dtype=float),
    "legacy_moments_k4": np.array([0.83, 0.25, 0.10, 0.05], dtype=float),
    "legacy_depth_k2": np.array([1.26, 0.10, 0.10], dtype=float),
    "legacy_depth_k2_ratio": np.array([1.19, 0.10, 0.60], dtype=float),
}


DEFAULT_DELTA_BY_SPEC = {
    "cand3_type_shift_admission": 0.05,
}


def default_beta_for_spec(spec_name: str) -> np.ndarray:
    if spec_name not in DEFAULT_BETA_BY_SPEC:
        raise ValueError(f"No default beta registered for spec '{spec_name}'")
    return DEFAULT_BETA_BY_SPEC[spec_name].copy()


def default_delta_for_spec(spec_name: str) -> float:
    return float(DEFAULT_DELTA_BY_SPEC.get(spec_name, 0.0))


@dataclass
class TaskBDGP:
    # Observed sample size
    N_obs: int = 100
    J: int = 3

    # Valuation + due diligence
    gamma: float = 1.3
    sigma_nu: float = 0.2
    sigma_eta: float = 0.1

    # Misreporting
    kappa: float = float(np.log(1.5))
    misreporting_mode: MisreportingMode = "scale"

    # Cutoff process
    spec_name: str = "cand1_type_spr"
    beta: np.ndarray = None  # set in __post_init__
    sigma_omega: float = 0.1

    # Types
    p_S: float = 0.5  # iid strategic probability

    # Exogenous shifter (Z_i ~ N(0, sigma_Z^2))
    sigma_Z: float = 1.0
    psi_Z_val: float = 0.0  # if nonzero, Z shifts valuations

    # Candidate 3: bidder-adjusted admission
    delta: Optional[float] = None

    # Candidate 4: rounding precision proxy
    tick_F: Tuple[float, float] = (0.01, 0.02)
    tick_S: Tuple[float, float] = (0.02, 0.05)
    apply_rounding: bool = True

    def __post_init__(self) -> None:
        if self.J < 2:
            raise ValueError("J must be >= 2")
        if self.N_obs <= 0:
            raise ValueError("N_obs must be positive")
        if self.sigma_nu <= 0 or self.sigma_eta <= 0:
            raise ValueError("sigma_nu and sigma_eta must be positive")
        if self.sigma_omega < 0:
            raise ValueError("sigma_omega must be nonnegative")
        if not (0.0 <= self.p_S <= 1.0):
            raise ValueError("p_S must be in [0,1]")
        if self.spec_name not in TASKB_SPECS:
            raise ValueError(f"Unknown spec_name '{self.spec_name}'")

        spec = TASKB_SPECS[self.spec_name]
        if self.beta is None:
            self.beta = default_beta_for_spec(self.spec_name)
        else:
            self.beta = np.asarray(self.beta, dtype=float)
        if self.beta.shape != (spec.k_beta,):
            raise ValueError(f"beta has shape {self.beta.shape}, expected ({spec.k_beta},) for spec '{self.spec_name}'")

        if self.delta is None:
            self.delta = default_delta_for_spec(self.spec_name)


class TaskBDataGenerator:
    def __init__(self, dgp: TaskBDGP):
        self.dgp = dgp

    def generate(self) -> Tuple[List[TaskBAuctionObs], Dict]:
        auctions: List[TaskBAuctionObs] = []

        n_initiated = 0
        n_dropped_all_reject = 0
        n_all_admitted = 0
        n_two_sided = 0
        n_one_sided = 0

        spec_name = str(self.dgp.spec_name)
        alpha_F = mean_log_uniform(*self.dgp.tick_F)
        alpha_S = mean_log_uniform(*self.dgp.tick_S)

        while len(auctions) < self.dgp.N_obs:
            n_initiated += 1
            n = int(self.dgp.J)
            lam_f = lambda_f_first_price(n)
            lam_i = informal_bid_multiplier(n, float(self.dgp.kappa), mode=self.dgp.misreporting_mode)

            Z_i = float(np.random.normal(0.0, float(self.dgp.sigma_Z)))
            T = (np.random.rand(n) < float(self.dgp.p_S)).astype(np.int8)  # 0=F,1=S

            v = float(self.dgp.gamma) + float(self.dgp.psi_Z_val) * Z_i + np.random.normal(0.0, float(self.dgp.sigma_nu), size=n)
            bI_true = lam_i * v

            tick = np.empty(n, dtype=float)
            loF, hiF = self.dgp.tick_F
            loS, hiS = self.dgp.tick_S
            for j in range(n):
                tick[j] = float(np.random.uniform(loS, hiS)) if int(T[j]) == 1 else float(np.random.uniform(loF, hiF))

            bI_obs = bI_true.copy()
            if bool(self.dgp.apply_rounding):
                bI_obs = _round_to_tick(bI_obs, tick)

            X_i = _compute_X_i(
                spec_name,
                bI_obs=bI_obs,
                T=T,
                Z=Z_i,
                tick=tick,
                tick_alpha_F=alpha_F,
                tick_alpha_S=alpha_S,
            )

            b_star_i = float(X_i @ self.dgp.beta + np.random.normal(0.0, float(self.dgp.sigma_omega)))

            if spec_name == "cand3_type_shift_admission":
                admitted = bI_obs >= (b_star_i + float(self.dgp.delta) * T.astype(float))
            else:
                admitted = bI_obs >= b_star_i

            n_adm = int(np.sum(admitted))
            n_rej = n - n_adm

            if n_adm == 0:
                n_dropped_all_reject += 1
                continue

            if n_rej == 0:
                n_all_admitted += 1
                n_one_sided += 1
            else:
                n_two_sided += 1

            bF = np.full(n, np.nan, dtype=float)
            eta = np.random.normal(0.0, float(self.dgp.sigma_eta), size=n_adm)
            u_adm = v[admitted] + eta
            bF[admitted] = lam_f * u_adm

            auctions.append(
                TaskBAuctionObs(
                    auction_id=len(auctions),
                    bI=bI_obs.astype(float),
                    admitted=admitted.astype(bool),
                    bF=bF.astype(float),
                    T=T.astype(np.int8),
                    Z=Z_i,
                    tick=tick.astype(float),
                    tick_alpha_F=float(alpha_F),
                    tick_alpha_S=float(alpha_S),
                )
            )

        summary = {
            "n_observed": len(auctions),
            "n_initiated": n_initiated,
            "n_dropped_all_reject": n_dropped_all_reject,
            "keep_rate_pct": 100.0 * len(auctions) / float(n_initiated) if n_initiated else 0.0,
            "n_all_admitted": n_all_admitted,
            "n_two_sided": n_two_sided,
            "n_one_sided": n_one_sided,
        }
        return auctions, summary
