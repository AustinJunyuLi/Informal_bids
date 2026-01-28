"""
Data classes and generators for auction simulations.

This module contains:
- DGP parameter dataclasses for Task A (single cutoff) and Task B (two-stage DGP)
- Auction data dataclasses
- Data generators for simulation

The data classes are designed to be data-source agnostic - samplers work with
AuctionData objects regardless of whether they came from simulation or real files.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np

from .utils import (
    draw_covariates,
    compute_mean_x,
    MisreportingMode,
    CutoffSpec,
    cutoff_expected_k,
    compute_cutoff_features,
    informal_bid_multiplier,
    lambda_f_first_price,
)


# =============================================================================
# DGP Parameters
# =============================================================================

@dataclass
class TaskADGPParameters:
    """Data Generating Process parameters for Task A (single cutoff).

    Attributes:
        N: Number of auctions
        J: Number of bidders per auction
        mu_v: Mean valuation
        sigma_v: Std dev of valuation shock
        b_star: Baseline admission cutoff (intercept)
        beta: Auction covariates coefficients (including intercept)
        sigma_b: Std dev of auction-level cutoff shocks
        x_mean: Mean of non-intercept covariates
        x_std: Std dev of non-intercept covariates
    """
    N: int
    J: int
    mu_v: float
    sigma_v: float
    b_star: float
    beta: Optional[np.ndarray] = None
    sigma_b: float = 0.0
    x_mean: float = 0.0
    x_std: float = 1.0

    def __post_init__(self):
        if self.beta is None:
            self.beta = np.array([self.b_star], dtype=float)
        else:
            self.beta = np.atleast_1d(self.beta).astype(float)
        if self.beta.ndim != 1:
            raise ValueError("beta must be a 1D array or scalar")
        if self.beta.shape[0] == 1:
            self.b_star = float(self.beta[0])

    @property
    def k_covariates(self) -> int:
        return int(self.beta.shape[0])

    def mean_x(self) -> np.ndarray:
        return compute_mean_x(self.k_covariates, self.x_mean)

    def cutoff_at_mean_x(self) -> float:
        return float(self.beta @ self.mean_x())

    def __repr__(self):
        return (f"DGP(N={self.N}, J={self.J}, mu_v={self.mu_v}, "
                f"sigma_v={self.sigma_v}, beta={self.beta})")


@dataclass
class TaskBDGPParameters:
    """Data Generating Process parameters for Task B (two-stage DGP).

    Attributes:
        N: Number of observed auctions (reach formal stage, S=1)
        J: Number of bidders per auction (fixed for now; later allow sensitivity)
        gamma: Mean valuation in v_ij = gamma + nu_ij
        sigma_nu: Std dev of valuation shock nu_ij
        sigma_eta: Std dev of due diligence shock eta_ij
        kappa: Unconstrained misreporting parameter with 1 + tilde_alpha = exp(kappa)
        beta_cutoff: Cutoff-process coefficients (intercept first). Intercept-only baseline
            uses beta_cutoff = [c]. Later versions can include moments of informal bids.
        sigma_omega: Std dev of cutoff shock omega_i
    """
    N: int
    J: int
    gamma: float
    sigma_nu: float
    sigma_eta: float
    kappa: float = 0.0
    misreporting_mode: MisreportingMode = "scale"
    cutoff_spec: Optional[CutoffSpec] = None
    beta_cutoff: Optional[np.ndarray] = None
    sigma_omega: float = 0.0

    def __post_init__(self):
        if self.J <= 1:
            raise ValueError("J must be >= 2 (first-price shading uses 1 - 1/J)")
        if self.sigma_nu <= 0:
            raise ValueError("sigma_nu must be positive")
        if self.sigma_eta <= 0:
            raise ValueError("sigma_eta must be positive")

        if self.misreporting_mode not in ("scale", "shift"):
            raise ValueError("misreporting_mode must be 'scale' or 'shift'")

        if self.beta_cutoff is None:
            self.beta_cutoff = np.array([1.4], dtype=float)
        else:
            self.beta_cutoff = np.atleast_1d(self.beta_cutoff).astype(float)
        if self.beta_cutoff.ndim != 1:
            raise ValueError("beta_cutoff must be a 1D array or scalar")

        if self.cutoff_spec is None:
            if self.beta_cutoff.size == 1:
                self.cutoff_spec = "intercept"
            elif self.beta_cutoff.size == 4:
                self.cutoff_spec = "moments_k4"
            elif self.beta_cutoff.size == 3:
                self.cutoff_spec = "depth_k2"
            else:
                raise ValueError("cutoff_spec must be provided when beta_cutoff length is not 1, 3, or 4")
        else:
            if self.cutoff_spec not in ("intercept", "moments_k4", "depth_k2", "depth_k2_ratio"):
                raise ValueError("cutoff_spec must be one of: intercept, moments_k4, depth_k2, depth_k2_ratio")

        expected_k = cutoff_expected_k(self.cutoff_spec)
        if self.beta_cutoff.size != expected_k:
            raise ValueError(
                f"beta_cutoff length {self.beta_cutoff.size} does not match cutoff_spec "
                f"'{self.cutoff_spec}' (expected {expected_k})"
            )
        if expected_k > 1 and self.J < 3:
            raise ValueError("cutoff_spec with moments requires J >= 3")

    @property
    def k_covariates(self) -> int:
        return int(self.beta_cutoff.shape[0])

    @property
    def tilde_alpha(self) -> float:
        """Scale misreporting term: tilde_alpha = lambda_i / lambda_f - 1."""
        return float(self.lambda_i / self.lambda_f - 1.0)

    @property
    def alpha_additive(self) -> float:
        """Meeting-notes misreporting term alpha where lambda_i = lambda_f + alpha."""
        return float(self.lambda_i - self.lambda_f)

    @property
    def lambda_f(self) -> float:
        """Formal-stage shading multiplier (no misreporting)."""
        return float(lambda_f_first_price(self.J))

    @property
    def lambda_i(self) -> float:
        """Informal-stage multiplier lambda_I(J,kappa) under the configured misreporting_mode."""
        return float(informal_bid_multiplier(self.J, self.kappa, mode=self.misreporting_mode))

    def cutoff_at_intercept(self) -> float:
        """Cutoff at baseline covariates (intercept-only baseline)."""
        return float(self.beta_cutoff[0])

    def __repr__(self):
        return (
            "DGP("
            f"N={self.N}, J={self.J}, gamma={self.gamma}, sigma_nu={self.sigma_nu}, "
            f"sigma_eta={self.sigma_eta}, kappa={self.kappa}, misreporting_mode={self.misreporting_mode}, "
            f"cutoff_spec={self.cutoff_spec}, beta_cutoff={self.beta_cutoff}, "
            f"sigma_omega={self.sigma_omega}"
            ")"
        )


# =============================================================================
# MCMC Configuration
# =============================================================================

@dataclass
class MCMCConfig:
    """MCMC sampler configuration (shared by Task A and Task B)."""
    n_iterations: int = 20000
    burn_in: int = 10000
    thinning: int = 10
    n_chains: int = 3

    # Task A priors
    mu_prior_mean: float = 1.3
    mu_prior_std: float = 0.5
    beta_prior_mean: Optional[np.ndarray] = None
    beta_prior_std: Optional[np.ndarray] = None

    # Task B (two-stage) config and priors
    task_b_stage: int = 1  # 1: fix (sigma_nu, sigma_eta), 2: estimate sigma_eta, 3: estimate sigma_nu + sigma_eta

    # Task B misreporting parameterization (must match the DGP used to generate the data)
    task_b_misreporting_mode: MisreportingMode = "scale"

    # Cutoff process priors. These may be scalars or 1D arrays:
    # - If k=1 (intercept-only), a scalar prior is used directly.
    # - If k>1 (moments cutoff), a scalar mean is treated as the intercept prior and
    #   remaining slopes default to 0.0 unless an explicit vector is provided.
    task_b_cutoff_prior_mean: float = 1.4
    task_b_cutoff_prior_std: float = 0.5

    task_b_gamma_prior_mean: float = 1.3
    task_b_gamma_prior_std: float = 0.5
    task_b_gamma_proposal_sd: float = 0.02

    # Misreporting option C: 1 + tilde_alpha = exp(kappa), kappa in R
    task_b_kappa_prior_mean: float = 0.0
    task_b_kappa_prior_std: float = 0.5
    task_b_kappa_init: float = 0.0
    task_b_kappa_proposal_sd: float = 0.1

    # Cutoff shock init and priors
    task_b_sigma_omega_init: float = 0.1
    task_b_sigma_omega_prior_a: float = 2.0
    task_b_sigma_omega_prior_b: float = 0.1

    # Staged valuation / due diligence variance handling
    task_b_sigma_nu_fixed: Optional[float] = None
    task_b_sigma_eta_fixed: Optional[float] = None
    task_b_sigma_nu_prior_a: float = 2.0
    task_b_sigma_nu_prior_b: float = 0.1
    task_b_sigma_eta_prior_a: float = 2.0
    task_b_sigma_eta_prior_b: float = 0.1

    # Variance priors (shared)
    sigma_prior_a: float = 2.0
    sigma_prior_b: float = 0.1

    # Performance
    task_b_use_numba: bool = True


# =============================================================================
# Auction Data Classes
# =============================================================================

@dataclass
class TaskAAuctionData:
    """Data for a single auction (Task A: single cutoff).

    This class is data-source agnostic - it can be populated from
    simulation (DataGenerator) or real files (RealDataLoader).
    """
    auction_id: int
    X_i: np.ndarray  # Covariates
    bids: np.ndarray  # All bids in the auction
    admitted: np.ndarray  # Boolean mask for admitted bids
    L_i: float  # Lower bound (max rejected bid)
    U_i: float  # Upper bound (min admitted bid)
    is_complete: bool  # True if both L_i and U_i are finite

    def __repr__(self):
        return (f"Auction {self.auction_id}: "
                f"[{self.L_i:.3f}, {self.U_i:.3f}], complete={self.is_complete}")


@dataclass
class BidderData:
    """Deprecated legacy type kept for backward references (unused on this branch)."""

    bidder_type: str
    bid: float
    admitted: bool


@dataclass
class TaskBAuctionData:
    """Auction data for Task B (two-stage DGP).

    Observables per auction (conditional on reaching the formal stage, S=1):
    - Informal bids for all J bidders
    - Admission indicators, yielding bounds [L_i, U_i] for the latent cutoff b*_i
    - Formal bids for admitted bidders only (NaN for rejected bidders)
    """

    auction_id: int
    X_i: np.ndarray  # Cutoff covariates (intercept-only baseline; later add moments)
    informal_bids: np.ndarray  # Shape (J,)
    admitted: np.ndarray  # Shape (J,), boolean
    formal_bids: np.ndarray  # Shape (J,), NaN for non-admitted bidders
    L_i: float
    U_i: float
    is_complete: bool  # True if both L_i and U_i are finite
    n_bidders: int


# =============================================================================
# Data Generators (for simulation)
# =============================================================================

class TaskADataGenerator:
    """Generate synthetic auction data for Task A (single cutoff)."""

    def __init__(self, params: TaskADGPParameters):
        self.params = params

    def _draw_covariates(self) -> np.ndarray:
        return draw_covariates(self.params.k_covariates,
                               self.params.x_mean, self.params.x_std)

    def generate_auction_data(self) -> Tuple[List[TaskAAuctionData], Dict]:
        """Generate N_observed auctions with J bidders each.

        In this simulation exercise we interpret `N` as the number of auctions
        observed at the *formal stage* (i.e., at least one bidder is admitted).
        Auctions with zero admitted bids ("all-reject") are not observed and are
        therefore generated but excluded from the returned sample.

        Returns:
            Tuple of (list of observed TaskAAuctionData, summary dict)
        """
        auctions: List[TaskAAuctionData] = []

        n_initiated = 0
        n_dropped_all_reject = 0

        # Among observed auctions
        n_incomplete = 0
        n_all_admitted = 0
        n_two_sided = 0

        while len(auctions) < self.params.N:
            n_initiated += 1

            # Generate valuations: v_ij = mu_v + epsilon_ij
            epsilon = np.random.normal(0, self.params.sigma_v, self.params.J)
            valuations = self.params.mu_v + epsilon

            # Informal bids equal valuations
            bids = valuations.copy()

            # Auction covariates and cutoff
            X_i = self._draw_covariates()
            b_star_i = float(X_i @ self.params.beta + np.random.normal(0, self.params.sigma_b))

            # Apply admission rule: j in A_i iff b_ij >= b_star
            admitted = bids >= b_star_i
            n_admitted = int(admitted.sum())
            n_rejected = int((~admitted).sum())

            # Selection: auction observed iff at least one admitted
            if n_admitted == 0:
                n_dropped_all_reject += 1
                continue

            # Compute interval bounds
            if n_rejected == 0:
                # All admitted: one-sided upper bound
                L_i = -np.inf
                U_i = float(np.min(bids))
                n_all_admitted += 1
                is_complete = False
            else:
                # Two-sided bounds
                rejected_bids = bids[~admitted]
                admitted_bids = bids[admitted]
                L_i = float(np.max(rejected_bids))
                U_i = float(np.min(admitted_bids))
                n_two_sided += 1
                is_complete = True

            if not is_complete:
                n_incomplete += 1

            auction = TaskAAuctionData(
                auction_id=len(auctions),
                X_i=X_i,
                bids=bids,
                admitted=admitted,
                L_i=L_i,
                U_i=U_i,
                is_complete=is_complete,
            )
            auctions.append(auction)

        summary = {
            # Observed sample
            'n_observed': len(auctions),
            'n_complete': n_two_sided,
            'n_incomplete': n_incomplete,
            'n_all_admitted': n_all_admitted,
            'pct_incomplete': 100 * n_incomplete / len(auctions) if auctions else 0,
            # Initiated sample (including unobserved all-reject)
            'n_initiated': n_initiated,
            'n_dropped_all_reject': n_dropped_all_reject,
            'keep_rate_pct': 100 * len(auctions) / n_initiated if n_initiated else 0,
        }

        return auctions, summary


class TaskBDataGenerator:
    """Generate auction data for Task B (two-stage DGP)."""

    def __init__(self, params: TaskBDGPParameters):
        self.params = params

    def _cutoff_covariates(self, informal_bids: np.ndarray) -> np.ndarray:
        """Construct cutoff covariates X_i from informal bids based on cutoff_spec."""
        return compute_cutoff_features(informal_bids, self.params.cutoff_spec)

    def generate_auction_data(self) -> Tuple[List[TaskBAuctionData], Dict]:
        """Generate N observed auctions (conditional on reaching the formal stage)."""
        auctions: List[TaskBAuctionData] = []

        n_initiated = 0
        n_dropped_all_reject = 0

        n_incomplete = 0
        n_all_admitted = 0
        n_two_sided = 0

        while len(auctions) < self.params.N:
            n_initiated += 1
            n_bidders = int(self.params.J)
            lambda_f = float(lambda_f_first_price(n_bidders))
            lambda_i = float(informal_bid_multiplier(n_bidders, float(self.params.kappa), mode=self.params.misreporting_mode))

            # Draw valuations and informal bids
            v = self.params.gamma + np.random.normal(0.0, self.params.sigma_nu, size=n_bidders)
            informal_bids = lambda_i * v

            # Cutoff covariates (optionally moments of informal bids)
            X_i = self._cutoff_covariates(informal_bids)

            # Draw cutoff and apply admission
            b_star_i = float(X_i @ self.params.beta_cutoff + np.random.normal(0.0, self.params.sigma_omega))
            admitted = informal_bids >= b_star_i
            n_admitted = int(admitted.sum())
            n_rejected = n_bidders - n_admitted

            # Selection: auction observed iff at least one admitted bidder
            if n_admitted == 0:
                n_dropped_all_reject += 1
                continue

            # Formal bids for admitted bidders
            formal_bids = np.full(n_bidders, np.nan, dtype=float)
            eta = np.random.normal(0.0, self.params.sigma_eta, size=n_admitted)
            u_adm = v[admitted] + eta
            formal_bids[admitted] = lambda_f * u_adm

            # Bounds from observed admission indicators
            if n_rejected == 0:
                L_i = -np.inf
                U_i = float(np.min(informal_bids))
                is_complete = False
                n_all_admitted += 1
                n_incomplete += 1
            else:
                L_i = float(np.max(informal_bids[~admitted]))
                U_i = float(np.min(informal_bids[admitted]))
                is_complete = True
                n_two_sided += 1

            auctions.append(
                TaskBAuctionData(
                    auction_id=len(auctions),
                    X_i=X_i,
                    informal_bids=informal_bids.astype(float),
                    admitted=admitted.astype(bool),
                    formal_bids=formal_bids.astype(float),
                    L_i=L_i,
                    U_i=U_i,
                    is_complete=is_complete,
                    n_bidders=n_bidders,
                )
            )

        summary = {
            'n_observed': len(auctions),
            'n_complete': n_two_sided,
            'n_incomplete': n_incomplete,
            'n_all_admitted': n_all_admitted,
            'pct_incomplete': 100 * n_incomplete / len(auctions) if auctions else 0,
            'n_initiated': n_initiated,
            'n_dropped_all_reject': n_dropped_all_reject,
            'keep_rate_pct': 100 * len(auctions) / n_initiated if n_initiated else 0,
        }

        return auctions, summary
