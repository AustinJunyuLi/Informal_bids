"""
Data classes and generators for auction simulations.

This module contains:
- DGP parameter dataclasses for Task A (single cutoff) and Task B (type-specific)
- Auction data dataclasses
- Data generators for simulation

The data classes are designed to be data-source agnostic - samplers work with
AuctionData objects regardless of whether they came from simulation or real files.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np

from .utils import draw_covariates, compute_mean_x


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
    """Data Generating Process parameters for Task B (type-specific cutoffs).

    Attributes:
        N: Number of auctions
        J: Number of bidders per auction
        mu_v: Mean valuation
        sigma_v: Std dev of valuation shock
        b_star_S: True cutoff for type S (intercept)
        b_star_F: True cutoff for type F (intercept)
        prob_type_S: Probability bidder is type S
        beta_S: Coefficients for type S cutoff
        beta_F: Coefficients for type F cutoff
        sigma_b_S: Std dev of type S cutoff shocks
        sigma_b_F: Std dev of type F cutoff shocks
        x_mean: Mean of non-intercept covariates
        x_std: Std dev of non-intercept covariates
    """
    N: int
    J: int
    mu_v: float
    sigma_v: float
    b_star_S: float
    b_star_F: float
    prob_type_S: float = 0.5
    beta_S: Optional[np.ndarray] = None
    beta_F: Optional[np.ndarray] = None
    sigma_b_S: float = 0.0
    sigma_b_F: float = 0.0
    x_mean: float = 0.0
    x_std: float = 1.0

    def __post_init__(self):
        if self.beta_S is None:
            self.beta_S = np.array([self.b_star_S], dtype=float)
        else:
            self.beta_S = np.atleast_1d(self.beta_S).astype(float)
        if self.beta_F is None:
            self.beta_F = np.array([self.b_star_F], dtype=float)
        else:
            self.beta_F = np.atleast_1d(self.beta_F).astype(float)

        if self.beta_S.ndim != 1 or self.beta_F.ndim != 1:
            raise ValueError("beta_S and beta_F must be 1D arrays or scalars")
        if self.beta_S.shape[0] != self.beta_F.shape[0]:
            raise ValueError("beta_S and beta_F must have same length")
        if self.beta_S.shape[0] == 1:
            self.b_star_S = float(self.beta_S[0])
            self.b_star_F = float(self.beta_F[0])

    @property
    def k_covariates(self) -> int:
        return int(self.beta_S.shape[0])

    def mean_x(self) -> np.ndarray:
        return compute_mean_x(self.k_covariates, self.x_mean)

    def cutoff_at_mean_x(self) -> Tuple[float, float]:
        x_bar = self.mean_x()
        return float(self.beta_S @ x_bar), float(self.beta_F @ x_bar)

    def __repr__(self):
        return (f"DGP(N={self.N}, J={self.J}, mu_v={self.mu_v}, "
                f"sigma_v={self.sigma_v}, beta_S={self.beta_S}, beta_F={self.beta_F})")


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

    # Task B priors (type-specific)
    mu_S_prior_mean: float = 1.45
    mu_S_prior_std: float = 0.3
    mu_F_prior_mean: float = 1.35
    mu_F_prior_std: float = 0.3
    beta_S_prior_mean: Optional[np.ndarray] = None
    beta_S_prior_std: Optional[np.ndarray] = None
    beta_F_prior_mean: Optional[np.ndarray] = None
    beta_F_prior_std: Optional[np.ndarray] = None

    # Variance priors (shared)
    sigma_prior_a: float = 2.0
    sigma_prior_b: float = 0.1


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
    """Individual bidder data for Task B."""
    bidder_type: str  # 'S' or 'F'
    bid: float
    admitted: bool


@dataclass
class TaskBAuctionData:
    """Auction with typed bidders (Task B: type-specific cutoffs).

    This class is data-source agnostic - it can be populated from
    simulation (DataGenerator) or real files (RealDataLoader).
    """
    auction_id: int
    X_i: np.ndarray  # Covariates
    bidders: List[BidderData]
    L_S: float  # Max rejected S-bid
    U_S: float  # Min admitted S-bid
    L_F: float  # Max rejected F-bid
    U_F: float  # Min admitted F-bid
    has_S_bounds: bool  # True if L_S and U_S are both finite
    has_F_bounds: bool  # True if L_F and U_F are both finite


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
    """Generate auction data with typed bidders for Task B."""

    def __init__(self, params: TaskBDGPParameters):
        self.params = params

    def _draw_covariates(self) -> np.ndarray:
        return draw_covariates(self.params.k_covariates,
                               self.params.x_mean, self.params.x_std)

    def generate_auction_data(self) -> Tuple[List[TaskBAuctionData], Dict]:
        """Generate N_observed auctions with typed bidders.

        We interpret `N` as the number of auctions observed at the formal stage,
        i.e. auctions with at least one admitted bidder overall. Auctions with
        zero admitted overall are generated but excluded from the returned sample.

        Returns:
            Tuple of (list of observed TaskBAuctionData, summary dict)
        """
        auctions: List[TaskBAuctionData] = []

        n_initiated = 0
        n_dropped_no_admitted = 0

        # Counts among observed auctions
        n_incomplete_S = 0
        n_incomplete_F = 0

        while len(auctions) < self.params.N:
            n_initiated += 1
            bidders: List[BidderData] = []

            X_i = self._draw_covariates()
            b_star_S_i = float(X_i @ self.params.beta_S + np.random.normal(0, self.params.sigma_b_S))
            b_star_F_i = float(X_i @ self.params.beta_F + np.random.normal(0, self.params.sigma_b_F))

            # Generate J bidders
            for _ in range(self.params.J):
                bidder_type = 'S' if np.random.rand() < self.params.prob_type_S else 'F'
                epsilon = np.random.normal(0, self.params.sigma_v)
                bid = float(self.params.mu_v + epsilon)

                if bidder_type == 'S':
                    admitted = bid >= b_star_S_i
                else:
                    admitted = bid >= b_star_F_i

                bidders.append(BidderData(bidder_type, bid, admitted))

            # Selection: auction observed iff at least one admitted overall
            if not any(b.admitted for b in bidders):
                n_dropped_no_admitted += 1
                continue

            # Compute type-specific interval bounds
            S_rejected = [b.bid for b in bidders if b.bidder_type == 'S' and not b.admitted]
            S_admitted = [b.bid for b in bidders if b.bidder_type == 'S' and b.admitted]
            F_rejected = [b.bid for b in bidders if b.bidder_type == 'F' and not b.admitted]
            F_admitted = [b.bid for b in bidders if b.bidder_type == 'F' and b.admitted]

            # Type S bounds
            if len(S_rejected) > 0 and len(S_admitted) > 0:
                L_S = float(np.max(S_rejected))
                U_S = float(np.min(S_admitted))
                has_S_bounds = True
            elif len(S_rejected) > 0:
                L_S = float(np.max(S_rejected))
                U_S = np.inf
                has_S_bounds = False
                n_incomplete_S += 1
            elif len(S_admitted) > 0:
                L_S = -np.inf
                U_S = float(np.min(S_admitted))
                has_S_bounds = False
                n_incomplete_S += 1
            else:
                # No type-S bidders in this auction
                L_S = -np.inf
                U_S = np.inf
                has_S_bounds = False

            # Type F bounds
            if len(F_rejected) > 0 and len(F_admitted) > 0:
                L_F = float(np.max(F_rejected))
                U_F = float(np.min(F_admitted))
                has_F_bounds = True
            elif len(F_rejected) > 0:
                L_F = float(np.max(F_rejected))
                U_F = np.inf
                has_F_bounds = False
                n_incomplete_F += 1
            elif len(F_admitted) > 0:
                L_F = -np.inf
                U_F = float(np.min(F_admitted))
                has_F_bounds = False
                n_incomplete_F += 1
            else:
                # No type-F bidders in this auction
                L_F = -np.inf
                U_F = np.inf
                has_F_bounds = False

            auction = TaskBAuctionData(
                auction_id=len(auctions),
                X_i=X_i,
                bidders=bidders,
                L_S=L_S,
                U_S=U_S,
                L_F=L_F,
                U_F=U_F,
                has_S_bounds=has_S_bounds,
                has_F_bounds=has_F_bounds,
            )
            auctions.append(auction)

        # Summary for observed auctions
        n_complete_both = sum(1 for a in auctions if a.has_S_bounds and a.has_F_bounds)
        n_complete_S = sum(1 for a in auctions if a.has_S_bounds)
        n_complete_F = sum(1 for a in auctions if a.has_F_bounds)

        summary = {
            'n_observed': len(auctions),
            'n_complete_both': n_complete_both,
            'n_complete_S_only': n_complete_S,
            'n_complete_F_only': n_complete_F,
            'n_incomplete_S': n_incomplete_S,
            'n_incomplete_F': n_incomplete_F,
            'n_initiated': n_initiated,
            'n_dropped_no_admitted': n_dropped_no_admitted,
            'keep_rate_pct': 100 * len(auctions) / n_initiated if n_initiated else 0,
        }

        return auctions, summary
