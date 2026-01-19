"""
MCMC samplers for auction cutoff estimation.

This module contains the MCMC sampler implementations for:
- Task A: Single constant cutoff
- Task B: Type-specific cutoffs (S and F)

The samplers are data-source agnostic - they work with AuctionData objects
regardless of whether they came from simulation or real files.
"""

import numpy as np
from scipy.stats import invgamma
from typing import List, Dict, Tuple, Optional

from .data import TaskAAuctionData, TaskBAuctionData, MCMCConfig
from .utils import (
    sample_truncated_normal,
    gelman_rubin,
    selection_prob_at_least_one_exceeds_cutoff,
)

# Numba acceleration (optional)
try:
    from .numba_kernels import HAS_NUMBA, task_a_run_chain_intercept, task_b_run_chain_intercept
except Exception:
    HAS_NUMBA = False
    task_a_run_chain_intercept = None
    task_b_run_chain_intercept = None


class TaskAMCMCSampler:
    """MCMC sampler for Task A (single constant cutoff).

    Model:
        b^{I*}_i = mu + nu_i
        nu_i ~ N(0, sigma^2)
        nu_i in [L_i - mu, U_i - mu]

    Uses data augmentation with a selection-aware cutoff update.
    """

    def __init__(
        self,
        auctions: List[TaskAAuctionData],
        config: MCMCConfig,
        *,
        bid_mu: float,
        bid_sigma: float,
    ):
        self.auctions = auctions
        self.config = config
        self.bid_mu = float(bid_mu)
        self.bid_sigma = float(bid_sigma)

        # Use auctions observed at the formal stage (at least one admitted).
        # For Task A, this corresponds to auctions with a finite upper bound U_i.
        self.working_auctions = [a for a in auctions if np.isfinite(a.U_i)]
        self.N = len(self.working_auctions)

        self.n_dropped_all_reject = sum(1 for a in auctions if not np.isfinite(a.U_i))
        self.n_two_sided = sum(1 for a in self.working_auctions if np.isfinite(a.L_i))
        self.n_one_sided_upper = self.N - self.n_two_sided

        if self.N > 0:
            self.k = int(self.working_auctions[0].X_i.shape[0])
            self.X = np.vstack([a.X_i for a in self.working_auctions])
            self.L = np.array([a.L_i for a in self.working_auctions], dtype=np.float64)
            self.U = np.array([a.U_i for a in self.working_auctions], dtype=np.float64)
            self.n_bidders = np.array([len(a.bids) for a in self.working_auctions], dtype=np.int64)
        elif auctions:
            self.k = int(auctions[0].X_i.shape[0])
            self.X = np.empty((0, self.k))
            self.L = np.empty((0,), dtype=np.float64)
            self.U = np.empty((0,), dtype=np.float64)
            self.n_bidders = np.empty((0,), dtype=np.int64)
        else:
            self.k = 1
            self.X = np.empty((0, self.k))
            self.L = np.empty((0,), dtype=np.float64)
            self.U = np.empty((0,), dtype=np.float64)
            self.n_bidders = np.empty((0,), dtype=np.int64)

        print(
            f"MCMC (selection-aware) using {self.N} observed auctions "
            f"(two-sided: {self.n_two_sided}, one-sided upper: {self.n_one_sided_upper}); "
            f"dropped {self.n_dropped_all_reject} all-reject"
        )

    def _get_beta_prior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get prior mean and std for beta coefficients."""
        mean = (self.config.beta_prior_mean
                if self.config.beta_prior_mean is not None
                else self.config.mu_prior_mean)
        std = (self.config.beta_prior_std
               if self.config.beta_prior_std is not None
               else self.config.mu_prior_std)
        mean = np.atleast_1d(mean).astype(float)
        std = np.atleast_1d(std).astype(float)
        if mean.size == 1:
            mean = np.full(self.k, mean.item())
        if std.size == 1:
            std = np.full(self.k, std.item())
        if mean.size != self.k or std.size != self.k:
            raise ValueError("beta prior has wrong length")
        return mean, std

    def run_chain(self, chain_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single MCMC chain.

        Returns:
            Tuple of (beta_chain, sigma_chain)
        """
        print(f"\nRunning chain {chain_id}...", flush=True)

        # Initialize
        all_bids = np.concatenate([a.bids for a in self.working_auctions
                                   if len(a.bids) > 0])
        beta = np.zeros(self.k)
        beta[0] = np.median(all_bids) if len(all_bids) > 0 else 1.3
        sigma = 0.1

        beta_chain = np.zeros((self.config.n_iterations, self.k))
        sigma_chain = np.zeros(self.config.n_iterations)

        # Priors
        beta_prior_mean, beta_prior_std = self._get_beta_prior()
        V0_inv = np.diag(1.0 / (beta_prior_std ** 2))
        a_prior = self.config.sigma_prior_a
        b_prior = self.config.sigma_prior_b

        # Use Numba-accelerated kernel for intercept-only case
        if HAS_NUMBA and self.k == 1 and task_a_run_chain_intercept is not None:
            seed = int(np.random.randint(0, 2**31 - 1)) + 9973 * int(chain_id)
            mu_chain_1d, sigma_chain = task_a_run_chain_intercept(
                self.L,
                self.U,
                self.n_bidders,
                int(self.config.n_iterations),
                float(beta_prior_mean[0]), float(beta_prior_std[0]),
                float(a_prior), float(b_prior),
                float(beta[0]),
                float(sigma),
                float(self.bid_mu),
                float(self.bid_sigma),
                seed,
            )
            beta_chain = mu_chain_1d.reshape((-1, 1))
            print(f"Chain {chain_id} complete!", flush=True)
            return beta_chain, sigma_chain

        # Initialize latent cutoffs b*_i
        b_star = np.zeros(self.N, dtype=np.float64)
        for i, auction in enumerate(self.working_auctions):
            xb = float(auction.X_i @ beta)
            b_star[i] = sample_truncated_normal(xb, sigma, auction.L_i, auction.U_i)

        # Gibbs sampling
        for t in range(self.config.n_iterations):
            # Step 1: Sample latent b*_i via independence MH with selection correction.
            # Proposal: naive truncated normal (the previous baseline).
            accept_count = 0
            for i, auction in enumerate(self.working_auctions):
                xb = float(auction.X_i @ beta)
                b_prop = sample_truncated_normal(xb, sigma, auction.L_i, auction.U_i)

                p_old = selection_prob_at_least_one_exceeds_cutoff(
                    b_star[i], self.bid_mu, self.bid_sigma, int(self.n_bidders[i])
                )
                p_prop = selection_prob_at_least_one_exceeds_cutoff(
                    b_prop, self.bid_mu, self.bid_sigma, int(self.n_bidders[i])
                )

                alpha = min(1.0, p_old / p_prop)
                if np.random.rand() < alpha:
                    b_star[i] = b_prop
                    accept_count += 1

            # Step 2: Update beta (conjugate normal update)
            XtX = self.X.T @ self.X
            V_post = np.linalg.inv(V0_inv + XtX / (sigma ** 2))
            beta_post = V_post @ (V0_inv @ beta_prior_mean +
                                  (self.X.T @ b_star) / (sigma ** 2))
            beta = np.random.multivariate_normal(beta_post, V_post)

            # Step 3: Update sigma^2 (conjugate inverse-gamma update)
            resid = b_star - (self.X @ beta)
            a_post = a_prior + self.N / 2
            b_post = b_prior + np.sum(resid ** 2) / 2
            sigma_sq = invgamma.rvs(a_post, scale=b_post)
            sigma = np.sqrt(sigma_sq)

            beta_chain[t] = beta
            sigma_chain[t] = sigma

            if (t + 1) % 5000 == 0:
                accept_rate = accept_count / self.N if self.N > 0 else 0.0
                print(f"  Iteration {t+1}/{self.config.n_iterations}, "
                      f"mu0={beta[0]:.3f}, sigma={sigma:.3f}, "
                      f"mh_accept={accept_rate:.2f}", flush=True)

        print(f"Chain {chain_id} complete!", flush=True)
        return beta_chain, sigma_chain

    def run(self) -> Dict:
        """Run MCMC with multiple chains.

        Returns:
            Dictionary with posterior samples and diagnostics
        """
        all_chains_beta = []
        all_chains_sigma = []

        for chain_id in range(self.config.n_chains):
            beta_chain, sigma_chain = self.run_chain(chain_id)
            all_chains_beta.append(beta_chain)
            all_chains_sigma.append(sigma_chain)

        # Apply burn-in and thinning
        beta_samples = all_chains_beta[0][self.config.burn_in::self.config.thinning]
        x_bar = self.X.mean(axis=0) if self.N > 0 else np.ones(self.k)
        mu_samples = beta_samples @ x_bar
        sigma_samples = all_chains_sigma[0][self.config.burn_in::self.config.thinning]

        # Compute Gelman-Rubin statistic
        rhat_beta = gelman_rubin([chain[self.config.burn_in:] for chain in all_chains_beta])
        rhat_sigma = gelman_rubin([chain[self.config.burn_in:] for chain in all_chains_sigma])

        return {
            'mu_samples': mu_samples,
            'beta_samples': beta_samples,
            'sigma_samples': sigma_samples,
            'all_chains_mu': [chain @ x_bar for chain in all_chains_beta],
            'all_chains_beta': all_chains_beta,
            'all_chains_sigma': all_chains_sigma,
            'rhat_mu': rhat_beta,
            'rhat_sigma': rhat_sigma
        }


class TaskBMCMCSampler:
    """MCMC sampler for Task B (type-specific cutoffs).

    Estimates separate cutoffs for Type S and Type F bidders.
    """

    def __init__(self, auctions: List[TaskBAuctionData], config: MCMCConfig):
        self.auctions = auctions
        self.config = config

        # Observed sample: drop auctions with zero admitted overall.
        # In Task B this corresponds to auctions where both U_S and U_F are infinite.
        self.observed_auctions = [a for a in auctions if (np.isfinite(a.U_S) or np.isfinite(a.U_F))]
        self.n_dropped_no_admitted = len(auctions) - len(self.observed_auctions)

        # For each type, include auctions that provide any information (one- or two-sided).
        self.S_auctions = [a for a in self.observed_auctions if not (np.isinf(a.L_S) and np.isinf(a.U_S))]
        self.F_auctions = [a for a in self.observed_auctions if not (np.isinf(a.L_F) and np.isinf(a.U_F))]

        self.N_S = len(self.S_auctions)
        self.N_F = len(self.F_auctions)

        if self.N_S > 0:
            self.k = int(self.S_auctions[0].X_i.shape[0])
        elif self.N_F > 0:
            self.k = int(self.F_auctions[0].X_i.shape[0])
        elif auctions:
            self.k = int(auctions[0].X_i.shape[0])
        else:
            self.k = 1

        self.X_S = (np.vstack([a.X_i for a in self.S_auctions])
                   if self.N_S > 0 else np.empty((0, self.k)))
        self.X_F = (np.vstack([a.X_i for a in self.F_auctions])
                   if self.N_F > 0 else np.empty((0, self.k)))

        self.L_S = np.array([a.L_S for a in self.S_auctions], dtype=np.float64)
        self.U_S = np.array([a.U_S for a in self.S_auctions], dtype=np.float64)
        self.L_F = np.array([a.L_F for a in self.F_auctions], dtype=np.float64)
        self.U_F = np.array([a.U_F for a in self.F_auctions], dtype=np.float64)

        if self.N_S > 0:
            self.x_bar = self.X_S.mean(axis=0)
        elif self.N_F > 0:
            self.x_bar = self.X_F.mean(axis=0)
        else:
            self.x_bar = np.ones(self.k)

        print(
            f"MCMC using {self.N_S} auctions with S information, {self.N_F} with F information; "
            f"dropped {self.n_dropped_no_admitted} auctions with zero admitted overall"
        )

    def _get_beta_prior(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get prior mean and std for beta coefficients."""
        if which == 'S':
            mean = (self.config.beta_S_prior_mean
                    if self.config.beta_S_prior_mean is not None
                    else self.config.mu_S_prior_mean)
            std = (self.config.beta_S_prior_std
                   if self.config.beta_S_prior_std is not None
                   else self.config.mu_S_prior_std)
        else:
            mean = (self.config.beta_F_prior_mean
                    if self.config.beta_F_prior_mean is not None
                    else self.config.mu_F_prior_mean)
            std = (self.config.beta_F_prior_std
                   if self.config.beta_F_prior_std is not None
                   else self.config.mu_F_prior_std)
        mean = np.atleast_1d(mean).astype(float)
        std = np.atleast_1d(std).astype(float)
        if mean.size == 1:
            mean = np.full(self.k, mean.item())
        if std.size == 1:
            std = np.full(self.k, std.item())
        if mean.size != self.k or std.size != self.k:
            raise ValueError("beta prior has wrong length")
        return mean, std

    def run_chain(self, chain_id: int = 0) -> Dict:
        """Run single MCMC chain.

        Returns:
            Dictionary with chain samples for all parameters
        """
        print(f"\nRunning chain {chain_id}...", flush=True)

        # Initialize
        beta_S = np.zeros(self.k)
        beta_S[0] = self.config.mu_S_prior_mean
        beta_F = np.zeros(self.k)
        beta_F[0] = self.config.mu_F_prior_mean
        sigma_S = 0.1
        sigma_F = 0.1

        # Storage
        beta_S_chain = np.zeros((self.config.n_iterations, self.k))
        beta_F_chain = np.zeros((self.config.n_iterations, self.k))
        mu_S_chain = np.zeros(self.config.n_iterations)
        mu_F_chain = np.zeros(self.config.n_iterations)
        sigma_S_chain = np.zeros(self.config.n_iterations)
        sigma_F_chain = np.zeros(self.config.n_iterations)

        # Priors
        beta_S_prior_mean, beta_S_prior_std = self._get_beta_prior('S')
        beta_F_prior_mean, beta_F_prior_std = self._get_beta_prior('F')
        V0_S_inv = np.diag(1.0 / (beta_S_prior_std ** 2))
        V0_F_inv = np.diag(1.0 / (beta_F_prior_std ** 2))
        a_prior = self.config.sigma_prior_a
        b_prior = self.config.sigma_prior_b

        # Use Numba-accelerated kernel for intercept-only case
        if HAS_NUMBA and self.k == 1 and task_b_run_chain_intercept is not None:
            seed = int(np.random.randint(0, 2**31 - 1)) + 7919 * int(chain_id)
            mu_S_chain, mu_F_chain, sigma_S_chain, sigma_F_chain = task_b_run_chain_intercept(
                self.L_S, self.U_S, self.L_F, self.U_F,
                int(self.config.n_iterations),
                float(self.config.mu_S_prior_mean), float(self.config.mu_S_prior_std),
                float(self.config.mu_F_prior_mean), float(self.config.mu_F_prior_std),
                float(a_prior), float(b_prior),
                float(beta_S[0]), float(beta_F[0]),
                float(sigma_S), float(sigma_F), seed,
            )
            beta_S_chain = mu_S_chain.reshape((-1, 1))
            beta_F_chain = mu_F_chain.reshape((-1, 1))
            print(f"Chain {chain_id} complete!", flush=True)
            return {
                'beta_S': beta_S_chain,
                'beta_F': beta_F_chain,
                'mu_S': mu_S_chain,
                'mu_F': mu_F_chain,
                'sigma_S': sigma_S_chain,
                'sigma_F': sigma_F_chain,
            }

        # Gibbs sampling
        for t in range(self.config.n_iterations):
            # Step 1: Sample latent thresholds for type S
            nu_S_samples = np.zeros(self.N_S)
            b_star_S_samples = np.zeros(self.N_S)

            for i, auction in enumerate(self.S_auctions):
                xb_S = float(auction.X_i @ beta_S)
                lower = auction.L_S - xb_S
                upper = auction.U_S - xb_S
                nu_S = sample_truncated_normal(0, sigma_S, lower, upper)
                nu_S_samples[i] = nu_S
                b_star_S_samples[i] = xb_S + nu_S

            # Step 2: Update beta_S
            XtX_S = self.X_S.T @ self.X_S
            V_S_post = np.linalg.inv(V0_S_inv + XtX_S / (sigma_S ** 2))
            beta_S_post = V_S_post @ (V0_S_inv @ beta_S_prior_mean +
                                      (self.X_S.T @ b_star_S_samples) / (sigma_S ** 2))
            beta_S = np.random.multivariate_normal(beta_S_post, V_S_post)

            # Step 3: Update sigma_S^2
            a_S_post = a_prior + self.N_S / 2
            b_S_post = b_prior + np.sum(nu_S_samples ** 2) / 2
            sigma_S_sq = invgamma.rvs(a_S_post, scale=b_S_post)
            sigma_S = np.sqrt(sigma_S_sq)

            # Step 4: Sample latent thresholds for type F
            nu_F_samples = np.zeros(self.N_F)
            b_star_F_samples = np.zeros(self.N_F)

            for i, auction in enumerate(self.F_auctions):
                xb_F = float(auction.X_i @ beta_F)
                lower = auction.L_F - xb_F
                upper = auction.U_F - xb_F
                nu_F = sample_truncated_normal(0, sigma_F, lower, upper)
                nu_F_samples[i] = nu_F
                b_star_F_samples[i] = xb_F + nu_F

            # Step 5: Update beta_F
            XtX_F = self.X_F.T @ self.X_F
            V_F_post = np.linalg.inv(V0_F_inv + XtX_F / (sigma_F ** 2))
            beta_F_post = V_F_post @ (V0_F_inv @ beta_F_prior_mean +
                                      (self.X_F.T @ b_star_F_samples) / (sigma_F ** 2))
            beta_F = np.random.multivariate_normal(beta_F_post, V_F_post)

            # Step 6: Update sigma_F^2
            a_F_post = a_prior + self.N_F / 2
            b_F_post = b_prior + np.sum(nu_F_samples ** 2) / 2
            sigma_F_sq = invgamma.rvs(a_F_post, scale=b_F_post)
            sigma_F = np.sqrt(sigma_F_sq)

            # Store
            beta_S_chain[t] = beta_S
            beta_F_chain[t] = beta_F
            mu_S_chain[t] = float(beta_S @ self.x_bar)
            mu_F_chain[t] = float(beta_F @ self.x_bar)
            sigma_S_chain[t] = sigma_S
            sigma_F_chain[t] = sigma_F

            if (t + 1) % 5000 == 0:
                print(f"  Iter {t+1}: mu_S0={beta_S[0]:.3f}, mu_F0={beta_F[0]:.3f}, "
                      f"Delta={beta_S[0] - beta_F[0]:.3f}", flush=True)

        print(f"Chain {chain_id} complete!", flush=True)
        return {
            'beta_S': beta_S_chain,
            'beta_F': beta_F_chain,
            'mu_S': mu_S_chain,
            'mu_F': mu_F_chain,
            'sigma_S': sigma_S_chain,
            'sigma_F': sigma_F_chain
        }

    def run(self) -> Dict:
        """Run multiple chains.

        Returns:
            Dictionary with posterior samples and diagnostics
        """
        all_chains = []

        for chain_id in range(self.config.n_chains):
            chain_results = self.run_chain(chain_id)
            all_chains.append(chain_results)

        # Extract first chain for inference
        beta_S_samples = all_chains[0]['beta_S'][self.config.burn_in::self.config.thinning]
        beta_F_samples = all_chains[0]['beta_F'][self.config.burn_in::self.config.thinning]
        mu_S_samples = beta_S_samples @ self.x_bar
        mu_F_samples = beta_F_samples @ self.x_bar
        sigma_S_samples = all_chains[0]['sigma_S'][self.config.burn_in::self.config.thinning]
        sigma_F_samples = all_chains[0]['sigma_F'][self.config.burn_in::self.config.thinning]

        # Compute gap
        gap_samples = mu_S_samples - mu_F_samples

        # Gelman-Rubin
        rhat_mu_S = gelman_rubin([c['beta_S'][self.config.burn_in:] for c in all_chains])
        rhat_mu_F = gelman_rubin([c['beta_F'][self.config.burn_in:] for c in all_chains])

        return {
            'mu_S_samples': mu_S_samples,
            'mu_F_samples': mu_F_samples,
            'beta_S_samples': beta_S_samples,
            'beta_F_samples': beta_F_samples,
            'sigma_S_samples': sigma_S_samples,
            'sigma_F_samples': sigma_F_samples,
            'gap_samples': gap_samples,
            'all_chains': all_chains,
            'rhat_mu_S': rhat_mu_S,
            'rhat_mu_F': rhat_mu_F
        }
