"""
MCMC samplers for auction cutoff estimation.

This module contains the MCMC sampler implementations for:
- Task A: Single constant cutoff
- Task B: Two-stage DGP (informal + formal bids)

The samplers are data-source agnostic - they work with AuctionData objects
regardless of whether they came from simulation or real files.
"""

import numpy as np
from scipy.stats import invgamma
from typing import List, Dict, Tuple, Optional

from .data import TaskAAuctionData, TaskBAuctionData, MCMCConfig
from .analysis import compute_collinearity_diagnostics
from .utils import (
    sample_truncated_normal,
    gelman_rubin,
    selection_prob_at_least_one_exceeds_cutoff,
    selection_prob_reaches_formal_stage,
    informal_bid_multiplier_option_c,
    informal_bid_multiplier,
    misreporting_measures,
)

# Numba acceleration (optional)
try:
    from .numba_kernels import (
        HAS_NUMBA,
        seed_numba_rng,
        task_a_run_chain_intercept,
        task_b_update_b_star,
        task_b_logpost_gamma,
        task_b_logpost_kappa,
        task_b_log_selection_sum,
        task_b_sum_sq_v,
    )
except Exception:
    HAS_NUMBA = False
    task_a_run_chain_intercept = None
    seed_numba_rng = None
    task_b_update_b_star = None
    task_b_logpost_gamma = None
    task_b_logpost_kappa = None
    task_b_log_selection_sum = None
    task_b_sum_sq_v = None


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

        # Apply burn-in and thinning (pool chains for posterior summaries)
        burn = int(self.config.burn_in)
        thin = int(self.config.thinning)
        x_bar = self.X.mean(axis=0) if self.N > 0 else np.ones(self.k)

        beta_slices = [chain[burn::thin] for chain in all_chains_beta]
        sigma_slices = [chain[burn::thin] for chain in all_chains_sigma]

        beta_samples = np.concatenate(beta_slices, axis=0) if beta_slices else np.empty((0, self.k))
        sigma_samples = np.concatenate(sigma_slices, axis=0) if sigma_slices else np.empty((0,))
        mu_samples = beta_samples @ x_bar if beta_samples.size else np.empty((0,))
        # Compute Gelman-Rubin statistic
        rhat_beta = gelman_rubin([chain[self.config.burn_in:] for chain in all_chains_beta])
        rhat_sigma = gelman_rubin([chain[self.config.burn_in:] for chain in all_chains_sigma])
        rhat_mu = gelman_rubin([(chain[self.config.burn_in:] @ x_bar) for chain in all_chains_beta])

        return {
            'mu_samples': mu_samples,
            'beta_samples': beta_samples,
            'sigma_samples': sigma_samples,
            'all_chains_mu': [chain @ x_bar for chain in all_chains_beta],
            'all_chains_beta': all_chains_beta,
            'all_chains_sigma': all_chains_sigma,
            'rhat_mu': rhat_mu,
            'rhat_beta': rhat_beta,
            'rhat_sigma': rhat_sigma
        }


class TaskBMCMCSampler:
    """MCMC sampler for Task B (two-stage DGP; selection-aware likelihood).

    DGP (Jan 14, 2026 notes), intercept-only cutoff baseline:
        v_ij = gamma + nu_ij,  nu_ij ~ N(0, sigma_nu^2)
        b^I_ij = (1 - 1/n_i) * exp(kappa) * v_ij        (option C reparam)
        admit iff b^I_ij >= b^*_i
        u_ij = v_ij + eta_ij,  eta_ij ~ N(0, sigma_eta^2)
        b^F_ij = (1 - 1/n_i) * u_ij  (for admitted bidders only)

    Observed sample conditions on reaching formal stage (S_i=1). We correct via:
        p(data_i | theta, S_i=1) ∝ p(data_i | theta) / Pr(S_i=1 | theta).
    """

    def __init__(self, auctions: List[TaskBAuctionData], config: MCMCConfig):
        self.auctions = auctions
        self.config = config

        self.misreporting_mode = getattr(self.config, "task_b_misreporting_mode", "scale")
        if self.misreporting_mode not in ("scale", "shift"):
            raise ValueError("task_b_misreporting_mode must be 'scale' or 'shift'")

        self.working_auctions = [a for a in auctions if np.isfinite(a.U_i)]
        self.N = len(self.working_auctions)

        self.n_dropped_all_reject = len(auctions) - self.N
        self.n_two_sided = sum(1 for a in self.working_auctions if np.isfinite(a.L_i))
        self.n_one_sided_upper = self.N - self.n_two_sided

        if self.N > 0:
            self.k = int(self.working_auctions[0].X_i.shape[0])
            self.X = np.vstack([a.X_i for a in self.working_auctions])
            self.L = np.array([a.L_i for a in self.working_auctions], dtype=np.float64)
            self.U = np.array([a.U_i for a in self.working_auctions], dtype=np.float64)
            self.n_bidders = np.array([a.n_bidders for a in self.working_auctions], dtype=np.int64)
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

        # Cache bid arrays for likelihood evaluation
        self.informal_bids = [a.informal_bids.astype(float) for a in self.working_auctions]
        self.formal_bids = [a.formal_bids.astype(float) for a in self.working_auctions]
        self.admitted = [a.admitted.astype(bool) for a in self.working_auctions]

        self.lambda_f = np.array([1.0 - 1.0 / float(n) for n in self.n_bidders], dtype=float)

        self.collinearity = compute_collinearity_diagnostics(self.X)
        if (
            self.collinearity.get("flag_high_corr")
            or self.collinearity.get("flag_high_cond")
            or self.collinearity.get("flag_high_vif")
        ):
            print(
                "WARNING: Cutoff covariates exhibit severe collinearity "
                f"(max |corr|={self.collinearity.get('max_abs_corr'):.3f}, "
                f"cond={self.collinearity.get('condition_number'):.2e}, "
                f"max VIF={self.collinearity.get('max_vif')}). "
                "Identification may be weak or not achieved.",
                flush=True,
            )

        self._pack_task_b_arrays()

        print(
            "Task B MCMC (two-stage, selection-aware) using "
            f"{self.N} observed auctions (two-sided: {self.n_two_sided}, "
            f"one-sided upper: {self.n_one_sided_upper}); dropped {self.n_dropped_all_reject} all-reject",
            flush=True,
        )

    def _pack_task_b_arrays(self) -> None:
        """Pack bidder-level arrays into contiguous buffers for Numba kernels."""
        if self.N == 0:
            self._offsets = np.empty((0,), dtype=np.int64)
            self._bI = np.empty((0,), dtype=np.float64)
            self._bF = np.empty((0,), dtype=np.float64)
            self._adm = np.empty((0,), dtype=np.bool_)
            return

        total = int(np.sum(self.n_bidders))
        self._offsets = np.zeros(self.N, dtype=np.int64)
        self._bI = np.empty((total,), dtype=np.float64)
        self._bF = np.empty((total,), dtype=np.float64)
        self._adm = np.empty((total,), dtype=np.bool_)

        cursor = 0
        for i in range(self.N):
            self._offsets[i] = cursor
            n = int(self.n_bidders[i])
            bI = self.informal_bids[i].astype(np.float64, copy=False)
            bF = self.formal_bids[i].astype(np.float64, copy=False)
            adm = self.admitted[i].astype(np.bool_, copy=False)
            self._bI[cursor:cursor + n] = bI
            self._bF[cursor:cursor + n] = bF
            self._adm[cursor:cursor + n] = adm
            cursor += n

    def _get_cutoff_beta_prior(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = (
            self.config.beta_prior_mean
            if self.config.beta_prior_mean is not None
            else self.config.task_b_cutoff_prior_mean
        )
        std = (
            self.config.beta_prior_std
            if self.config.beta_prior_std is not None
            else self.config.task_b_cutoff_prior_std
        )
        mean = np.atleast_1d(mean).astype(float)
        std = np.atleast_1d(std).astype(float)
        if mean.size == 1:
            # For the moments cutoff (k>1), treat a scalar as an intercept prior and
            # default the remaining slopes to 0.0, rather than repeating the intercept.
            if self.k == 1:
                mean = np.array([mean.item()], dtype=float)
            else:
                mean_full = np.zeros(self.k, dtype=float)
                mean_full[0] = mean.item()
                mean = mean_full
        if std.size == 1:
            std = np.full(self.k, std.item())
        if mean.size != self.k or std.size != self.k:
            raise ValueError("cutoff beta prior has wrong length")
        return mean, std

    def _stage_sigmas(self) -> Tuple[float, float]:
        stage = int(self.config.task_b_stage)
        if stage not in (1, 2, 3):
            raise ValueError("task_b_stage must be 1, 2, or 3")

        if self.config.task_b_sigma_nu_fixed is None or self.config.task_b_sigma_eta_fixed is None:
            raise ValueError("task_b_sigma_nu_fixed and task_b_sigma_eta_fixed must be set for Task B")

        sigma_nu = float(self.config.task_b_sigma_nu_fixed)
        sigma_eta = float(self.config.task_b_sigma_eta_fixed)

        if sigma_nu <= 0 or sigma_eta <= 0:
            raise ValueError("fixed sigmas must be positive")

        return sigma_nu, sigma_eta

    def _log_posterior_kappa(
        self,
        kappa: float,
        *,
        gamma: float,
        sigma_nu: float,
        sigma_eta: float,
        b_star: np.ndarray,
    ) -> float:
        # Prior on kappa (unconstrained), centered at 0 => tilde_alpha=0.
        kappa0 = float(self.config.task_b_kappa_prior_mean)
        s_kappa = float(self.config.task_b_kappa_prior_std)
        if s_kappa <= 0:
            raise ValueError("task_b_kappa_prior_std must be positive")

        logp = -0.5 * ((kappa - kappa0) ** 2) / (s_kappa * s_kappa) - np.log(s_kappa)

        # Likelihood contributions
        log2pi = 1.8378770664093453  # log(2*pi)

        for i in range(self.N):
            n = int(self.n_bidders[i])
            lambda_i = informal_bid_multiplier(n, kappa, mode=self.misreporting_mode)

            # Informal bids density: b^I = lambda_i * v, v ~ N(gamma, sigma_nu^2)
            bI = self.informal_bids[i]
            v = bI / lambda_i
            z = (v - gamma) / sigma_nu
            logp += -0.5 * np.sum(z * z) - bI.size * (np.log(sigma_nu) + np.log(lambda_i) + 0.5 * log2pi)

            # Formal bids conditional on v for admitted bidders:
            # b^F / lambda_f = v + eta, eta ~ N(0, sigma_eta^2)
            adm = self.admitted[i]
            if np.any(adm):
                bF = self.formal_bids[i][adm]
                v_adm = v[adm]
                resid = bF / float(self.lambda_f[i]) - v_adm
                logp += (
                    -0.5 * np.sum((resid / sigma_eta) ** 2)
                    - resid.size * (np.log(sigma_eta) + np.log(float(self.lambda_f[i])) + 0.5 * log2pi)
                )

            # Selection penalty: -log Pr(S=1 | b_star_i)
            p_select = selection_prob_reaches_formal_stage(
                float(b_star[i]), float(gamma), float(sigma_nu), n, float(kappa), mode=self.misreporting_mode
            )
            logp += -np.log(p_select)

        return float(logp)

    def run_chain(self, chain_id: int = 0) -> Dict:
        print(f"\nRunning Task B chain {chain_id}...", flush=True)

        # Stage config
        stage = int(self.config.task_b_stage)
        sigma_nu, sigma_eta = self._stage_sigmas()

        # Priors
        beta_prior_mean, beta_prior_std = self._get_cutoff_beta_prior()
        V0_inv = np.diag(1.0 / (beta_prior_std ** 2))

        # Initialize parameters
        beta = beta_prior_mean.copy()
        sigma_omega = float(self.config.task_b_sigma_omega_init)

        gamma = float(self.config.task_b_gamma_prior_mean)
        kappa = float(self.config.task_b_kappa_init)

        # If later stages allow estimating sigmas, initialize from fixed values.
        sigma_nu_current = float(sigma_nu)
        sigma_eta_current = float(sigma_eta)

        # Storage
        beta_chain = np.zeros((self.config.n_iterations, self.k), dtype=float)
        gamma_chain = np.zeros(self.config.n_iterations, dtype=float)
        kappa_chain = np.zeros(self.config.n_iterations, dtype=float)
        sigma_omega_chain = np.zeros(self.config.n_iterations, dtype=float)
        sigma_nu_chain = np.zeros(self.config.n_iterations, dtype=float)
        sigma_eta_chain = np.zeros(self.config.n_iterations, dtype=float)

        a_omega = float(self.config.task_b_sigma_omega_prior_a)
        b_omega = float(self.config.task_b_sigma_omega_prior_b)

        a_nu = float(self.config.task_b_sigma_nu_prior_a)
        b_nu = float(self.config.task_b_sigma_nu_prior_b)

        a_eta = float(self.config.task_b_sigma_eta_prior_a)
        b_eta = float(self.config.task_b_sigma_eta_prior_b)

        gamma0 = float(self.config.task_b_gamma_prior_mean)
        s_gamma = float(self.config.task_b_gamma_prior_std)
        if s_gamma <= 0:
            raise ValueError("task_b_gamma_prior_std must be positive")

        gamma_prop_sd = float(self.config.task_b_gamma_proposal_sd)
        if gamma_prop_sd <= 0:
            raise ValueError("task_b_gamma_proposal_sd must be positive")

        use_numba = bool(
            HAS_NUMBA
            and getattr(self.config, "task_b_use_numba", True)
            and task_b_update_b_star is not None
            and task_b_logpost_gamma is not None
            and task_b_logpost_kappa is not None
            and task_b_log_selection_sum is not None
            and task_b_sum_sq_v is not None
        )
        mode_flag = 0 if self.misreporting_mode == "scale" else 1
        if use_numba and seed_numba_rng is not None:
            seed = int(np.random.randint(0, 2**31 - 1)) + 991 * int(chain_id)
            seed_numba_rng(seed)

        # Initialize latent cutoffs from naive truncated normal
        b_star = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            xb = float(self.X[i] @ beta)
            b_star[i] = sample_truncated_normal(xb, sigma_omega, self.L[i], self.U[i])

        # MH tuning for kappa
        kappa_prop_sd = float(self.config.task_b_kappa_proposal_sd)
        if kappa_prop_sd <= 0:
            raise ValueError("task_b_kappa_proposal_sd must be positive")

        n_kappa_accept = 0
        n_bstar_accept = 0
        n_bstar_total = 0

        for t in range(self.config.n_iterations):
            # Step 1: update latent cutoffs b*_i via independence MH
            if use_numba:
                acc, total = task_b_update_b_star(
                    b_star,
                    self.X,
                    self.L,
                    self.U,
                    self.n_bidders,
                    beta,
                    float(sigma_omega),
                    float(gamma),
                    float(sigma_nu_current),
                    float(kappa),
                    int(mode_flag),
                )
                n_bstar_accept += int(acc)
                n_bstar_total += int(total)
            else:
                for i in range(self.N):
                    xb = float(self.X[i] @ beta)
                    b_prop = sample_truncated_normal(xb, sigma_omega, self.L[i], self.U[i])

                    n = int(self.n_bidders[i])
                    p_old = selection_prob_reaches_formal_stage(
                        b_star[i], gamma, sigma_nu_current, n, kappa, mode=self.misreporting_mode
                    )
                    p_prop = selection_prob_reaches_formal_stage(
                        b_prop, gamma, sigma_nu_current, n, kappa, mode=self.misreporting_mode
                    )
                    alpha = p_old / p_prop

                    n_bstar_total += 1
                    if alpha >= 1.0 or np.random.rand() < alpha:
                        b_star[i] = b_prop
                        n_bstar_accept += 1

            # Step 2: update cutoff beta via conjugate regression
            if self.N > 0:
                XtX = self.X.T @ self.X
                V_post = np.linalg.inv(V0_inv + XtX / (sigma_omega ** 2))
                beta_post = V_post @ (
                    V0_inv @ beta_prior_mean + (self.X.T @ b_star) / (sigma_omega ** 2)
                )
                beta = np.random.multivariate_normal(beta_post, V_post)

            # Step 3: update sigma_omega^2
            a_post = a_omega + 0.5 * self.N
            resid = b_star - self.X @ beta
            b_post = b_omega + 0.5 * float(np.sum(resid * resid))
            sigma_omega_sq = invgamma.rvs(a_post, scale=b_post)
            sigma_omega = float(np.sqrt(sigma_omega_sq))

            # Step 4: update gamma (given kappa => valuations are scaled informal bids)
            if use_numba:
                n_v = int(self._bI.size)
                if n_v > 0:
                    gamma_prop = float(gamma + np.random.normal(0.0, gamma_prop_sd))
                    log_alpha = task_b_logpost_gamma(
                        float(gamma_prop),
                        float(gamma0),
                        float(s_gamma),
                        b_star,
                        self._bI,
                        self._offsets,
                        self.n_bidders,
                        float(sigma_nu_current),
                        float(kappa),
                        int(mode_flag),
                    ) - task_b_logpost_gamma(
                        float(gamma),
                        float(gamma0),
                        float(s_gamma),
                        b_star,
                        self._bI,
                        self._offsets,
                        self.n_bidders,
                        float(sigma_nu_current),
                        float(kappa),
                        int(mode_flag),
                    )
                    if log_alpha >= 0.0 or np.log(np.random.rand()) < log_alpha:
                        gamma = gamma_prop
            else:
                v_all = []
                for i in range(self.N):
                    n = int(self.n_bidders[i])
                    lambda_i = informal_bid_multiplier(n, kappa, mode=self.misreporting_mode)
                    v_all.append(self.informal_bids[i] / lambda_i)
                v_all = np.concatenate(v_all) if v_all else np.empty((0,), dtype=float)

                n_v = int(v_all.size)
                if n_v > 0:
                    gamma_prop = float(gamma + np.random.normal(0.0, gamma_prop_sd))

                    # Random-walk MH for gamma with the full selection-aware posterior:
                    #   p(gamma | ...) ∝ prior(gamma) * p(v_all | gamma, sigma_nu) * ∏ 1/Pr(S=1|b*_i, gamma, ...)
                    def logpost_gamma(g: float) -> float:
                        lp = -0.5 * ((g - gamma0) ** 2) / (s_gamma * s_gamma)
                        lp += -0.5 * float(np.sum(((v_all - g) / sigma_nu_current) ** 2))
                        lp += -float(n_v) * float(np.log(sigma_nu_current))
                        for ii in range(self.N):
                            n = int(self.n_bidders[ii])
                            p = selection_prob_reaches_formal_stage(
                                float(b_star[ii]), float(g), float(sigma_nu_current), n, float(kappa), mode=self.misreporting_mode
                            )
                            lp += -float(np.log(p))
                        return float(lp)

                    log_alpha = logpost_gamma(gamma_prop) - logpost_gamma(gamma)
                    if log_alpha >= 0.0 or np.log(np.random.rand()) < log_alpha:
                        gamma = gamma_prop

            # Step 5: optionally update sigma_nu^2 (stage 3)
            if stage == 3 and n_v > 0:
                a_post = a_nu + 0.5 * n_v
                if use_numba:
                    ss, _ = task_b_sum_sq_v(
                        self._bI,
                        self._offsets,
                        self.n_bidders,
                        float(gamma),
                        float(kappa),
                        int(mode_flag),
                    )
                else:
                    ss = float(np.sum((v_all - gamma) ** 2))
                b_post = b_nu + 0.5 * ss
                # Selection-aware correction: propose from the conjugate IG (ignoring selection),
                # then accept/reject using the selection penalty ratio.
                sigma_nu_sq_prop = float(invgamma.rvs(a_post, scale=b_post))
                sigma_nu_prop = float(np.sqrt(sigma_nu_sq_prop))

                if use_numba:
                    logp_select_old = task_b_log_selection_sum(
                        b_star,
                        self.n_bidders,
                        float(gamma),
                        float(sigma_nu_current),
                        float(kappa),
                        int(mode_flag),
                    )
                    logp_select_prop = task_b_log_selection_sum(
                        b_star,
                        self.n_bidders,
                        float(gamma),
                        float(sigma_nu_prop),
                        float(kappa),
                        int(mode_flag),
                    )
                else:
                    logp_select_old = 0.0
                    logp_select_prop = 0.0
                    for i in range(self.N):
                        n = int(self.n_bidders[i])
                        p_old = selection_prob_reaches_formal_stage(
                            float(b_star[i]), float(gamma), float(sigma_nu_current), n, float(kappa), mode=self.misreporting_mode
                        )
                        p_prop = selection_prob_reaches_formal_stage(
                            float(b_star[i]), float(gamma), float(sigma_nu_prop), n, float(kappa), mode=self.misreporting_mode
                        )
                        logp_select_old += float(np.log(p_old))
                        logp_select_prop += float(np.log(p_prop))

                log_alpha = logp_select_old - logp_select_prop
                if log_alpha >= 0.0 or np.log(np.random.rand()) < log_alpha:
                    sigma_nu_current = sigma_nu_prop

            # Step 6: optionally update sigma_eta^2 (stage 2/3)
            if stage in (2, 3):
                eta_all = []
                for i in range(self.N):
                    adm = self.admitted[i]
                    if not np.any(adm):
                        continue
                    n = int(self.n_bidders[i])
                    lambda_i = informal_bid_multiplier(n, kappa, mode=self.misreporting_mode)
                    v = self.informal_bids[i] / lambda_i
                    resid = self.formal_bids[i][adm] / float(self.lambda_f[i]) - v[adm]
                    eta_all.append(resid)
                eta_all = np.concatenate(eta_all) if eta_all else np.empty((0,), dtype=float)
                n_eta = int(eta_all.size)
                if n_eta > 0:
                    a_post = a_eta + 0.5 * n_eta
                    b_post = b_eta + 0.5 * float(np.sum(eta_all * eta_all))
                    sigma_eta_sq = invgamma.rvs(a_post, scale=b_post)
                    sigma_eta_current = float(np.sqrt(sigma_eta_sq))

            # Step 7: update kappa via random-walk MH
            kappa_prop = float(kappa + np.random.normal(0.0, kappa_prop_sd))
            if use_numba:
                logp_old = task_b_logpost_kappa(
                    float(kappa),
                    float(self.config.task_b_kappa_prior_mean),
                    float(self.config.task_b_kappa_prior_std),
                    float(gamma),
                    float(sigma_nu_current),
                    float(sigma_eta_current),
                    b_star,
                    self._bI,
                    self._bF,
                    self._adm,
                    self._offsets,
                    self.n_bidders,
                    self.lambda_f,
                    int(mode_flag),
                )
                logp_prop = task_b_logpost_kappa(
                    float(kappa_prop),
                    float(self.config.task_b_kappa_prior_mean),
                    float(self.config.task_b_kappa_prior_std),
                    float(gamma),
                    float(sigma_nu_current),
                    float(sigma_eta_current),
                    b_star,
                    self._bI,
                    self._bF,
                    self._adm,
                    self._offsets,
                    self.n_bidders,
                    self.lambda_f,
                    int(mode_flag),
                )
            else:
                logp_old = self._log_posterior_kappa(
                    kappa,
                    gamma=gamma,
                    sigma_nu=sigma_nu_current,
                    sigma_eta=sigma_eta_current,
                    b_star=b_star,
                )
                logp_prop = self._log_posterior_kappa(
                    kappa_prop,
                    gamma=gamma,
                    sigma_nu=sigma_nu_current,
                    sigma_eta=sigma_eta_current,
                    b_star=b_star,
                )
            log_alpha = logp_prop - logp_old
            if log_alpha >= 0.0 or np.log(np.random.rand()) < log_alpha:
                kappa = kappa_prop
                n_kappa_accept += 1

            beta_chain[t] = beta
            gamma_chain[t] = gamma
            kappa_chain[t] = kappa
            sigma_omega_chain[t] = sigma_omega
            sigma_nu_chain[t] = sigma_nu_current
            sigma_eta_chain[t] = sigma_eta_current

            if (t + 1) % 2000 == 0:
                kappa_acc = n_kappa_accept / float(t + 1)
                bstar_acc = n_bstar_accept / float(max(1, n_bstar_total))
                print(
                    f"  Iter {t+1}: gamma={gamma:.3f}, kappa={kappa:.3f}, "
                    f"tilde_alpha={misreporting_measures(int(self.n_bidders[0]), float(kappa), mode=self.misreporting_mode)[2]:.3f}, c0={beta[0]:.3f}, "
                    f"sigma_omega={sigma_omega:.3f}, acc(kappa)={kappa_acc:.2f}, acc(b*)={bstar_acc:.2f}",
                    flush=True,
                )

        print(f"Chain {chain_id} complete!", flush=True)
        return {
            'beta': beta_chain,
            'gamma': gamma_chain,
            'kappa': kappa_chain,
            'sigma_omega': sigma_omega_chain,
            'sigma_nu': sigma_nu_chain,
            'sigma_eta': sigma_eta_chain,
        }

    def run(self) -> Dict:
        """Run multiple chains and pool posterior samples."""
        all_chains: List[Dict] = []
        for chain_id in range(self.config.n_chains):
            all_chains.append(self.run_chain(chain_id))

        burn = int(self.config.burn_in)
        thin = int(self.config.thinning)

        beta_samples = np.concatenate([c['beta'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0, self.k))
        gamma_samples = np.concatenate([c['gamma'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0,))
        kappa_samples = np.concatenate([c['kappa'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0,))
        sigma_omega_samples = np.concatenate([c['sigma_omega'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0,))
        sigma_nu_samples = np.concatenate([c['sigma_nu'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0,))
        sigma_eta_samples = np.concatenate([c['sigma_eta'][burn::thin] for c in all_chains], axis=0) if all_chains else np.empty((0,))

        n_ref = int(self.n_bidders[0]) if getattr(self, 'n_bidders', np.array([])).size else 3
        tilde_alpha_samples = np.array([misreporting_measures(n_ref, float(k), mode=self.misreporting_mode)[2] for k in kappa_samples], dtype=float)
        alpha_additive_samples = np.array([misreporting_measures(n_ref, float(k), mode=self.misreporting_mode)[3] for k in kappa_samples], dtype=float)

        # Gelman-Rubin diagnostics (use unthinned, post burn-in draws)
        rhat_beta = gelman_rubin([c['beta'][burn:] for c in all_chains])
        rhat_gamma = gelman_rubin([c['gamma'][burn:] for c in all_chains])
        rhat_kappa = gelman_rubin([c['kappa'][burn:] for c in all_chains])

        return {
            'beta_samples': beta_samples,
            'cutoff_c_samples': beta_samples[:, 0] if beta_samples.size else np.empty((0,)),
            'gamma_samples': gamma_samples,
            'kappa_samples': kappa_samples,
            'tilde_alpha_samples': tilde_alpha_samples,
            'alpha_additive_samples': alpha_additive_samples,
            'sigma_omega_samples': sigma_omega_samples,
            'sigma_nu_samples': sigma_nu_samples,
            'sigma_eta_samples': sigma_eta_samples,
            'misreporting_mode': self.misreporting_mode,
            'all_chains': all_chains,
            'rhat_beta': rhat_beta,
            'rhat_gamma': rhat_gamma,
            'rhat_kappa': rhat_kappa,
            'collinearity_diagnostics': self.collinearity,
        }
