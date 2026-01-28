"""
Publication-quality plotting for MCMC results.

This module contains visualization classes for:
- Task A: Single cutoff diagnostics and interval plots
- Task B: Two-stage diagnostics (gamma, tilde_alpha, cutoff)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .data import TaskAAuctionData, TaskBAuctionData


class TaskAVisualizer:
    """Create publication-quality plots for Task A (single cutoff)."""

    @staticmethod
    def plot_diagnostics(results: Dict, true_b_star: float, save_path: str):
        """Create comprehensive diagnostic plots.

        Args:
            results: Output from TaskAMCMCSampler.run()
            true_b_star: True cutoff value
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

        mu_samples = results['mu_samples']
        sigma_samples = results['sigma_samples']
        all_chains_mu = results['all_chains_mu']

        # 1. Trace plot for mu (all chains)
        ax = fig.add_subplot(gs[0, :2])
        colors = sns.color_palette("husl", len(all_chains_mu))
        for i, chain in enumerate(all_chains_mu):
            ax.plot(chain, alpha=0.6, linewidth=0.8, color=colors[i],
                   label=f'Chain {i+1}')
        ax.axhline(true_b_star, color='red', linestyle='--', linewidth=1.5,
                  label='True b*', zorder=10)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('mu')
        ax.set_title('(A) Trace Plot: mu (All Chains)')
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # 2. Posterior distribution of mu
        ax = fig.add_subplot(gs[0, 2])
        ax.hist(mu_samples, bins=40, density=True, alpha=0.7,
               color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(true_b_star, color='red', linestyle='--', linewidth=1.5,
                  label='True b*')
        ax.axvline(np.mean(mu_samples), color='darkblue', linestyle='-',
                  linewidth=1.5, label='Posterior mean')
        ax.set_xlabel('mu')
        ax.set_ylabel('Density')
        ax.set_title('(B) Posterior: mu')
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Posterior distribution of sigma
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(sigma_samples, bins=40, density=True, alpha=0.7,
               color='seagreen', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('sigma')
        ax.set_ylabel('Density')
        ax.set_title('(C) Posterior: sigma')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Autocorrelation for mu
        ax = fig.add_subplot(gs[1, 1])
        max_lag = min(100, len(mu_samples) // 2)
        acf = [np.corrcoef(mu_samples[:-lag], mu_samples[lag:])[0, 1]
               if lag > 0 else 1.0 for lag in range(max_lag)]
        ax.plot(range(max_lag), acf, linewidth=1.5, color='steelblue')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('(D) Autocorrelation: mu')
        ax.grid(True, alpha=0.3)

        # 5. Running mean
        ax = fig.add_subplot(gs[1, 2])
        running_mean = np.cumsum(mu_samples) / np.arange(1, len(mu_samples) + 1)
        ax.plot(running_mean, linewidth=1.5, color='steelblue')
        ax.axhline(true_b_star, color='red', linestyle='--', linewidth=1.5,
                  label='True b*')
        ax.set_xlabel('Iteration (post burn-in)')
        ax.set_ylabel('Running Mean')
        ax.set_title('(E) Convergence: Running Mean')
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_intervals(auctions: List[TaskAAuctionData], true_b_star: float,
                      estimated_b_star: float, save_path: str):
        """Plot interval bounds for complete auctions only.

        Args:
            auctions: List of auction data
            true_b_star: True cutoff value
            estimated_b_star: Estimated cutoff
            save_path: Path to save the figure
        """
        complete_auctions = [a for a in auctions if a.is_complete]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by lower bound
        sorted_auctions = sorted(complete_auctions, key=lambda a: a.L_i)

        for i, auction in enumerate(sorted_auctions):
            ax.plot([auction.L_i, auction.U_i], [i, i], 'o-',
                   color='steelblue', linewidth=2, markersize=5, alpha=0.7)

        # Add cutoffs
        ax.axvline(true_b_star, color='red', linestyle='--', linewidth=2,
                  label=f'True b* = {true_b_star:.3f}', zorder=10)
        ax.axvline(estimated_b_star, color='darkgreen', linestyle='-', linewidth=2,
                  label=f'Estimated mu = {estimated_b_star:.3f}', zorder=10)

        ax.set_xlabel('Bid Value', fontsize=12)
        ax.set_ylabel('Auction Index (sorted by lower bound)', fontsize=12)
        ax.set_title(f'Interval Bounds [L_i, U_i] for Complete Auctions (N={len(sorted_auctions)})',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Interval plot saved to {save_path}")
        plt.close()


class TaskBVisualizer:
    """Create plots for Task B (two-stage DGP)."""

    @staticmethod
    def plot_diagnostics(
        results: Dict,
        *,
        true_gamma: float,
        true_tilde_alpha: float,
        true_beta_cutoff: Optional[np.ndarray] = None,
        true_cutoff_c: Optional[float] = None,
        true_sigma_omega: Optional[float] = None,
        true_sigma_nu: Optional[float] = None,
        true_sigma_eta: Optional[float] = None,
        save_path: str,
    ):
        def beta_names(k: int) -> List[str]:
            if k == 1:
                return ["c"]
            if k == 3:
                return ["c", "depth_mean_23", "depth_gap_23"]
            if k == 4:
                return ["c", "theta1", "theta2", "theta3"]
            return [f"beta_{j}" for j in range(k)]

        gamma = results["gamma_samples"]
        tilde_alpha = results["tilde_alpha_samples"]

        beta_samples = results.get("beta_samples")
        if beta_samples is None or beta_samples.size == 0:
            raise ValueError("Task B diagnostics require beta_samples")
        k = int(beta_samples.shape[1]) if beta_samples.ndim == 2 else 1

        if true_beta_cutoff is None:
            if true_cutoff_c is None:
                true_beta = None
            else:
                true_beta = np.full(k, np.nan, dtype=float)
                true_beta[0] = float(true_cutoff_c)
        else:
            true_beta = np.atleast_1d(true_beta_cutoff).astype(float)
            if true_beta.size != k:
                raise ValueError(f"true_beta_cutoff has length {true_beta.size}, expected k={k}")

        params: List[Tuple[str, np.ndarray, Optional[float]]] = [
            ("gamma", gamma, float(true_gamma)),
            ("tilde_alpha", tilde_alpha, float(true_tilde_alpha)),
        ]

        for j, name in enumerate(beta_names(k)):
            tv = None if true_beta is None else float(true_beta[j])
            params.append((f"beta_cutoff:{name}", beta_samples[:, j], tv))

        sigma_omega = results.get("sigma_omega_samples")
        if sigma_omega is not None and sigma_omega.size:
            params.append(("sigma_omega", sigma_omega, None if true_sigma_omega is None else float(true_sigma_omega)))

        sigma_eta = results.get("sigma_eta_samples")
        if sigma_eta is not None and sigma_eta.size and np.std(sigma_eta) > 1e-12:
            params.append(("sigma_eta", sigma_eta, None if true_sigma_eta is None else float(true_sigma_eta)))

        sigma_nu = results.get("sigma_nu_samples")
        if sigma_nu is not None and sigma_nu.size and np.std(sigma_nu) > 1e-12:
            params.append(("sigma_nu", sigma_nu, None if true_sigma_nu is None else float(true_sigma_nu)))

        n_params = len(params)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / float(n_cols)))

        fig, axes = plt.subplots(
            nrows=2 * n_rows,
            ncols=n_cols,
            figsize=(5.0 * n_cols, 2.8 * 2 * n_rows),
        )
        axes = np.atleast_2d(axes)

        # Hide all axes first; enable as we fill.
        for ax in axes.flat:
            ax.set_visible(False)

        for idx, (label, samples, true_value) in enumerate(params):
            r = idx // n_cols
            c = idx % n_cols
            ax_trace = axes[2 * r, c]
            ax_post = axes[2 * r + 1, c]
            ax_trace.set_visible(True)
            ax_post.set_visible(True)

            letter = chr(ord("A") + idx)

            # Trace
            ax_trace.plot(samples, linewidth=0.8, alpha=0.8, color="steelblue")
            if true_value is not None and np.isfinite(true_value):
                ax_trace.axhline(true_value, color="red", linestyle="--", linewidth=1.3, label="True")
                ax_trace.legend(frameon=True, framealpha=0.9)
            ax_trace.set_title(f"({letter}) Trace: {label}")
            ax_trace.set_xlabel("Iteration (post burn-in)")
            ax_trace.grid(True, alpha=0.3)

            # Posterior
            ax_post.hist(samples, bins=40, density=True, alpha=0.7, color="slategray", edgecolor="black", linewidth=0.5)
            if true_value is not None and np.isfinite(true_value):
                ax_post.axvline(true_value, color="red", linestyle="--", linewidth=1.3, label="True")
            ax_post.axvline(float(np.mean(samples)), color="navy", linestyle="-", linewidth=1.3, label="Mean")
            ax_post.set_title(f"Posterior: {label}")
            ax_post.grid(True, alpha=0.3, axis="y")
            ax_post.legend(frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Diagnostics saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_informal_vs_formal(
        auctions: List[TaskBAuctionData],
        *,
        kappa_est: float,
        misreporting_mode: str = "scale",
        save_path: str,
    ):
        xs = []
        ys = []
        for a in auctions:
            adm = a.admitted
            if not np.any(adm):
                continue
            xs.append(a.informal_bids[adm])
            ys.append(a.formal_bids[adm])
        if not xs:
            return

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x, y, s=15, alpha=0.5, color='steelblue', edgecolors='none')
        ax.set_xlabel('Informal bid (admitted)')
        ax.set_ylabel('Formal bid')
        ax.set_title('Admitted bidders: informal vs formal')
        ax.grid(True, alpha=0.3)

        n_ref = int(auctions[0].n_bidders) if auctions else 3
        lambda_f = 1.0 - 1.0 / float(n_ref)
        if misreporting_mode == 'scale':
            slope = 1.0 / float(np.exp(kappa_est))
            guide_label = f'y = x / exp(kappa) (kappa={kappa_est:.2f})'
        else:
            slope = float(lambda_f) / float(np.exp(kappa_est))
            guide_label = f'y = lambda_f * x / exp(kappa) (kappa={kappa_est:.2f})'
        xline = np.linspace(float(np.min(x)), float(np.max(x)), 50)
        ax.plot(
            xline,
            slope * xline,
            color='darkred',
            linewidth=2,
            label=guide_label,
        )
        ax.legend(frameon=True, framealpha=0.9)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
        plt.close()
