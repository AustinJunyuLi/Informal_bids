"""
Publication-quality plotting for MCMC results.

This module contains visualization classes for:
- Task A: Single cutoff diagnostics and interval plots
- Task B: Type-specific cutoff diagnostics and interval plots
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
    """Create publication-quality plots for Task B (type-specific cutoffs)."""

    @staticmethod
    def plot_diagnostics(results: Dict, true_b_star_S: float, true_b_star_F: float,
                        save_path: str):
        """Comprehensive diagnostic plots for type-specific cutoffs.

        Args:
            results: Output from TaskBMCMCSampler.run()
            true_b_star_S: True cutoff for type S
            true_b_star_F: True cutoff for type F
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

        mu_S = results['mu_S_samples']
        mu_F = results['mu_F_samples']
        gap = results['gap_samples']
        all_chains = results['all_chains']

        # Color scheme
        color_S = 'steelblue'
        color_F = 'coral'
        color_gap = 'purple'

        # Row 1: Trace plots
        ax = fig.add_subplot(gs[0, :2])
        for i, chain in enumerate(all_chains):
            alpha = 0.7 if i == 0 else 0.3
            ax.plot(chain['mu_S'], alpha=alpha, linewidth=0.8, color=color_S,
                   label=f'mu_S Chain {i+1}' if i == 0 else '')
            ax.plot(chain['mu_F'], alpha=alpha, linewidth=0.8, color=color_F,
                   label=f'mu_F Chain {i+1}' if i == 0 else '')
        ax.axhline(true_b_star_S, color='darkblue', linestyle='--', linewidth=1.5,
                  label=f'True b*_S = {true_b_star_S}')
        ax.axhline(true_b_star_F, color='darkred', linestyle='--', linewidth=1.5,
                  label=f'True b*_F = {true_b_star_F}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cutoff Value')
        ax.set_title('(A) Trace Plots: mu_S and mu_F')
        ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Row 1: Gap trace plot
        ax = fig.add_subplot(gs[0, 2:])
        for i, chain in enumerate(all_chains):
            gap_chain = chain['mu_S'] - chain['mu_F']
            alpha = 0.7 if i == 0 else 0.3
            ax.plot(gap_chain, alpha=alpha, linewidth=0.8, color=color_gap,
                   label=f'Delta Chain {i+1}' if i == 0 else '')
        ax.axhline(true_b_star_S - true_b_star_F, color='purple', linestyle='--',
                  linewidth=1.5, label=f'True Delta = {true_b_star_S - true_b_star_F:.3f}')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gap (mu_S - mu_F)')
        ax.set_title('(B) Trace Plot: Gap Delta')
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Row 2: Posterior distributions
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(mu_S, bins=40, density=True, alpha=0.7, color=color_S,
               edgecolor='black', linewidth=0.5)
        ax.axvline(true_b_star_S, color='darkblue', linestyle='--', linewidth=1.5,
                  label='True b*_S')
        ax.axvline(np.mean(mu_S), color='navy', linestyle='-', linewidth=1.5,
                  label='Posterior mean')
        ax.set_xlabel('mu_S')
        ax.set_ylabel('Density')
        ax.set_title('(C) Posterior: mu_S')
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[1, 1])
        ax.hist(mu_F, bins=40, density=True, alpha=0.7, color=color_F,
               edgecolor='black', linewidth=0.5)
        ax.axvline(true_b_star_F, color='darkred', linestyle='--', linewidth=1.5,
                  label='True b*_F')
        ax.axvline(np.mean(mu_F), color='firebrick', linestyle='-', linewidth=1.5,
                  label='Posterior mean')
        ax.set_xlabel('mu_F')
        ax.set_ylabel('Density')
        ax.set_title('(D) Posterior: mu_F')
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[1, 2])
        ax.hist(gap, bins=40, density=True, alpha=0.7, color=color_gap,
               edgecolor='black', linewidth=0.5)
        ax.axvline(true_b_star_S - true_b_star_F, color='purple', linestyle='--',
                  linewidth=1.5, label='True Delta')
        ax.axvline(np.mean(gap), color='indigo', linestyle='-', linewidth=1.5,
                  label='Posterior mean')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Gap Delta = mu_S - mu_F')
        ax.set_ylabel('Density')
        ax.set_title('(E) Posterior: Gap Delta')
        prob_positive = np.mean(gap > 0)
        ax.text(0.05, 0.95, f'Pr(Delta > 0) = {prob_positive:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        # Row 2: Credible intervals comparison
        ax = fig.add_subplot(gs[1, 3])
        ci_S = np.percentile(mu_S, [2.5, 97.5])
        ci_F = np.percentile(mu_F, [2.5, 97.5])

        # Plot CIs
        ax.barh([1], [ci_S[1] - ci_S[0]], left=[ci_S[0]], height=0.4,
               color=color_S, alpha=0.6, label='Type S: 95% CI')
        ax.barh([0], [ci_F[1] - ci_F[0]], left=[ci_F[0]], height=0.4,
               color=color_F, alpha=0.6, label='Type F: 95% CI')

        # Plot point estimates
        ax.scatter([np.mean(mu_S)], [1], color='navy', s=100, marker='o',
                  zorder=10, label='Posterior mean')
        ax.scatter([np.mean(mu_F)], [0], color='firebrick', s=100, marker='o',
                  zorder=10)

        # Plot true values
        ax.scatter([true_b_star_S], [1], color='darkblue', s=100, marker='x',
                  linewidths=3, zorder=10, label='True value')
        ax.scatter([true_b_star_F], [0], color='darkred', s=100, marker='x',
                  linewidths=3, zorder=10)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Type F', 'Type S'])
        ax.set_xlabel('Cutoff Value')
        ax.set_title('(F) Estimation Summary: 95% CIs')
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_type_intervals(auctions: List[TaskBAuctionData], true_b_star_S: float,
                           true_b_star_F: float, estimated_S: float,
                           estimated_F: float, save_path: str):
        """Plot type-specific interval bounds.

        Args:
            auctions: List of auction data
            true_b_star_S: True cutoff for type S
            true_b_star_F: True cutoff for type F
            estimated_S: Estimated cutoff for type S
            estimated_F: Estimated cutoff for type F
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Type S intervals
        S_auctions = [a for a in auctions if a.has_S_bounds]
        S_sorted = sorted(S_auctions, key=lambda a: a.L_S)

        for i, auction in enumerate(S_sorted):
            ax1.plot([auction.L_S, auction.U_S], [i, i], 'o-',
                    color='steelblue', linewidth=2, markersize=5, alpha=0.7)

        ax1.axvline(true_b_star_S, color='darkblue', linestyle='--', linewidth=2,
                   label=f'True b*_S = {true_b_star_S:.3f}', zorder=10)
        ax1.axvline(estimated_S, color='navy', linestyle='-', linewidth=2,
                   label=f'Estimated mu_S = {estimated_S:.3f}', zorder=10)
        ax1.set_xlabel('Bid Value', fontsize=12)
        ax1.set_ylabel('Auction Index (sorted)', fontsize=12)
        ax1.set_title(f'Type S: Interval Bounds [L_S, U_S] (N={len(S_sorted)})',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best', frameon=True, framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='x')

        # Type F intervals
        F_auctions = [a for a in auctions if a.has_F_bounds]
        F_sorted = sorted(F_auctions, key=lambda a: a.L_F)

        for i, auction in enumerate(F_sorted):
            ax2.plot([auction.L_F, auction.U_F], [i, i], 'o-',
                    color='coral', linewidth=2, markersize=5, alpha=0.7)

        ax2.axvline(true_b_star_F, color='darkred', linestyle='--', linewidth=2,
                   label=f'True b*_F = {true_b_star_F:.3f}', zorder=10)
        ax2.axvline(estimated_F, color='firebrick', linestyle='-', linewidth=2,
                   label=f'Estimated mu_F = {estimated_F:.3f}', zorder=10)
        ax2.set_xlabel('Bid Value', fontsize=12)
        ax2.set_ylabel('Auction Index (sorted)', fontsize=12)
        ax2.set_title(f'Type F: Interval Bounds [L_F, U_F] (N={len(F_sorted)})',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', frameon=True, framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Interval plots saved to {save_path}")
        plt.close()
