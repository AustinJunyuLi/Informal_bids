"""
Unified sensitivity analysis framework.

This module provides classes for running sensitivity analyses that
vary sample size N and assess estimation performance for both
Task A (single cutoff) and Task B (type-specific cutoffs).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from .data import (
    TaskADGPParameters, TaskBDGPParameters, MCMCConfig,
    TaskADataGenerator, TaskBDataGenerator
)
from .samplers import TaskAMCMCSampler, TaskBMCMCSampler


class TaskASensitivityAnalysis:
    """Sensitivity analysis for Task A (single cutoff).

    Varies sample size N and assesses estimation performance.
    """

    def __init__(self, n_replications: int = 10):
        self.n_replications = n_replications

    def run_single_replication(self, dgp_params: TaskADGPParameters,
                               mcmc_config: MCMCConfig, rep_id: int) -> Dict:
        """Run one replication and return metrics."""
        generator = TaskADataGenerator(dgp_params)
        auctions, stats = generator.generate_auction_data()

        n_complete = sum(1 for a in auctions if a.is_complete)
        pct_incomplete = 100 * (1 - n_complete / len(auctions)) if auctions else 0

        if len(auctions) < 2:
            # Not enough observed auctions
            return {
                'N': dgp_params.N,
                'rep_id': rep_id,
                'n_complete': n_complete,
                'pct_incomplete': pct_incomplete,
                'converged': False,
                'bias': np.nan,
                'rmse': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'ci_width': np.nan,
                'coverage': False,
                'rhat': np.nan
            }

        # Run MCMC (sampler will include one-sided upper bounds and drop all-reject)
        sampler = TaskAMCMCSampler(auctions, mcmc_config)
        results = sampler.run()

        mu_samples = results['mu_samples']
        mu_hat = np.mean(mu_samples)
        ci_lower, ci_upper = np.percentile(mu_samples, [2.5, 97.5])

        true_b_star = dgp_params.cutoff_at_mean_x()
        bias = mu_hat - true_b_star
        rmse = np.sqrt(bias**2 + np.var(mu_samples))
        ci_width = ci_upper - ci_lower
        coverage = (ci_lower <= true_b_star <= ci_upper)
        rhat = results['rhat_mu']

        return {
            'N': dgp_params.N,
            'rep_id': rep_id,
            'n_complete': n_complete,
            'pct_incomplete': pct_incomplete,
            'converged': rhat < 1.1,
            'mu_hat': mu_hat,
            'bias': bias,
            'rmse': rmse,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'coverage': coverage,
            'rhat': rhat
        }

    def sensitivity_sample_size(self, N_values: List[int] = None,
                                J: int = 3, mu_v: float = 1.3,
                                sigma_v: float = 0.2, b_star: float = 1.4) -> pd.DataFrame:
        """Run sample size sensitivity analysis.

        Args:
            N_values: List of sample sizes to test
            J: Number of bidders per auction
            mu_v: Mean valuation
            sigma_v: Std dev of valuation
            b_star: True cutoff

        Returns:
            DataFrame with results from all replications
        """
        if N_values is None:
            N_values = [20, 50, 100, 200, 500]

        print("\n" + "="*70)
        print("TASK A: SAMPLE SIZE SENSITIVITY ANALYSIS")
        print("="*70)
        print(f"Replications per N: {self.n_replications}")
        print(f"Sample sizes: {N_values}")
        print()

        mcmc_config = MCMCConfig(
            n_iterations=10000,
            burn_in=5000,
            thinning=10,
            n_chains=2
        )

        results_list = []

        for N in N_values:
            print(f"\nRunning N={N} ({self.n_replications} replications)...")

            dgp_params = TaskADGPParameters(
                N=N, J=J, mu_v=mu_v, sigma_v=sigma_v, b_star=b_star
            )

            for rep in range(self.n_replications):
                print(f"  Rep {rep+1}/{self.n_replications}...", end='', flush=True)
                result = self.run_single_replication(dgp_params, mcmc_config, rep)
                results_list.append(result)

                if result['converged']:
                    print(f" bias={result['bias']:.4f}, coverage={result['coverage']}")
                else:
                    print(" NOT CONVERGED or insufficient data")

        df = pd.DataFrame(results_list)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS BY SAMPLE SIZE")
        print("="*70)

        summary = df.groupby('N').agg({
            'n_complete': 'mean',
            'pct_incomplete': 'mean',
            'bias': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'ci_width': ['mean', 'std'],
            'coverage': 'mean',
            'converged': 'mean',
            'rhat': 'mean'
        }).round(4)

        print(summary)

    def plot_results(self, df: pd.DataFrame, save_path: str):
        """Create comprehensive sensitivity plots."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Task A: Sample Size Sensitivity Analysis',
                    fontsize=14, fontweight='bold', y=0.98)

        summary = df.groupby('N').agg({
            'bias': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'ci_width': ['mean', 'std'],
            'coverage': 'mean',
            'pct_incomplete': 'mean',
            'n_complete': 'mean'
        }).reset_index()

        N_values = summary['N'].values

        # Panel 1: Bias vs N
        ax = axes[0, 0]
        bias_mean = summary['bias']['mean'].values
        bias_std = summary['bias']['std'].values
        ax.errorbar(N_values, bias_mean, yerr=bias_std, marker='o',
                   markersize=8, linewidth=2, capsize=5, label='Mean bias +/- SD')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Unbiased')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Bias')
        ax.set_title('(A) Bias vs Sample Size')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 2: RMSE vs N
        ax = axes[0, 1]
        rmse_mean = summary['rmse']['mean'].values
        rmse_std = summary['rmse']['std'].values
        ax.errorbar(N_values, rmse_mean, yerr=rmse_std, marker='s',
                   markersize=8, linewidth=2, capsize=5, color='darkorange',
                   label='Mean RMSE +/- SD')
        ref_rmse = rmse_mean[0] * np.sqrt(N_values[0] / N_values)
        ax.plot(N_values, ref_rmse, 'k--', alpha=0.5, linewidth=1,
               label='1/sqrt(N) reference')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('RMSE')
        ax.set_title('(B) RMSE vs Sample Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 3: CI Width vs N
        ax = axes[0, 2]
        ci_mean = summary['ci_width']['mean'].values
        ci_std = summary['ci_width']['std'].values
        ax.errorbar(N_values, ci_mean, yerr=ci_std, marker='^',
                   markersize=8, linewidth=2, capsize=5, color='green',
                   label='Mean CI width +/- SD')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('95% CI Width')
        ax.set_title('(C) Credible Interval Width vs N')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 4: Coverage Rate
        ax = axes[1, 0]
        coverage = summary['coverage']['mean'].values
        ax.plot(N_values, coverage * 100, marker='D', markersize=8,
               linewidth=2, color='purple', label='Coverage rate')
        ax.axhline(95, color='red', linestyle='--', linewidth=1, label='Nominal 95%')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Coverage Rate (%)')
        ax.set_title('(D) 95% CI Coverage Rate')
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 5: % Incomplete Auctions
        ax = axes[1, 1]
        pct_incomplete = summary['pct_incomplete']['mean'].values
        ax.plot(N_values, pct_incomplete, marker='v', markersize=8,
               linewidth=2, color='brown', label='% Incomplete auctions')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('% Incomplete Auctions')
        ax.set_title('(E) Data Characteristics')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 6: Average Complete Auctions
        ax = axes[1, 2]
        n_complete = summary['n_complete']['mean'].values
        ax.plot(N_values, n_complete, marker='o', markersize=8,
               linewidth=2, color='teal', label='Mean # complete auctions')
        ax.plot(N_values, N_values, 'k--', alpha=0.3, linewidth=1,
               label='N (all complete)')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('# Complete Auctions')
        ax.set_title('(F) Effective Sample Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        plt.savefig(save_path, dpi=300)
        print(f"\nSensitivity plots saved to {save_path}")
        plt.close()


class TaskBSensitivityAnalysis:
    """Sensitivity analysis for Task B (type-specific cutoffs)."""

    def __init__(self, n_replications: int = 10):
        self.n_replications = n_replications

    def run_single_replication(self, dgp_params: TaskBDGPParameters,
                               mcmc_config: MCMCConfig, rep_id: int) -> Dict:
        """Run one replication and return metrics."""
        generator = TaskBDataGenerator(dgp_params)
        auctions, stats = generator.generate_auction_data()

        n_S_bounds = sum(1 for a in auctions if not (np.isinf(a.L_S) and np.isinf(a.U_S)))
        n_F_bounds = sum(1 for a in auctions if not (np.isinf(a.L_F) and np.isinf(a.U_F)))

        if n_S_bounds < 2 or n_F_bounds < 2:
            return {
                'N': dgp_params.N,
                'rep_id': rep_id,
                'n_S_bounds': n_S_bounds,
                'n_F_bounds': n_F_bounds,
                'converged': False,
                'bias_S': np.nan,
                'bias_F': np.nan,
                'bias_gap': np.nan,
                'rmse_S': np.nan,
                'rmse_F': np.nan,
                'rmse_gap': np.nan,
                'ci_width_S': np.nan,
                'ci_width_F': np.nan,
                'coverage_S': False,
                'coverage_F': False,
                'prob_S_greater_F': np.nan,
                'rhat_S': np.nan,
                'rhat_F': np.nan
            }

        sampler = TaskBMCMCSampler(auctions, mcmc_config)
        results = sampler.run()

        mu_S = results['mu_S_samples']
        mu_F = results['mu_F_samples']
        gap = results['gap_samples']

        mu_S_hat = np.mean(mu_S)
        mu_F_hat = np.mean(mu_F)
        gap_hat = np.mean(gap)

        ci_S = np.percentile(mu_S, [2.5, 97.5])
        ci_F = np.percentile(mu_F, [2.5, 97.5])

        true_S, true_F = dgp_params.cutoff_at_mean_x()
        bias_S = mu_S_hat - true_S
        bias_F = mu_F_hat - true_F
        true_gap = true_S - true_F
        bias_gap = gap_hat - true_gap

        rmse_S = np.sqrt(bias_S**2 + np.var(mu_S))
        rmse_F = np.sqrt(bias_F**2 + np.var(mu_F))
        rmse_gap = np.sqrt(bias_gap**2 + np.var(gap))

        coverage_S = (ci_S[0] <= true_S <= ci_S[1])
        coverage_F = (ci_F[0] <= true_F <= ci_F[1])

        prob_S_greater_F = np.mean(gap > 0)

        rhat_S = results['rhat_mu_S']
        rhat_F = results['rhat_mu_F']

        return {
            'N': dgp_params.N,
            'rep_id': rep_id,
            'n_S_bounds': n_S_bounds,
            'n_F_bounds': n_F_bounds,
            'converged': (rhat_S < 1.1 and rhat_F < 1.1),
            'mu_S_hat': mu_S_hat,
            'mu_F_hat': mu_F_hat,
            'gap_hat': gap_hat,
            'bias_S': bias_S,
            'bias_F': bias_F,
            'bias_gap': bias_gap,
            'rmse_S': rmse_S,
            'rmse_F': rmse_F,
            'rmse_gap': rmse_gap,
            'ci_lower_S': ci_S[0],
            'ci_upper_S': ci_S[1],
            'ci_width_S': ci_S[1] - ci_S[0],
            'ci_lower_F': ci_F[0],
            'ci_upper_F': ci_F[1],
            'ci_width_F': ci_F[1] - ci_F[0],
            'coverage_S': coverage_S,
            'coverage_F': coverage_F,
            'prob_S_greater_F': prob_S_greater_F,
            'rhat_S': rhat_S,
            'rhat_F': rhat_F
        }

    def sensitivity_sample_size(self, N_values: List[int] = None,
                                J: int = 3, mu_v: float = 1.3,
                                sigma_v: float = 0.2,
                                b_star_S: float = 1.45,
                                b_star_F: float = 1.35,
                                prob_type_S: float = 0.5) -> pd.DataFrame:
        """Run sample size sensitivity analysis.

        Args:
            N_values: List of sample sizes to test
            J: Number of bidders per auction
            mu_v: Mean valuation
            sigma_v: Std dev of valuation
            b_star_S: True cutoff for type S
            b_star_F: True cutoff for type F
            prob_type_S: Probability of type S

        Returns:
            DataFrame with results from all replications
        """
        if N_values is None:
            N_values = [20, 50, 100, 200]

        print("\n" + "="*70)
        print("TASK B: SAMPLE SIZE SENSITIVITY ANALYSIS")
        print("Type-Specific Cutoffs (S vs F)")
        print("="*70)
        print(f"Replications per N: {self.n_replications}")
        print(f"Sample sizes: {N_values}")
        print()

        mcmc_config = MCMCConfig(
            n_iterations=10000,
            burn_in=5000,
            thinning=10,
            n_chains=2
        )

        results_list = []

        for N in N_values:
            print(f"\nRunning N={N} ({self.n_replications} replications)...")

            dgp_params = TaskBDGPParameters(
                N=N, J=J, mu_v=mu_v, sigma_v=sigma_v,
                b_star_S=b_star_S, b_star_F=b_star_F, prob_type_S=prob_type_S
            )

            for rep in range(self.n_replications):
                print(f"  Rep {rep+1}/{self.n_replications}...", end='', flush=True)
                result = self.run_single_replication(dgp_params, mcmc_config, rep)
                results_list.append(result)

                if result['converged']:
                    print(f" bias_S={result['bias_S']:.4f}, bias_F={result['bias_F']:.4f}, "
                          f"Pr(S>F)={result['prob_S_greater_F']:.2f}")
                else:
                    print(" NOT CONVERGED or insufficient data")

        df = pd.DataFrame(results_list)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS BY SAMPLE SIZE")
        print("="*70)

        summary = df.groupby('N').agg({
            'n_S_bounds': 'mean',
            'n_F_bounds': 'mean',
            'bias_S': ['mean', 'std'],
            'bias_F': ['mean', 'std'],
            'bias_gap': ['mean', 'std'],
            'rmse_S': ['mean', 'std'],
            'rmse_F': ['mean', 'std'],
            'ci_width_S': ['mean', 'std'],
            'ci_width_F': ['mean', 'std'],
            'coverage_S': 'mean',
            'coverage_F': 'mean',
            'prob_S_greater_F': 'mean',
            'converged': 'mean',
            'rhat_S': 'mean',
            'rhat_F': 'mean'
        }).round(4)

        print(summary)

    def plot_results(self, df: pd.DataFrame, save_path: str):
        """Create comprehensive sensitivity plots for Task B."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Task B: Sample Size Sensitivity Analysis (Type-Specific Cutoffs)',
                    fontsize=14, fontweight='bold', y=0.98)

        summary = df.groupby('N').agg({
            'bias_S': ['mean', 'std'],
            'bias_F': ['mean', 'std'],
            'bias_gap': ['mean', 'std'],
            'rmse_S': ['mean', 'std'],
            'rmse_F': ['mean', 'std'],
            'ci_width_S': ['mean', 'std'],
            'ci_width_F': ['mean', 'std'],
            'coverage_S': 'mean',
            'coverage_F': 'mean',
            'prob_S_greater_F': 'mean',
            'n_S_bounds': 'mean',
            'n_F_bounds': 'mean'
        }).reset_index()

        N_values = summary['N'].values

        # Panel 1: Bias for Type S
        ax = axes[0, 0]
        bias_S_mean = summary['bias_S']['mean'].values
        bias_S_std = summary['bias_S']['std'].values
        ax.errorbar(N_values, bias_S_mean, yerr=bias_S_std, marker='o',
                   markersize=8, linewidth=2, capsize=5, color='steelblue',
                   label='Type S bias +/- SD')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Unbiased')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Bias (Type S)')
        ax.set_title('(A) Bias: Type S Cutoff')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 2: Bias for Type F
        ax = axes[0, 1]
        bias_F_mean = summary['bias_F']['mean'].values
        bias_F_std = summary['bias_F']['std'].values
        ax.errorbar(N_values, bias_F_mean, yerr=bias_F_std, marker='s',
                   markersize=8, linewidth=2, capsize=5, color='coral',
                   label='Type F bias +/- SD')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Unbiased')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Bias (Type F)')
        ax.set_title('(B) Bias: Type F Cutoff')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 3: Bias for Gap
        ax = axes[0, 2]
        bias_gap_mean = summary['bias_gap']['mean'].values
        bias_gap_std = summary['bias_gap']['std'].values
        ax.errorbar(N_values, bias_gap_mean, yerr=bias_gap_std, marker='^',
                   markersize=8, linewidth=2, capsize=5, color='purple',
                   label='Gap bias +/- SD')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Unbiased')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Bias (Gap = S - F)')
        ax.set_title('(C) Bias: Cutoff Gap')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 4: RMSE comparison
        ax = axes[1, 0]
        rmse_S_mean = summary['rmse_S']['mean'].values
        rmse_F_mean = summary['rmse_F']['mean'].values
        ax.plot(N_values, rmse_S_mean, marker='o', markersize=8, linewidth=2,
               color='steelblue', label='Type S')
        ax.plot(N_values, rmse_F_mean, marker='s', markersize=8, linewidth=2,
               color='coral', label='Type F')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('RMSE')
        ax.set_title('(D) RMSE Comparison')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 5: CI Width comparison
        ax = axes[1, 1]
        ci_S_mean = summary['ci_width_S']['mean'].values
        ci_F_mean = summary['ci_width_F']['mean'].values
        ax.plot(N_values, ci_S_mean, marker='o', markersize=8, linewidth=2,
               color='steelblue', label='Type S')
        ax.plot(N_values, ci_F_mean, marker='s', markersize=8, linewidth=2,
               color='coral', label='Type F')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('95% CI Width')
        ax.set_title('(E) Credible Interval Width')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 6: Coverage rates
        ax = axes[1, 2]
        coverage_S = summary['coverage_S']['mean'].values * 100
        coverage_F = summary['coverage_F']['mean'].values * 100
        ax.plot(N_values, coverage_S, marker='o', markersize=8, linewidth=2,
               color='steelblue', label='Type S')
        ax.plot(N_values, coverage_F, marker='s', markersize=8, linewidth=2,
               color='coral', label='Type F')
        ax.axhline(95, color='red', linestyle='--', linewidth=1, label='Nominal 95%')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Coverage Rate (%)')
        ax.set_title('(F) Coverage Rate')
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 7: Probability S > F
        ax = axes[2, 0]
        prob_S_greater = summary['prob_S_greater_F']['mean'].values * 100
        ax.plot(N_values, prob_S_greater, marker='D', markersize=8, linewidth=2,
               color='purple', label='Pr(mu_S > mu_F)')
        ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Perfect separation')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Probability (%)')
        ax.set_title('(G) Type Separation: Pr(mu_S > mu_F)')
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 8: Data availability
        ax = axes[2, 1]
        n_S = summary['n_S_bounds']['mean'].values
        n_F = summary['n_F_bounds']['mean'].values
        ax.plot(N_values, n_S, marker='o', markersize=8, linewidth=2,
               color='steelblue', label='# Auctions w/ S info')
        ax.plot(N_values, n_F, marker='s', markersize=8, linewidth=2,
               color='coral', label='# Auctions w/ F info')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('# Auctions with Information')
        ax.set_title('(H) Data Availability by Type')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

        # Panel 9: Summary text
        ax = axes[2, 2]
        ax.axis('off')

        n20 = df[df['N'] == 20]
        if len(n20) > 0:
            summary_text = "N=20 Summary Statistics:\n\n"
            summary_text += f"Type S:\n"
            summary_text += f"  Mean bias: {n20['bias_S'].mean():.4f}\n"
            summary_text += f"  Mean RMSE: {n20['rmse_S'].mean():.4f}\n"
            summary_text += f"  Coverage: {n20['coverage_S'].mean()*100:.1f}%\n\n"
            summary_text += f"Type F:\n"
            summary_text += f"  Mean bias: {n20['bias_F'].mean():.4f}\n"
            summary_text += f"  Mean RMSE: {n20['rmse_F'].mean():.4f}\n"
            summary_text += f"  Coverage: {n20['coverage_F'].mean()*100:.1f}%\n\n"
            summary_text += f"Gap:\n"
            summary_text += f"  Mean bias: {n20['bias_gap'].mean():.4f}\n"
            summary_text += f"  Pr(S>F): {n20['prob_S_greater_F'].mean()*100:.1f}%\n"

            ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        plt.savefig(save_path, dpi=300)
        print(f"\nSensitivity plots saved to {save_path}")
        plt.close()
