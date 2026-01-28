"""
Unified sensitivity analysis framework.

This module provides classes for running sensitivity analyses that
vary sample size N and assess estimation performance for both
Task A (single cutoff) and Task B (two-stage DGP).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

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
                'b_star': dgp_params.b_star,
                'rep_id': rep_id,
                'n_observed': stats.get('n_observed', len(auctions)),
                'n_initiated': stats.get('n_initiated', np.nan),
                'n_dropped_all_reject': stats.get('n_dropped_all_reject', np.nan),
                'keep_rate_pct': stats.get('keep_rate_pct', np.nan),
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
        sampler = TaskAMCMCSampler(
            auctions,
            mcmc_config,
            bid_mu=dgp_params.mu_v,
            bid_sigma=dgp_params.sigma_v,
        )
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
            'b_star': dgp_params.b_star,
            'rep_id': rep_id,
            'n_observed': stats.get('n_observed', len(auctions)),
            'n_initiated': stats.get('n_initiated', np.nan),
            'n_dropped_all_reject': stats.get('n_dropped_all_reject', np.nan),
            'keep_rate_pct': stats.get('keep_rate_pct', np.nan),
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
        print(f"True cutoff (b*): {b_star}")
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
            'keep_rate_pct': 'mean',
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
    """Sensitivity analysis for Task B (two-stage DGP)."""

    def __init__(self, n_replications: int = 10):
        self.n_replications = n_replications

    @staticmethod
    def _beta_names(k: int) -> List[str]:
        if k == 1:
            return ["c"]
        if k == 3:
            return ["c", "depth_mean_23", "depth_gap_23"]
        if k == 4:
            return ["c", "theta1", "theta2", "theta3"]
        return [f"beta_{j}" for j in range(k)]

    @staticmethod
    def _summarize_samples(samples: np.ndarray, true_value: float) -> Dict[str, float]:
        est = float(np.mean(samples))
        ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
        bias = est - float(true_value)
        rmse = float(np.sqrt(np.mean((samples - float(true_value)) ** 2)))
        return {
            "estimate": est,
            "bias": bias,
            "rmse": rmse,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_upper - ci_lower),
            "coverage": bool(ci_lower <= true_value <= ci_upper),
        }

    def run_single_replication(
        self,
        dgp_params: TaskBDGPParameters,
        mcmc_config: MCMCConfig,
        rep_id: int,
    ) -> Dict:
        generator = TaskBDataGenerator(dgp_params)
        auctions, stats = generator.generate_auction_data()

        n_complete = sum(1 for a in auctions if a.is_complete)
        pct_incomplete = 100 * (1 - n_complete / len(auctions)) if auctions else 0.0

        if len(auctions) < 2:
            return {
                'N': dgp_params.N,
                'J': dgp_params.J,
                'rep_id': rep_id,
                'n_observed': stats.get('n_observed', len(auctions)),
                'n_initiated': stats.get('n_initiated', np.nan),
                'n_dropped_all_reject': stats.get('n_dropped_all_reject', np.nan),
                'keep_rate_pct': stats.get('keep_rate_pct', np.nan),
                'n_complete': n_complete,
                'pct_incomplete': pct_incomplete,
                'converged': False,
                'bias_gamma': np.nan,
                'bias_tilde_alpha': np.nan,
                'bias_cutoff_c': np.nan,
                'bias_theta1': np.nan,
                'bias_theta2': np.nan,
                'bias_theta3': np.nan,
                'bias_depth_mean_23': np.nan,
                'bias_depth_gap_23': np.nan,
                'rmse_gamma': np.nan,
                'rmse_tilde_alpha': np.nan,
                'rmse_cutoff_c': np.nan,
                'rmse_theta1': np.nan,
                'rmse_theta2': np.nan,
                'rmse_theta3': np.nan,
                'rmse_depth_mean_23': np.nan,
                'rmse_depth_gap_23': np.nan,
                'ci_width_gamma': np.nan,
                'ci_width_tilde_alpha': np.nan,
                'ci_width_cutoff_c': np.nan,
                'ci_width_theta1': np.nan,
                'ci_width_theta2': np.nan,
                'ci_width_theta3': np.nan,
                'ci_width_depth_mean_23': np.nan,
                'ci_width_depth_gap_23': np.nan,
                'coverage_gamma': False,
                'coverage_tilde_alpha': False,
                'coverage_cutoff_c': False,
                'coverage_theta1': False,
                'coverage_theta2': False,
                'coverage_theta3': False,
                'coverage_depth_mean_23': False,
                'coverage_depth_gap_23': False,
                'bias_sigma_omega': np.nan,
                'rmse_sigma_omega': np.nan,
                'ci_width_sigma_omega': np.nan,
                'coverage_sigma_omega': False,
                'bias_sigma_eta': np.nan,
                'rmse_sigma_eta': np.nan,
                'ci_width_sigma_eta': np.nan,
                'coverage_sigma_eta': False,
                'bias_sigma_nu': np.nan,
                'rmse_sigma_nu': np.nan,
                'ci_width_sigma_nu': np.nan,
                'coverage_sigma_nu': False,
                'rhat_gamma': np.nan,
                'rhat_kappa': np.nan,
                'rhat_beta': np.nan,
            }

        sampler = TaskBMCMCSampler(auctions, mcmc_config)
        results = sampler.run()

        gamma = results["gamma_samples"]
        tilde_alpha = results["tilde_alpha_samples"]
        beta = results["beta_samples"]
        cutoff_c = results.get("cutoff_c_samples", beta[:, 0])

        sigma_omega = results.get("sigma_omega_samples")
        sigma_eta = results.get("sigma_eta_samples")
        sigma_nu = results.get("sigma_nu_samples")

        true_gamma = float(dgp_params.gamma)
        true_alpha = float(dgp_params.tilde_alpha)
        true_beta = np.atleast_1d(dgp_params.beta_cutoff).astype(float)
        true_sigma_omega = float(dgp_params.sigma_omega)
        true_sigma_eta = float(dgp_params.sigma_eta)
        true_sigma_nu = float(dgp_params.sigma_nu)

        gamma_m = self._summarize_samples(gamma, true_gamma)
        alpha_m = self._summarize_samples(tilde_alpha, true_alpha)
        c_m = self._summarize_samples(cutoff_c, float(true_beta[0]))

        k = int(beta.shape[1]) if beta.ndim == 2 else 1
        if true_beta.size != k:
            raise ValueError(f"True beta_cutoff has length {true_beta.size}, but sampler has k={k}")

        beta_metrics = {name: self._summarize_samples(beta[:, j], float(true_beta[j]))
                        for j, name in enumerate(self._beta_names(k))}

        sigma_omega_m = self._summarize_samples(sigma_omega, true_sigma_omega) if sigma_omega is not None else None
        sigma_eta_m = self._summarize_samples(sigma_eta, true_sigma_eta) if sigma_eta is not None else None
        sigma_nu_m = self._summarize_samples(sigma_nu, true_sigma_nu) if sigma_nu is not None else None

        rhat_beta = float(results['rhat_beta'])
        rhat_gamma = float(results['rhat_gamma'])
        rhat_kappa = float(results['rhat_kappa'])

        return {
            'N': dgp_params.N,
            'J': dgp_params.J,
            'rep_id': rep_id,
            'n_observed': stats.get('n_observed', len(auctions)),
            'n_initiated': stats.get('n_initiated', np.nan),
            'n_dropped_all_reject': stats.get('n_dropped_all_reject', np.nan),
            'keep_rate_pct': stats.get('keep_rate_pct', np.nan),
            'n_complete': n_complete,
            'pct_incomplete': pct_incomplete,
            'converged': (rhat_beta < 1.1 and rhat_gamma < 1.1 and rhat_kappa < 1.1),
            'bias_gamma': gamma_m["bias"],
            'bias_tilde_alpha': alpha_m["bias"],
            'bias_cutoff_c': c_m["bias"],
            'rmse_gamma': gamma_m["rmse"],
            'rmse_tilde_alpha': alpha_m["rmse"],
            'rmse_cutoff_c': c_m["rmse"],
            'ci_width_gamma': gamma_m["ci_width"],
            'ci_width_tilde_alpha': alpha_m["ci_width"],
            'ci_width_cutoff_c': c_m["ci_width"],
            'coverage_gamma': gamma_m["coverage"],
            'coverage_tilde_alpha': alpha_m["coverage"],
            'coverage_cutoff_c': c_m["coverage"],
            'bias_theta1': beta_metrics.get("theta1", {}).get("bias", np.nan),
            'bias_theta2': beta_metrics.get("theta2", {}).get("bias", np.nan),
            'bias_theta3': beta_metrics.get("theta3", {}).get("bias", np.nan),
            'bias_depth_mean_23': beta_metrics.get("depth_mean_23", {}).get("bias", np.nan),
            'bias_depth_gap_23': beta_metrics.get("depth_gap_23", {}).get("bias", np.nan),
            'rmse_theta1': beta_metrics.get("theta1", {}).get("rmse", np.nan),
            'rmse_theta2': beta_metrics.get("theta2", {}).get("rmse", np.nan),
            'rmse_theta3': beta_metrics.get("theta3", {}).get("rmse", np.nan),
            'rmse_depth_mean_23': beta_metrics.get("depth_mean_23", {}).get("rmse", np.nan),
            'rmse_depth_gap_23': beta_metrics.get("depth_gap_23", {}).get("rmse", np.nan),
            'ci_width_theta1': beta_metrics.get("theta1", {}).get("ci_width", np.nan),
            'ci_width_theta2': beta_metrics.get("theta2", {}).get("ci_width", np.nan),
            'ci_width_theta3': beta_metrics.get("theta3", {}).get("ci_width", np.nan),
            'ci_width_depth_mean_23': beta_metrics.get("depth_mean_23", {}).get("ci_width", np.nan),
            'ci_width_depth_gap_23': beta_metrics.get("depth_gap_23", {}).get("ci_width", np.nan),
            'coverage_theta1': beta_metrics.get("theta1", {}).get("coverage", False),
            'coverage_theta2': beta_metrics.get("theta2", {}).get("coverage", False),
            'coverage_theta3': beta_metrics.get("theta3", {}).get("coverage", False),
            'coverage_depth_mean_23': beta_metrics.get("depth_mean_23", {}).get("coverage", False),
            'coverage_depth_gap_23': beta_metrics.get("depth_gap_23", {}).get("coverage", False),
            'bias_sigma_omega': (np.nan if sigma_omega_m is None else sigma_omega_m["bias"]),
            'rmse_sigma_omega': (np.nan if sigma_omega_m is None else sigma_omega_m["rmse"]),
            'ci_width_sigma_omega': (np.nan if sigma_omega_m is None else sigma_omega_m["ci_width"]),
            'coverage_sigma_omega': (False if sigma_omega_m is None else sigma_omega_m["coverage"]),
            'bias_sigma_eta': (np.nan if sigma_eta_m is None else sigma_eta_m["bias"]),
            'rmse_sigma_eta': (np.nan if sigma_eta_m is None else sigma_eta_m["rmse"]),
            'ci_width_sigma_eta': (np.nan if sigma_eta_m is None else sigma_eta_m["ci_width"]),
            'coverage_sigma_eta': (False if sigma_eta_m is None else sigma_eta_m["coverage"]),
            'bias_sigma_nu': (np.nan if sigma_nu_m is None else sigma_nu_m["bias"]),
            'rmse_sigma_nu': (np.nan if sigma_nu_m is None else sigma_nu_m["rmse"]),
            'ci_width_sigma_nu': (np.nan if sigma_nu_m is None else sigma_nu_m["ci_width"]),
            'coverage_sigma_nu': (False if sigma_nu_m is None else sigma_nu_m["coverage"]),
            'rhat_beta': rhat_beta,
            'rhat_gamma': rhat_gamma,
            'rhat_kappa': rhat_kappa,
        }

    def sensitivity_sample_size(
        self,
        N_values: Optional[List[int]] = None,
        *,
        J: int = 3,
        gamma: float = 1.3,
        sigma_nu: float = 0.2,
        sigma_eta: float = 0.1,
        kappa: float = float(np.log(1.5)),
        misreporting_mode: str = "scale",
        beta_cutoff: Optional[np.ndarray] = None,
        cutoff_c: float = 1.4,
        cutoff_spec: Optional[str] = None,
        sigma_omega: float = 0.1,
        stage: int = 1,
    ) -> pd.DataFrame:
        if N_values is None:
            N_values = [20, 50, 100, 200]

        print("\n" + "="*70)
        print("TASK B: TWO-STAGE SAMPLE SIZE SENSITIVITY")
        print("="*70)
        print(f"Replications per N: {self.n_replications}")
        print(f"Sample sizes: {N_values}")
        print(f"J (bidders per auction): {J}")
        print(f"Stage: {stage}")
        print(f"Misreporting mode: {misreporting_mode}")
        print()

        mcmc_config = MCMCConfig(
            n_iterations=8000,
            burn_in=4000,
            thinning=10,
            n_chains=2,
            task_b_stage=stage,
            task_b_misreporting_mode=misreporting_mode,
            task_b_sigma_nu_fixed=sigma_nu,
            task_b_sigma_eta_fixed=sigma_eta,
            task_b_kappa_init=kappa,
        )

        beta_cutoff_arr = (
            np.array([cutoff_c], dtype=float)
            if beta_cutoff is None
            else np.atleast_1d(beta_cutoff).astype(float)
        )

        results_list = []
        for N in N_values:
            print(f"\nRunning N={N} ({self.n_replications} replications)...")

            dgp_params = TaskBDGPParameters(
                N=N,
                J=J,
                gamma=gamma,
                sigma_nu=sigma_nu,
                sigma_eta=sigma_eta,
                kappa=kappa,
                misreporting_mode=misreporting_mode,
                cutoff_spec=cutoff_spec,
                beta_cutoff=beta_cutoff_arr,
                sigma_omega=sigma_omega,
            )

            for rep in range(self.n_replications):
                print(f"  Rep {rep+1}/{self.n_replications}...", end='', flush=True)
                result = self.run_single_replication(dgp_params, mcmc_config, rep)
                results_list.append(result)
                if result['converged']:
                    print(
                        f" bias(gamma)={result['bias_gamma']:.4f}, "
                        f"bias(alpha)={result['bias_tilde_alpha']:.4f}, "
                        f"keep={result['keep_rate_pct']:.1f}%"
                    )
                else:
                    print(" NOT CONVERGED or insufficient data")

        df = pd.DataFrame(results_list)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        print("\n" + "="*70)
        print("SUMMARY STATISTICS BY SAMPLE SIZE")
        print("="*70)

        summary = df.groupby('N').agg({
            'keep_rate_pct': ['mean', 'std'],
            'pct_incomplete': ['mean', 'std'],
            'bias_gamma': ['mean', 'std'],
            'bias_tilde_alpha': ['mean', 'std'],
            'bias_cutoff_c': ['mean', 'std'],
            'bias_theta1': ['mean', 'std'],
            'bias_theta2': ['mean', 'std'],
            'bias_theta3': ['mean', 'std'],
            'bias_depth_mean_23': ['mean', 'std'],
            'bias_depth_gap_23': ['mean', 'std'],
            'bias_sigma_omega': ['mean', 'std'],
            'bias_sigma_eta': ['mean', 'std'],
            'bias_sigma_nu': ['mean', 'std'],
            'rmse_gamma': ['mean', 'std'],
            'rmse_tilde_alpha': ['mean', 'std'],
            'rmse_cutoff_c': ['mean', 'std'],
            'rmse_theta1': ['mean', 'std'],
            'rmse_theta2': ['mean', 'std'],
            'rmse_theta3': ['mean', 'std'],
            'rmse_depth_mean_23': ['mean', 'std'],
            'rmse_depth_gap_23': ['mean', 'std'],
            'rmse_sigma_omega': ['mean', 'std'],
            'rmse_sigma_eta': ['mean', 'std'],
            'rmse_sigma_nu': ['mean', 'std'],
            'ci_width_gamma': ['mean', 'std'],
            'ci_width_tilde_alpha': ['mean', 'std'],
            'ci_width_cutoff_c': ['mean', 'std'],
            'ci_width_theta1': ['mean', 'std'],
            'ci_width_theta2': ['mean', 'std'],
            'ci_width_theta3': ['mean', 'std'],
            'ci_width_depth_mean_23': ['mean', 'std'],
            'ci_width_depth_gap_23': ['mean', 'std'],
            'ci_width_sigma_omega': ['mean', 'std'],
            'ci_width_sigma_eta': ['mean', 'std'],
            'ci_width_sigma_nu': ['mean', 'std'],
            'coverage_gamma': 'mean',
            'coverage_tilde_alpha': 'mean',
            'coverage_cutoff_c': 'mean',
            'coverage_theta1': 'mean',
            'coverage_theta2': 'mean',
            'coverage_theta3': 'mean',
            'coverage_depth_mean_23': 'mean',
            'coverage_depth_gap_23': 'mean',
            'coverage_sigma_omega': 'mean',
            'coverage_sigma_eta': 'mean',
            'coverage_sigma_nu': 'mean',
            'converged': 'mean',
            'rhat_gamma': 'mean',
            'rhat_kappa': 'mean',
        }).round(4)

        print(summary)

    def plot_results(self, df: pd.DataFrame, save_path: str):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle('Task B: Two-Stage Sample Size Sensitivity', fontsize=14, fontweight='bold', y=0.98)

        summary = df.groupby('N').agg({
            'bias_gamma': ['mean', 'std'],
            'bias_tilde_alpha': ['mean', 'std'],
            'bias_cutoff_c': ['mean', 'std'],
            'rmse_gamma': ['mean', 'std'],
            'rmse_tilde_alpha': ['mean', 'std'],
            'rmse_cutoff_c': ['mean', 'std'],
        }).reset_index()

        N_values = summary['N'].values

        # Bias panels
        for ax, key, title in [
            (axes[0, 0], 'bias_gamma', 'Bias: gamma'),
            (axes[0, 1], 'bias_tilde_alpha', 'Bias: tilde_alpha'),
            (axes[0, 2], 'bias_cutoff_c', 'Bias: cutoff c'),
        ]:
            mean = summary[key]['mean'].values
            std = summary[key]['std'].values
            ax.errorbar(N_values, mean, yerr=std, marker='o', linewidth=2, capsize=4)
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('N')

        # RMSE panels
        for ax, key, title in [
            (axes[1, 0], 'rmse_gamma', 'RMSE: gamma'),
            (axes[1, 1], 'rmse_tilde_alpha', 'RMSE: tilde_alpha'),
            (axes[1, 2], 'rmse_cutoff_c', 'RMSE: cutoff c'),
        ]:
            mean = summary[key]['mean'].values
            ax.plot(N_values, mean, marker='o', linewidth=2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('N')

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        plt.savefig(save_path, dpi=300)
        print(f"\nSensitivity plots saved to {save_path}")
        plt.close()

        # Optional: cutoff moments coefficients (theta1-3), if present.
        has_theta = (
            df[["bias_theta1", "bias_theta2", "bias_theta3"]]
            .notna()
            .to_numpy()
            .any()
        )
        if has_theta:
            betas_path = save_path.replace(".png", "_cutoff_betas.png")
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            fig.suptitle('Task B: Cutoff Moments Coefficients (theta) Sensitivity', fontsize=14, fontweight='bold', y=0.98)

            summary = df.groupby("N").agg({
                "bias_theta1": ["mean", "std"],
                "bias_theta2": ["mean", "std"],
                "bias_theta3": ["mean", "std"],
                "rmse_theta1": ["mean", "std"],
                "rmse_theta2": ["mean", "std"],
                "rmse_theta3": ["mean", "std"],
            }).reset_index()

            N_values = summary["N"].values

            for ax, key, title in [
                (axes[0, 0], "bias_theta1", "Bias: theta1 (top-1)"),
                (axes[0, 1], "bias_theta2", "Bias: theta2 (top-2 avg)"),
                (axes[0, 2], "bias_theta3", "Bias: theta3 (top-3 avg)"),
            ]:
                mean = summary[key]["mean"].values
                std = summary[key]["std"].values
                ax.errorbar(N_values, mean, yerr=std, marker="o", linewidth=2, capsize=4)
                ax.axhline(0, color="red", linestyle="--", linewidth=1)
                ax.set_xscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            for ax, key, title in [
                (axes[1, 0], "rmse_theta1", "RMSE: theta1"),
                (axes[1, 1], "rmse_theta2", "RMSE: theta2"),
                (axes[1, 2], "rmse_theta3", "RMSE: theta3"),
            ]:
                mean = summary[key]["mean"].values
                ax.plot(N_values, mean, marker="o", linewidth=2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            plt.savefig(betas_path, dpi=300)
            print(f"Sensitivity plots saved to {betas_path}")
            plt.close()

        has_depth = (
            df[["bias_depth_mean_23", "bias_depth_gap_23"]]
            .notna()
            .to_numpy()
            .any()
        )
        if has_depth:
            depth_path = save_path.replace(".png", "_cutoff_depth.png")
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            fig.suptitle('Task B: Depth Cutoff Coefficients Sensitivity', fontsize=14, fontweight='bold', y=0.98)

            summary = df.groupby("N").agg({
                "bias_depth_mean_23": ["mean", "std"],
                "bias_depth_gap_23": ["mean", "std"],
                "rmse_depth_mean_23": ["mean", "std"],
                "rmse_depth_gap_23": ["mean", "std"],
            }).reset_index()

            N_values = summary["N"].values

            for ax, key, title in [
                (axes[0, 0], "bias_depth_mean_23", "Bias: depth mean (b2+b3)/2"),
                (axes[0, 1], "bias_depth_gap_23", "Bias: depth gap (b2-b3)"),
            ]:
                mean = summary[key]["mean"].values
                std = summary[key]["std"].values
                ax.errorbar(N_values, mean, yerr=std, marker="o", linewidth=2, capsize=4)
                ax.axhline(0, color="red", linestyle="--", linewidth=1)
                ax.set_xscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            for ax, key, title in [
                (axes[1, 0], "rmse_depth_mean_23", "RMSE: depth mean"),
                (axes[1, 1], "rmse_depth_gap_23", "RMSE: depth gap"),
            ]:
                mean = summary[key]["mean"].values
                ax.plot(N_values, mean, marker="o", linewidth=2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            plt.savefig(depth_path, dpi=300)
            print(f"Sensitivity plots saved to {depth_path}")
            plt.close()

        # Optional: sigma panels for stages that estimate variances.
        has_sigma = (
            df[["ci_width_sigma_eta", "ci_width_sigma_nu"]]
            .fillna(0.0)
            .to_numpy()
            .max()
            > 1e-9
        )
        if has_sigma:
            sigmas_path = save_path.replace(".png", "_sigmas.png")
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            fig.suptitle('Task B: Variance Parameters Sensitivity', fontsize=14, fontweight='bold', y=0.98)

            summary = df.groupby("N").agg({
                "bias_sigma_omega": ["mean", "std"],
                "bias_sigma_eta": ["mean", "std"],
                "bias_sigma_nu": ["mean", "std"],
                "rmse_sigma_omega": ["mean", "std"],
                "rmse_sigma_eta": ["mean", "std"],
                "rmse_sigma_nu": ["mean", "std"],
            }).reset_index()

            N_values = summary["N"].values

            for ax, key, title in [
                (axes[0, 0], "bias_sigma_omega", "Bias: sigma_omega"),
                (axes[0, 1], "bias_sigma_eta", "Bias: sigma_eta"),
                (axes[0, 2], "bias_sigma_nu", "Bias: sigma_nu"),
            ]:
                mean = summary[key]["mean"].values
                std = summary[key]["std"].values
                ax.errorbar(N_values, mean, yerr=std, marker="o", linewidth=2, capsize=4)
                ax.axhline(0, color="red", linestyle="--", linewidth=1)
                ax.set_xscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            for ax, key, title in [
                (axes[1, 0], "rmse_sigma_omega", "RMSE: sigma_omega"),
                (axes[1, 1], "rmse_sigma_eta", "RMSE: sigma_eta"),
                (axes[1, 2], "rmse_sigma_nu", "RMSE: sigma_nu"),
            ]:
                mean = summary[key]["mean"].values
                ax.plot(N_values, mean, marker="o", linewidth=2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("N")

            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            plt.savefig(sigmas_path, dpi=300)
            print(f"Sensitivity plots saved to {sigmas_path}")
            plt.close()
