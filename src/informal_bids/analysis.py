"""
Results analysis and metrics computation.

This module contains classes for analyzing MCMC results and computing
performance metrics for both Task A (single cutoff) and Task B (type-specific).
"""

import numpy as np
from typing import Dict


def compute_parameter_metrics(samples: np.ndarray, true_value: float) -> Dict:
    """Compute standard metrics for a single parameter.

    Args:
        samples: Posterior samples
        true_value: True parameter value

    Returns:
        Dictionary with estimate, bias, RMSE, CI, coverage
    """
    estimate = np.mean(samples)
    ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
    return {
        'estimate': estimate,
        'median': np.median(samples),
        'bias': estimate - true_value,
        'rmse': np.sqrt(np.mean((samples - true_value) ** 2)),
        'posterior_sd': np.std(samples),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'coverage': (ci_lower <= true_value <= ci_upper)
    }


class TaskAResultsAnalyzer:
    """Analyze MCMC results for Task A (single cutoff)."""

    def __init__(self, mcmc_results: Dict, true_b_star: float):
        """Initialize analyzer.

        Args:
            mcmc_results: Output from TaskAMCMCSampler.run()
            true_b_star: True cutoff value
        """
        self.results = mcmc_results
        self.true_b_star = true_b_star

    def compute_metrics(self) -> Dict:
        """Compute performance metrics.

        Returns:
            Dictionary with mu_hat, bias, rmse, CI, coverage, R-hat
        """
        mu_samples = self.results['mu_samples']

        mu_hat = np.mean(mu_samples)
        mu_median = np.median(mu_samples)
        bias = mu_hat - self.true_b_star
        rmse = np.sqrt(np.mean((mu_samples - self.true_b_star) ** 2))
        ci_lower, ci_upper = np.percentile(mu_samples, [2.5, 97.5])
        coverage = (ci_lower <= self.true_b_star <= ci_upper)

        return {
            'mu_hat': mu_hat,
            'mu_median': mu_median,
            'bias': bias,
            'rmse': rmse,
            'posterior_sd': np.std(mu_samples),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'coverage': coverage,
            'rhat_mu': self.results['rhat_mu'],
            'rhat_sigma': self.results['rhat_sigma']
        }

    def print_summary(self, metrics: Dict):
        """Print formatted summary."""
        print("\n" + "="*60)
        print("MCMC ESTIMATION RESULTS")
        print("="*60)
        print(f"True cutoff:          b* = {self.true_b_star:.4f}")
        print(f"Posterior mean:       mu = {metrics['mu_hat']:.4f}")
        print(f"Bias:                     {metrics['bias']:+.4f}")
        print(f"RMSE:                     {metrics['rmse']:.4f}")
        print(f"95% CI:               [{metrics['ci_lower']:.4f}, {metrics['ci_upper']:.4f}]")
        print(f"Coverage:                 {metrics['coverage']}")
        rhat_ok = metrics['rhat_mu'] < 1.1
        print(f"R-hat (mu):               {metrics['rhat_mu']:.4f} {'OK' if rhat_ok else 'WARNING'}")
        print("="*60 + "\n")


class TaskBResultsAnalyzer:
    """Analyze MCMC results for Task B (type-specific cutoffs)."""

    def __init__(self, mcmc_results: Dict, true_b_star_S: float, true_b_star_F: float):
        """Initialize analyzer.

        Args:
            mcmc_results: Output from TaskBMCMCSampler.run()
            true_b_star_S: True cutoff for type S
            true_b_star_F: True cutoff for type F
        """
        self.results = mcmc_results
        self.true_S = true_b_star_S
        self.true_F = true_b_star_F
        self.true_gap = true_b_star_S - true_b_star_F

    def compute_metrics(self) -> Dict:
        """Compute performance metrics for both types.

        Returns:
            Dictionary with metrics for S, F, and gap
        """
        mu_S = self.results['mu_S_samples']
        mu_F = self.results['mu_F_samples']
        gap = self.results['gap_samples']

        # Type S metrics
        mu_S_hat = np.mean(mu_S)
        bias_S = mu_S_hat - self.true_S
        rmse_S = np.sqrt(np.mean((mu_S - self.true_S) ** 2))
        ci_S = np.percentile(mu_S, [2.5, 97.5])
        coverage_S = (ci_S[0] <= self.true_S <= ci_S[1])

        # Type F metrics
        mu_F_hat = np.mean(mu_F)
        bias_F = mu_F_hat - self.true_F
        rmse_F = np.sqrt(np.mean((mu_F - self.true_F) ** 2))
        ci_F = np.percentile(mu_F, [2.5, 97.5])
        coverage_F = (ci_F[0] <= self.true_F <= ci_F[1])

        # Gap metrics
        gap_hat = np.mean(gap)
        bias_gap = gap_hat - self.true_gap
        rmse_gap = np.sqrt(np.mean((gap - self.true_gap) ** 2))
        ci_gap = np.percentile(gap, [2.5, 97.5])
        coverage_gap = (ci_gap[0] <= self.true_gap <= ci_gap[1])
        prob_S_greater = np.mean(gap > 0)

        return {
            'mu_S_hat': mu_S_hat,
            'bias_S': bias_S,
            'rmse_S': rmse_S,
            'ci_S': ci_S,
            'coverage_S': coverage_S,
            'mu_F_hat': mu_F_hat,
            'bias_F': bias_F,
            'rmse_F': rmse_F,
            'ci_F': ci_F,
            'coverage_F': coverage_F,
            'gap_hat': gap_hat,
            'bias_gap': bias_gap,
            'rmse_gap': rmse_gap,
            'ci_gap': ci_gap,
            'coverage_gap': coverage_gap,
            'prob_S_greater': prob_S_greater,
            'rhat_mu_S': self.results['rhat_mu_S'],
            'rhat_mu_F': self.results['rhat_mu_F']
        }

    def print_summary(self, metrics: Dict):
        """Print formatted summary."""
        print("\n" + "="*60)
        print("TASK B: TYPE-SPECIFIC CUTOFFS RESULTS")
        print("="*60)
        print(f"TYPE S:")
        print(f"  True b*_S:        {self.true_S:.4f}")
        print(f"  Estimated mu_S:   {metrics['mu_S_hat']:.4f}")
        print(f"  Bias:             {metrics['bias_S']:+.4f}")
        print(f"  RMSE:             {metrics['rmse_S']:.4f}")
        print(f"  95% CI:           [{metrics['ci_S'][0]:.4f}, {metrics['ci_S'][1]:.4f}]")
        print(f"  Coverage:         {metrics['coverage_S']}")
        print(f"  R-hat:            {metrics['rhat_mu_S']:.4f}")
        print(f"\nTYPE F:")
        print(f"  True b*_F:        {self.true_F:.4f}")
        print(f"  Estimated mu_F:   {metrics['mu_F_hat']:.4f}")
        print(f"  Bias:             {metrics['bias_F']:+.4f}")
        print(f"  RMSE:             {metrics['rmse_F']:.4f}")
        print(f"  95% CI:           [{metrics['ci_F'][0]:.4f}, {metrics['ci_F'][1]:.4f}]")
        print(f"  Coverage:         {metrics['coverage_F']}")
        print(f"  R-hat:            {metrics['rhat_mu_F']:.4f}")
        print(f"\nGAP (b*_S - b*_F):")
        print(f"  True gap:         {self.true_gap:.4f}")
        print(f"  Estimated gap:    {metrics['gap_hat']:.4f}")
        print(f"  Bias:             {metrics['bias_gap']:+.4f}")
        print(f"  Pr(mu_S > mu_F):  {metrics['prob_S_greater']:.4f}")
        print("="*60 + "\n")
