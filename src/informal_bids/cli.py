"""
Command-line interface entry points.

These functions are registered as console scripts in pyproject.toml.
Usage after installing the package:
    task-a-baseline
    task-b-baseline
    task-a-sensitivity
    task-b-sensitivity
"""

import numpy as np

from .config import (
    configure_plotting, TASK_A_BASELINE_DIR, TASK_B_BASELINE_DIR,
    TASK_A_SENSITIVITY_DIR, TASK_B_SENSITIVITY_DIR
)
from .data import TaskADGPParameters, TaskBDGPParameters, MCMCConfig, TaskADataGenerator, TaskBDataGenerator
from .samplers import TaskAMCMCSampler, TaskBMCMCSampler
from .analysis import TaskAResultsAnalyzer, TaskBResultsAnalyzer
from .visualization import TaskAVisualizer, TaskBVisualizer
from .sensitivity import TaskASensitivityAnalysis, TaskBSensitivityAnalysis


def task_a_baseline():
    """Run Task A baseline simulation."""
    configure_plotting()
    np.random.seed(42)

    print("\n" + "="*70)
    print("TASK A: BASELINE SIMULATION")
    print("Single Constant Cutoff Estimation via MCMC")
    print("="*70)

    TASK_A_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    dgp_params = TaskADGPParameters(
        N=100,
        J=3,
        mu_v=1.3,
        sigma_v=0.2,
        b_star=1.4
    )

    mcmc_config = MCMCConfig(
        n_iterations=20000,
        burn_in=10000,
        thinning=10,
        n_chains=3
    )

    print(f"\n{dgp_params}")

    generator = TaskADataGenerator(dgp_params)
    auctions, summary = generator.generate_auction_data()

    print(f"Generated {summary['n_observed']} observed auctions (formal stage):")
    print(f"  Two-sided:  {summary['n_complete']} ({100*summary['n_complete']/summary['n_observed']:.1f}%)")
    print(f"  One-sided:  {summary['n_incomplete']} ({summary['pct_incomplete']:.1f}%)")
    print(f"  Initiated:  {summary['n_initiated']} (dropped all-reject: {summary['n_dropped_all_reject']}, keep rate: {summary['keep_rate_pct']:.1f}%)")

    sampler = TaskAMCMCSampler(
        auctions,
        mcmc_config,
        bid_mu=dgp_params.mu_v,
        bid_sigma=dgp_params.sigma_v,
    )
    results = sampler.run()

    analyzer = TaskAResultsAnalyzer(results, dgp_params.cutoff_at_mean_x())
    metrics = analyzer.compute_metrics()
    analyzer.print_summary(metrics)

    print("\nGenerating plots...")
    TaskAVisualizer.plot_diagnostics(
        results,
        dgp_params.cutoff_at_mean_x(),
        str(TASK_A_BASELINE_DIR / "task_a_diagnostics.png")
    )

    TaskAVisualizer.plot_intervals(
        auctions,
        dgp_params.cutoff_at_mean_x(),
        metrics['mu_hat'],
        str(TASK_A_BASELINE_DIR / "task_a_intervals.png")
    )

    print("\nBaseline simulation complete!")
    return metrics, results


def task_b_baseline():
    """Run Task B baseline simulation."""
    configure_plotting()
    np.random.seed(123)

    print("\n" + "="*70)
    print("TASK B: BASELINE SIMULATION")
    print("Type-Specific Cutoffs (S vs F)")
    print("="*70)

    TASK_B_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    dgp_params = TaskBDGPParameters(
        N=100,
        J=3,
        mu_v=1.3,
        sigma_v=0.2,
        b_star_S=1.45,
        b_star_F=1.35,
        prob_type_S=0.5
    )

    print(f"\n{dgp_params}")

    generator = TaskBDataGenerator(dgp_params)
    auctions, summary = generator.generate_auction_data()

    print(f"\nGenerated {summary['n_observed']} observed auctions (formal stage):")
    print(f"  Two-sided (both S & F): {summary['n_complete_both']}")
    print(f"  Two-sided S bounds: {summary['n_complete_S_only']}")
    print(f"  Two-sided F bounds: {summary['n_complete_F_only']}")
    print(f"  Initiated:  {summary['n_initiated']} (dropped zero-admitted: {summary['n_dropped_no_admitted']}, keep rate: {summary['keep_rate_pct']:.1f}%)")

    mcmc_config = MCMCConfig()
    sampler = TaskBMCMCSampler(auctions, mcmc_config)
    results = sampler.run()

    true_S, true_F = dgp_params.cutoff_at_mean_x()
    analyzer = TaskBResultsAnalyzer(results, true_S, true_F)
    metrics = analyzer.compute_metrics()
    analyzer.print_summary(metrics)

    print("\nGenerating visualization plots...")
    TaskBVisualizer.plot_diagnostics(
        results,
        true_S,
        true_F,
        str(TASK_B_BASELINE_DIR / "task_b_diagnostics.png"),
    )

    TaskBVisualizer.plot_type_intervals(
        auctions,
        true_S,
        true_F,
        metrics['mu_S_hat'],
        metrics['mu_F_hat'],
        str(TASK_B_BASELINE_DIR / "task_b_intervals.png"),
    )

    print("\nTask B baseline complete!")
    return metrics, results, auctions


def task_a_sensitivity():
    """Run Task A sensitivity analysis."""
    configure_plotting()
    np.random.seed(2024)

    TASK_A_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    analysis = TaskASensitivityAnalysis(n_replications=10)

    # Meeting (2026-01-14) conversion diagnostic: higher conversion via lower true cutoffs.
    for b_star in (1.2, 1.1):
        df = analysis.sensitivity_sample_size(b_star=b_star)
        tag = str(b_star).replace('.', 'p')

        # Save CSV
        csv_path = TASK_A_SENSITIVITY_DIR / f"sample_size_sensitivity_bstar_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Generate plots
        plot_path = TASK_A_SENSITIVITY_DIR / f"sample_size_sensitivity_bstar_{tag}.png"
        analysis.plot_results(df, str(plot_path))

        # Save summary (include conversion diagnostics)
        summary_path = TASK_A_SENSITIVITY_DIR / f"summary_table_bstar_{tag}.csv"
        summary = df.groupby('N').agg({
            'n_complete': ['mean', 'std'],
            'pct_incomplete': ['mean', 'std'],
            'keep_rate_pct': ['mean', 'std'],
            'n_initiated': ['mean', 'std'],
            'bias': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'ci_width': ['mean', 'std'],
            'coverage': 'mean',
            'rhat': 'mean'
        }).round(4)
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")

        # Highlight N=20 results
        print("\n" + "="*70)
        print(f"N=20 STRESS TEST RESULTS (b*={b_star})")
        print("="*70)
        n20_results = df[df['N'] == 20]
        print(f"Mean keep rate (conversion): {n20_results['keep_rate_pct'].mean():.1f}%")
        print(f"Mean bias: {n20_results['bias'].mean():.4f}")
        print(f"Mean RMSE: {n20_results['rmse'].mean():.4f}")
        print(f"Coverage rate: {n20_results['coverage'].mean()*100:.1f}%")
        print(f"Mean CI width: {n20_results['ci_width'].mean():.4f}")
        print(f"Mean % incomplete: {n20_results['pct_incomplete'].mean():.1f}%")
        print(f"Mean # complete auctions: {n20_results['n_complete'].mean():.1f}")

    print("\n" + "="*70)
    print("TASK A SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)


def task_b_sensitivity():
    """Run Task B sensitivity analysis."""
    configure_plotting()
    np.random.seed(2025)

    TASK_B_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    analysis = TaskBSensitivityAnalysis(n_replications=10)
    df = analysis.sensitivity_sample_size()

    # Save CSV
    csv_path = TASK_B_SENSITIVITY_DIR / "sample_size_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate plots
    plot_path = TASK_B_SENSITIVITY_DIR / "sample_size_sensitivity.png"
    analysis.plot_results(df, str(plot_path))

    # Save summary
    summary_path = TASK_B_SENSITIVITY_DIR / "summary_table.csv"
    summary = df.groupby('N').agg({
        'n_S_bounds': ['mean', 'std'],
        'n_F_bounds': ['mean', 'std'],
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
        'rhat_S': 'mean',
        'rhat_F': 'mean'
    }).round(4)
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")

    # Highlight N=20 results
    print("\n" + "="*70)
    print("N=20 STRESS TEST RESULTS (Critical for real data)")
    print("="*70)
    n20_results = df[df['N'] == 20]
    print(f"Type S - Mean bias: {n20_results['bias_S'].mean():.4f}, RMSE: {n20_results['rmse_S'].mean():.4f}")
    print(f"Type F - Mean bias: {n20_results['bias_F'].mean():.4f}, RMSE: {n20_results['rmse_F'].mean():.4f}")
    print(f"Gap - Mean bias: {n20_results['bias_gap'].mean():.4f}")
    print(f"Coverage: S={n20_results['coverage_S'].mean()*100:.1f}%, F={n20_results['coverage_F'].mean()*100:.1f}%")
    print(f"Pr(mu_S > mu_F): {n20_results['prob_S_greater_F'].mean()*100:.1f}%")

    print("\n" + "="*70)
    print("TASK B SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Default: run Task A baseline
    task_a_baseline()
