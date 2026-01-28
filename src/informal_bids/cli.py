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
from .utils import calibrate_cutoff_intercept_for_target_mean, cutoff_feature_names


def _write_collinearity_report(col_diag: dict, save_path: str) -> None:
    """Write a short collinearity diagnostic report to disk."""
    if not col_diag:
        return
    lines = [
        "Cutoff covariate collinearity diagnostics",
        "----------------------------------------",
        f"n_obs: {col_diag.get('n_obs')}",
        f"n_features: {col_diag.get('n_features')}",
        f"n_features_ex_intercept: {col_diag.get('n_features_ex_intercept')}",
        f"max_abs_corr: {col_diag.get('max_abs_corr')}",
        f"condition_number: {col_diag.get('condition_number')}",
        f"vifs: {col_diag.get('vifs')}",
        f"max_vif: {col_diag.get('max_vif')}",
        f"flag_high_corr: {col_diag.get('flag_high_corr')}",
        f"flag_high_cond: {col_diag.get('flag_high_cond')}",
        f"flag_high_vif: {col_diag.get('flag_high_vif')}",
        "",
        "Rules of thumb:",
        "- max_abs_corr >= 0.98 and/or condition_number >= 1e4 indicate severe collinearity.",
        "- max_vif >= 10 indicates weak identification from multicollinearity.",
        "- In that case, identification of cutoff coefficients is weak or not achieved.",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    print("Two-Stage DGP (Selection-Aware)")
    print("="*70)

    baseline_dir = TASK_B_BASELINE_DIR / "moments"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Moments cutoff (meeting notes Eq. 8):
    #   b*_i = c + theta1*m1 + theta2*m2 + theta3*m3 + omega_i
    theta = np.array([0.25, 0.10, 0.05], dtype=float)

    dgp_J = 3
    dgp_gamma = 1.3
    dgp_sigma_nu = 0.2
    dgp_sigma_eta = 0.1
    dgp_sigma_omega = 0.1
    dgp_kappa = float(np.log(1.5))  # lambda_I(J=3,kappa) ≈ 1

    # Calibrate intercept so the unconditional mean cutoff is ~1.4.
    c = calibrate_cutoff_intercept_for_target_mean(
        target_mean_cutoff=1.4,
        theta=theta,
        n_bidders=dgp_J,
        gamma=dgp_gamma,
        sigma_nu=dgp_sigma_nu,
        kappa=dgp_kappa,
        cutoff_spec="moments_k4",
        n_sim=30000,
        seed=123,
    )
    beta_cutoff = np.concatenate(([c], theta))

    dgp_params = TaskBDGPParameters(
        N=100,
        J=dgp_J,
        gamma=dgp_gamma,
        sigma_nu=dgp_sigma_nu,
        sigma_eta=dgp_sigma_eta,
        kappa=dgp_kappa,
        cutoff_spec="moments_k4",
        beta_cutoff=beta_cutoff,
        sigma_omega=dgp_sigma_omega,
    )

    print(f"\n{dgp_params}")

    generator = TaskBDataGenerator(dgp_params)
    auctions, summary = generator.generate_auction_data()

    print(f"\nGenerated {summary['n_observed']} observed auctions (formal stage):")
    print(f"  Two-sided:  {summary['n_complete']} ({100*summary['n_complete']/summary['n_observed']:.1f}%)")
    print(f"  One-sided:  {summary['n_incomplete']} ({summary['pct_incomplete']:.1f}%)")
    print(f"  Initiated:  {summary['n_initiated']} (dropped all-reject: {summary['n_dropped_all_reject']}, keep rate: {summary['keep_rate_pct']:.1f}%)")

    mcmc_config = MCMCConfig(
        n_iterations=15000,
        burn_in=7500,
        thinning=10,
        n_chains=2,
        task_b_stage=1,
        task_b_sigma_nu_fixed=dgp_params.sigma_nu,
        task_b_sigma_eta_fixed=dgp_params.sigma_eta,
        task_b_kappa_init=dgp_params.kappa,
    )
    sampler = TaskBMCMCSampler(auctions, mcmc_config)
    results = sampler.run()

    analyzer = TaskBResultsAnalyzer(
        results,
        true_gamma=dgp_params.gamma,
        true_tilde_alpha=dgp_params.tilde_alpha,
        true_beta_cutoff=dgp_params.beta_cutoff,
        true_sigma_omega=dgp_params.sigma_omega,
        true_sigma_nu=dgp_params.sigma_nu,
        true_sigma_eta=dgp_params.sigma_eta,
    )
    metrics = analyzer.compute_metrics()
    analyzer.print_summary(metrics)

    col_diag = results.get("collinearity_diagnostics", {})
    if col_diag:
        _write_collinearity_report(col_diag, str(baseline_dir / "collinearity_report.txt"))

    print("\nGenerating visualization plots...")
    TaskBVisualizer.plot_diagnostics(
        results,
        true_gamma=dgp_params.gamma,
        true_tilde_alpha=dgp_params.tilde_alpha,
        true_beta_cutoff=dgp_params.beta_cutoff,
        true_sigma_omega=dgp_params.sigma_omega,
        true_sigma_nu=dgp_params.sigma_nu,
        true_sigma_eta=dgp_params.sigma_eta,
        save_path=str(baseline_dir / "task_b_diagnostics.png"),
    )

    TaskBVisualizer.plot_informal_vs_formal(
        auctions,
        kappa_est=float(np.mean(results['kappa_samples'])),
        misreporting_mode=str(results.get('misreporting_mode', 'scale')),
        save_path=str(baseline_dir / "task_b_scatter.png"),
    )

    print("\nTask B baseline complete!")
    return metrics, results, auctions


def task_b_depth_baseline():
    """Run Task B baseline simulation with depth-based cutoff moments."""
    configure_plotting()
    np.random.seed(124)

    print("\n" + "="*70)
    print("TASK B: DEPTH-BASED BASELINE SIMULATION")
    print("Two-Stage DGP (Selection-Aware)")
    print("="*70)

    baseline_dir = TASK_B_BASELINE_DIR / "depth"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Depth-based cutoff: [1, (b2+b3)/2, (b2-b3)]
    theta = np.array([0.10, 0.10], dtype=float)

    dgp_J = 3
    dgp_gamma = 1.3
    dgp_sigma_nu = 0.2
    dgp_sigma_eta = 0.1
    dgp_sigma_omega = 0.1
    dgp_kappa = float(np.log(1.5))

    c = calibrate_cutoff_intercept_for_target_mean(
        target_mean_cutoff=1.4,
        theta=theta,
        n_bidders=dgp_J,
        gamma=dgp_gamma,
        sigma_nu=dgp_sigma_nu,
        kappa=dgp_kappa,
        cutoff_spec="depth_k2",
        n_sim=30000,
        seed=124,
    )
    beta_cutoff = np.concatenate(([c], theta))

    dgp_params = TaskBDGPParameters(
        N=100,
        J=dgp_J,
        gamma=dgp_gamma,
        sigma_nu=dgp_sigma_nu,
        sigma_eta=dgp_sigma_eta,
        kappa=dgp_kappa,
        cutoff_spec="depth_k2",
        beta_cutoff=beta_cutoff,
        sigma_omega=dgp_sigma_omega,
    )

    print(f"\n{dgp_params}")
    print(f"Cutoff features: {cutoff_feature_names(dgp_params.cutoff_spec)}")

    generator = TaskBDataGenerator(dgp_params)
    auctions, summary = generator.generate_auction_data()

    print(f"\nGenerated {summary['n_observed']} observed auctions (formal stage):")
    print(f"  Two-sided:  {summary['n_complete']} ({100*summary['n_complete']/summary['n_observed']:.1f}%)")
    print(f"  One-sided:  {summary['n_incomplete']} ({summary['pct_incomplete']:.1f}%)")
    print(f"  Initiated:  {summary['n_initiated']} (dropped all-reject: {summary['n_dropped_all_reject']}, keep rate: {summary['keep_rate_pct']:.1f}%)")

    mcmc_config = MCMCConfig(
        n_iterations=15000,
        burn_in=7500,
        thinning=10,
        n_chains=2,
        task_b_stage=1,
        task_b_sigma_nu_fixed=dgp_params.sigma_nu,
        task_b_sigma_eta_fixed=dgp_params.sigma_eta,
        task_b_kappa_init=dgp_params.kappa,
    )
    sampler = TaskBMCMCSampler(auctions, mcmc_config)
    results = sampler.run()

    analyzer = TaskBResultsAnalyzer(
        results,
        true_gamma=dgp_params.gamma,
        true_tilde_alpha=dgp_params.tilde_alpha,
        true_beta_cutoff=dgp_params.beta_cutoff,
        true_sigma_omega=dgp_params.sigma_omega,
        true_sigma_nu=dgp_params.sigma_nu,
        true_sigma_eta=dgp_params.sigma_eta,
    )
    metrics = analyzer.compute_metrics()
    analyzer.print_summary(metrics)

    col_diag = results.get("collinearity_diagnostics", {})
    if col_diag:
        _write_collinearity_report(col_diag, str(baseline_dir / "collinearity_report.txt"))

    print("\nGenerating visualization plots...")
    TaskBVisualizer.plot_diagnostics(
        results,
        true_gamma=dgp_params.gamma,
        true_tilde_alpha=dgp_params.tilde_alpha,
        true_beta_cutoff=dgp_params.beta_cutoff,
        true_sigma_omega=dgp_params.sigma_omega,
        true_sigma_nu=dgp_params.sigma_nu,
        true_sigma_eta=dgp_params.sigma_eta,
        save_path=str(baseline_dir / "task_b_diagnostics.png"),
    )

    TaskBVisualizer.plot_informal_vs_formal(
        auctions,
        kappa_est=float(np.mean(results['kappa_samples'])),
        misreporting_mode=str(results.get('misreporting_mode', 'scale')),
        save_path=str(baseline_dir / "task_b_scatter.png"),
    )

    print("\nTask B depth-based baseline complete!")
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

    # Match the Task B baseline moments cutoff.
    theta = np.array([0.25, 0.10, 0.05], dtype=float)
    J = 3
    gamma = 1.3
    sigma_nu = 0.2
    sigma_eta = 0.1
    sigma_omega = 0.1
    kappa = float(np.log(1.5))
    c = calibrate_cutoff_intercept_for_target_mean(
        target_mean_cutoff=1.4,
        theta=theta,
        n_bidders=J,
        gamma=gamma,
        sigma_nu=sigma_nu,
        kappa=kappa,
        cutoff_spec="moments_k4",
        n_sim=30000,
        seed=2025,
    )
    beta_cutoff = np.concatenate(([c], theta))

    for stage in (1, 2, 3):
        df = analysis.sensitivity_sample_size(
            stage=stage,
            J=J,
            gamma=gamma,
            sigma_nu=sigma_nu,
            sigma_eta=sigma_eta,
            kappa=kappa,
            beta_cutoff=beta_cutoff,
            cutoff_spec="moments_k4",
            sigma_omega=sigma_omega,
        )

        tag = f"stage{stage}_moments"

        # Save CSV
        csv_path = TASK_B_SENSITIVITY_DIR / f"sample_size_sensitivity_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Generate plots
        plot_path = TASK_B_SENSITIVITY_DIR / f"sample_size_sensitivity_{tag}.png"
        analysis.plot_results(df, str(plot_path))

        # Save summary (wide table)
        summary_path = TASK_B_SENSITIVITY_DIR / f"summary_table_{tag}.csv"
        summary = df.groupby('N').agg({
            'keep_rate_pct': ['mean', 'std'],
            'pct_incomplete': ['mean', 'std'],
            'bias_gamma': ['mean', 'std'],
            'bias_tilde_alpha': ['mean', 'std'],
            'bias_cutoff_c': ['mean', 'std'],
            'bias_theta1': ['mean', 'std'],
            'bias_theta2': ['mean', 'std'],
            'bias_theta3': ['mean', 'std'],
            'bias_sigma_omega': ['mean', 'std'],
            'bias_sigma_eta': ['mean', 'std'],
            'bias_sigma_nu': ['mean', 'std'],
            'rmse_gamma': ['mean', 'std'],
            'rmse_tilde_alpha': ['mean', 'std'],
            'rmse_cutoff_c': ['mean', 'std'],
            'rmse_theta1': ['mean', 'std'],
            'rmse_theta2': ['mean', 'std'],
            'rmse_theta3': ['mean', 'std'],
            'rmse_sigma_omega': ['mean', 'std'],
            'rmse_sigma_eta': ['mean', 'std'],
            'rmse_sigma_nu': ['mean', 'std'],
            'ci_width_gamma': ['mean', 'std'],
            'ci_width_tilde_alpha': ['mean', 'std'],
            'ci_width_cutoff_c': ['mean', 'std'],
            'ci_width_theta1': ['mean', 'std'],
            'ci_width_theta2': ['mean', 'std'],
            'ci_width_theta3': ['mean', 'std'],
            'ci_width_sigma_omega': ['mean', 'std'],
            'ci_width_sigma_eta': ['mean', 'std'],
            'ci_width_sigma_nu': ['mean', 'std'],
            'coverage_gamma': 'mean',
            'coverage_tilde_alpha': 'mean',
            'coverage_cutoff_c': 'mean',
            'coverage_theta1': 'mean',
            'coverage_theta2': 'mean',
            'coverage_theta3': 'mean',
            'coverage_sigma_omega': 'mean',
            'coverage_sigma_eta': 'mean',
            'coverage_sigma_nu': 'mean',
            'rhat_gamma': 'mean',
            'rhat_kappa': 'mean',
            'rhat_beta': 'mean',
        }).round(4)
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")

        # Highlight N=20 results
        print("\n" + "="*70)
        print(f"N=20 STRESS TEST RESULTS ({tag})")
        print("="*70)
        n20_results = df[df['N'] == 20]
        print(f"Conversion (keep rate) mean: {n20_results['keep_rate_pct'].mean():.1f}%")
        print(f"Gamma - mean bias: {n20_results['bias_gamma'].mean():+.4f}, RMSE: {n20_results['rmse_gamma'].mean():.4f}")
        print(f"Alpha - mean bias: {n20_results['bias_tilde_alpha'].mean():+.4f}, RMSE: {n20_results['rmse_tilde_alpha'].mean():.4f}")
        print(f"c     - mean bias: {n20_results['bias_cutoff_c'].mean():+.4f}, RMSE: {n20_results['rmse_cutoff_c'].mean():.4f}")
        if n20_results['bias_theta1'].notna().any():
            print(
                f"theta1/theta2/theta3 - mean bias: "
                f"{n20_results['bias_theta1'].mean():+.4f}, "
                f"{n20_results['bias_theta2'].mean():+.4f}, "
                f"{n20_results['bias_theta3'].mean():+.4f}"
            )

    print("\n" + "="*70)
    print("TASK B SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)


def task_b_depth_sensitivity():
    """Run Task B sensitivity analysis for the depth-based cutoff."""
    configure_plotting()
    np.random.seed(2026)

    outdir = TASK_B_SENSITIVITY_DIR / "depth"
    outdir.mkdir(parents=True, exist_ok=True)

    analysis = TaskBSensitivityAnalysis(n_replications=10)

    theta = np.array([0.10, 0.10], dtype=float)
    J = 3
    gamma = 1.3
    sigma_nu = 0.2
    sigma_eta = 0.1
    sigma_omega = 0.1
    kappa = float(np.log(1.5))

    c = calibrate_cutoff_intercept_for_target_mean(
        target_mean_cutoff=1.4,
        theta=theta,
        n_bidders=J,
        gamma=gamma,
        sigma_nu=sigma_nu,
        kappa=kappa,
        cutoff_spec="depth_k2",
        n_sim=30000,
        seed=2026,
    )
    beta_cutoff = np.concatenate(([c], theta))

    for stage in (1, 2, 3):
        df = analysis.sensitivity_sample_size(
            stage=stage,
            J=J,
            gamma=gamma,
            sigma_nu=sigma_nu,
            sigma_eta=sigma_eta,
            kappa=kappa,
            beta_cutoff=beta_cutoff,
            cutoff_spec="depth_k2",
            sigma_omega=sigma_omega,
        )

        tag = f"stage{stage}_depth"
        csv_path = outdir / f"sample_size_sensitivity_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        plot_path = outdir / f"sample_size_sensitivity_{tag}.png"
        analysis.plot_results(df, str(plot_path))





def task_b_intercept_baseline():
    """Reproduce the analysis-report Task B intercept-only baseline (Stages 1-3).

    This entrypoint is meant to align with analysis_report.pdf §4.1. It generates
    diagnostics for each stage under an intercept-only cutoff.

    It does not run automatically; invoke via the console script
    `task-b-intercept-baseline` after installation.
    """
    configure_plotting()
    np.random.seed(42)

    N = 100
    J = 3
    gamma = 1.3
    sigma_nu = 0.2
    sigma_eta = 0.1
    kappa = float(np.log(1.5))  # makes lambda_I=(1-1/J)*exp(kappa)=1 when mode='scale'

    dgp_params = TaskBDGPParameters(
        N=N,
        J=J,
        gamma=gamma,
        sigma_nu=sigma_nu,
        sigma_eta=sigma_eta,
        kappa=kappa,
        misreporting_mode="scale",
        beta_cutoff=np.array([1.4]),
        sigma_omega=0.05,
    )

    auctions, stats = TaskBDataGenerator(dgp_params).generate_auction_data()
    print("Task B intercept-only baseline:")
    print(f"Generated {stats['n_observed']} observed auctions (formal stage):")
    print(f"  Two-sided:  {stats['n_complete']} ({100*stats['n_complete']/stats['n_observed']:.1f}%)")
    print(f"  One-sided:  {stats['n_incomplete']} ({stats['pct_incomplete']:.1f}%)")
    print(
        f"  Initiated:  {stats['n_initiated']} (dropped all-reject: "
        f"{stats['n_dropped_all_reject']}, keep rate: {stats['keep_rate_pct']:.1f}%)"
    )

    base_dir = TASK_B_BASELINE_DIR / "intercept"
    for stage in (1, 2, 3):
        outdir = base_dir / f"stage{stage}"
        outdir.mkdir(parents=True, exist_ok=True)

        mcmc_config = MCMCConfig(
            n_iterations=15000,
            burn_in=7500,
            thinning=5,
            n_chains=3,
            task_b_stage=stage,
            task_b_misreporting_mode="scale",
            task_b_sigma_nu_fixed=sigma_nu,
            task_b_sigma_eta_fixed=sigma_eta,
            task_b_kappa_init=kappa,
            task_b_kappa_prior_mean=0.0,
            task_b_kappa_prior_std=1.0,
            task_b_sigma_omega_init=0.05,
            task_b_sigma_omega_prior_a=2.0,
            task_b_sigma_omega_prior_b=0.05,
            task_b_gamma_prior_mean=gamma,
            task_b_gamma_prior_std=0.5,
            task_b_gamma_proposal_sd=0.05,
            task_b_kappa_proposal_sd=0.05,
        )

        sampler = TaskBMCMCSampler(auctions, mcmc_config)
        results = sampler.run()

        TaskBVisualizer.plot_diagnostics(
            results,
            true_gamma=gamma,
            true_tilde_alpha=dgp_params.tilde_alpha,
            true_cutoff_c=1.4,
            true_sigma_omega=dgp_params.sigma_omega,
            true_sigma_nu=dgp_params.sigma_nu,
            true_sigma_eta=dgp_params.sigma_eta,
            save_path=str(outdir / "task_b_diagnostics.png"),
        )
        TaskBVisualizer.plot_informal_vs_formal(
            auctions,
            kappa_est=float(np.mean(results['kappa_samples'])),
            misreporting_mode=str(results.get('misreporting_mode', 'scale')),
            save_path=str(outdir / "task_b_scatter.png"),
        )

        analyzer = TaskBResultsAnalyzer(
            results,
            true_gamma=gamma,
            true_tilde_alpha=dgp_params.tilde_alpha,
            true_cutoff_c=1.4,
            true_sigma_omega=dgp_params.sigma_omega,
            true_sigma_nu=dgp_params.sigma_nu,
            true_sigma_eta=dgp_params.sigma_eta,
        )
        metrics = analyzer.compute_metrics()
        analyzer.print_summary(metrics)



def task_b_intercept_sensitivity():
    """Reproduce the analysis-report Task B intercept-only Stage 1 sample-size sensitivity.

    The analysis_report.pdf uses 2 replications per N for this diagnostic.
    """
    configure_plotting()

    analysis = TaskBSensitivityAnalysis(n_replications=2)
    df = analysis.sensitivity_sample_size(
        N_values=[20, 50, 100, 200],
        stage=1,
        beta_cutoff=np.array([1.4]),
        J=3,
        gamma=1.3,
        sigma_nu=0.2,
        sigma_eta=0.1,
        kappa=float(np.log(1.5)),
        misreporting_mode="scale",
        cutoff_c=1.4,
    )

    outdir = TASK_B_SENSITIVITY_DIR / "14-01-26"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "sample_size_sensitivity_stage1_intercept.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved sensitivity results to {csv_path}")

    plot_path = outdir / "sample_size_sensitivity_stage1_intercept.png"
    analysis.plot_results(df, save_path=str(plot_path))
if __name__ == "__main__":
    # Default: run Task A baseline
    task_a_baseline()
