"""
Results analysis and metrics computation.

This module contains classes for analyzing MCMC results and computing
performance metrics for both Task A (single cutoff) and Task B (two-stage DGP).
"""

import numpy as np
from typing import Dict, List, Optional


def compute_collinearity_diagnostics(X: np.ndarray) -> Dict:
    """Compute simple collinearity diagnostics for cutoff covariates.

    Uses non-intercept columns only (if present) to avoid singularity from
    the constant intercept column.
    """
    if X.ndim != 2 or X.shape[0] < 2:
        return {
            "n_obs": int(X.shape[0]) if X.ndim == 2 else 0,
            "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
            "n_features_ex_intercept": 0,
            "max_abs_corr": np.nan,
            "condition_number": np.nan,
            "vifs": [],
            "max_vif": np.nan,
            "flag_high_corr": False,
            "flag_high_cond": False,
            "flag_high_vif": False,
        }

    n_obs, k = X.shape
    if k <= 1:
        return {
            "n_obs": int(n_obs),
            "n_features": int(k),
            "n_features_ex_intercept": 0,
            "max_abs_corr": np.nan,
            "condition_number": np.nan,
            "vifs": [],
            "max_vif": np.nan,
            "flag_high_corr": False,
            "flag_high_cond": False,
            "flag_high_vif": False,
        }

    Xn = X[:, 1:].astype(float)
    mean = Xn.mean(axis=0)
    std = Xn.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0
    Z = (Xn - mean) / std

    corr = np.corrcoef(Z, rowvar=False)
    if corr.size == 1:
        max_abs_corr = 0.0
    else:
        off_diag = corr - np.eye(corr.shape[0])
        max_abs_corr = float(np.max(np.abs(off_diag)))

    try:
        cond = float(np.linalg.cond(Z))
    except np.linalg.LinAlgError:
        cond = float("inf")

    vifs = []
    if Z.shape[1] >= 2:
        for j in range(Z.shape[1]):
            y = Z[:, j]
            Xo = np.delete(Z, j, axis=1)
            Xo = np.column_stack([np.ones(Xo.shape[0]), Xo])
            beta = np.linalg.lstsq(Xo, y, rcond=None)[0]
            y_hat = Xo @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
            if r2 >= 1.0:
                vifs.append(float("inf"))
            else:
                vifs.append(1.0 / (1.0 - r2))
    max_vif = float(np.max(vifs)) if vifs else np.nan

    flag_high_corr = bool(max_abs_corr >= 0.98)
    flag_high_cond = bool(cond >= 1e4)
    flag_high_vif = bool(max_vif >= 10.0) if np.isfinite(max_vif) else True

    return {
        "n_obs": int(n_obs),
        "n_features": int(k),
        "n_features_ex_intercept": int(k - 1),
        "max_abs_corr": max_abs_corr,
        "condition_number": cond,
        "vifs": vifs,
        "max_vif": max_vif,
        "flag_high_corr": flag_high_corr,
        "flag_high_cond": flag_high_cond,
        "flag_high_vif": flag_high_vif,
    }


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
    """Analyze MCMC results for Task B (two-stage DGP)."""

    def __init__(
        self,
        mcmc_results: Dict,
        *,
        true_gamma: float,
        true_tilde_alpha: float,
        true_beta_cutoff: Optional[np.ndarray] = None,
        true_cutoff_c: Optional[float] = None,
        true_sigma_omega: Optional[float] = None,
        true_sigma_nu: Optional[float] = None,
        true_sigma_eta: Optional[float] = None,
    ):
        self.results = mcmc_results
        self.true_gamma = float(true_gamma)
        self.true_tilde_alpha = float(true_tilde_alpha)
        if true_beta_cutoff is None:
            if true_cutoff_c is None:
                raise ValueError("Must provide true_beta_cutoff or true_cutoff_c")
            self.true_beta_cutoff = np.array([float(true_cutoff_c)], dtype=float)
        else:
            self.true_beta_cutoff = np.atleast_1d(true_beta_cutoff).astype(float)

        self.true_sigma_omega = None if true_sigma_omega is None else float(true_sigma_omega)
        self.true_sigma_nu = None if true_sigma_nu is None else float(true_sigma_nu)
        self.true_sigma_eta = None if true_sigma_eta is None else float(true_sigma_eta)

    @staticmethod
    def _beta_names(k: int) -> List[str]:
        if k == 1:
            return ["c"]
        if k == 3:
            return ["c", "depth_mean_23", "depth_gap_23"]
        if k == 4:
            return ["c", "theta1", "theta2", "theta3"]
        return [f"beta_{j}" for j in range(k)]

    def compute_metrics(self) -> Dict:
        gamma = self.results['gamma_samples']
        tilde_alpha = self.results['tilde_alpha_samples']
        beta_samples = self.results['beta_samples']
        cutoff_c = self.results.get('cutoff_c_samples', beta_samples[:, 0])
        sigma_omega = self.results.get('sigma_omega_samples')
        sigma_nu = self.results.get('sigma_nu_samples')
        sigma_eta = self.results.get('sigma_eta_samples')

        gamma_metrics = compute_parameter_metrics(gamma, self.true_gamma)
        alpha_metrics = compute_parameter_metrics(tilde_alpha, self.true_tilde_alpha)
        cutoff_metrics = compute_parameter_metrics(cutoff_c, float(self.true_beta_cutoff[0]))

        k = int(beta_samples.shape[1]) if beta_samples.ndim == 2 else 1
        beta_true = self.true_beta_cutoff
        if beta_true.size != k:
            raise ValueError(f"True beta_cutoff length {beta_true.size} does not match samples k={k}")

        cutoff_beta_metrics = {}
        for name, j in zip(self._beta_names(k), range(k)):
            cutoff_beta_metrics[name] = compute_parameter_metrics(beta_samples[:, j], float(beta_true[j]))

        sigma_metrics: Dict[str, Dict] = {}
        if sigma_omega is not None and self.true_sigma_omega is not None:
            sigma_metrics['sigma_omega'] = compute_parameter_metrics(sigma_omega, self.true_sigma_omega)
        if sigma_eta is not None and self.true_sigma_eta is not None:
            sigma_metrics['sigma_eta'] = compute_parameter_metrics(sigma_eta, self.true_sigma_eta)
        if sigma_nu is not None and self.true_sigma_nu is not None:
            sigma_metrics['sigma_nu'] = compute_parameter_metrics(sigma_nu, self.true_sigma_nu)

        return {
            'gamma': gamma_metrics,
            'tilde_alpha': alpha_metrics,
            'cutoff_c': cutoff_metrics,
            'cutoff_beta': cutoff_beta_metrics,
            'sigmas': sigma_metrics,
            'rhat_gamma': self.results['rhat_gamma'],
            'rhat_kappa': self.results['rhat_kappa'],
            'rhat_beta': self.results['rhat_beta'],
        }

    def print_summary(self, metrics: Dict):
        print("\n" + "="*60)
        print("TASK B: TWO-STAGE RESULTS (Selection-Aware)")
        print("="*60)

        g = metrics['gamma']
        a = metrics['tilde_alpha']
        c = metrics['cutoff_c']

        print("VALUATION MEAN (gamma):")
        print(f"  True:       {self.true_gamma:.4f}")
        print(f"  Estimate:   {g['estimate']:.4f}")
        print(f"  Bias:       {g['bias']:+.4f}")
        print(f"  95% CI:     [{g['ci_lower']:.4f}, {g['ci_upper']:.4f}]")
        print(f"  Coverage:   {g['coverage']}")

        print("\nMISREPORTING (tilde_alpha):")
        print(f"  True:       {self.true_tilde_alpha:.4f}")
        print(f"  Estimate:   {a['estimate']:.4f}")
        print(f"  Bias:       {a['bias']:+.4f}")
        print(f"  95% CI:     [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]")
        print(f"  Coverage:   {a['coverage']}")

        print("\nCUTOFF COEFFICIENTS (beta_cutoff):")
        c_true = float(self.true_beta_cutoff[0])
        print(f"  True c:     {c_true:.4f}")
        print(f"  Estimate:   {c['estimate']:.4f}")
        print(f"  Bias:       {c['bias']:+.4f}")
        print(f"  95% CI:     [{c['ci_lower']:.4f}, {c['ci_upper']:.4f}]")
        print(f"  Coverage:   {c['coverage']}")

        beta_metrics = metrics.get('cutoff_beta', {})
        beta_names = self._beta_names(int(self.true_beta_cutoff.size))
        for name, m in beta_metrics.items():
            if name == "c":
                continue
            idx = beta_names.index(name)
            print(
                f"  {name:>7s}: true={float(self.true_beta_cutoff[idx]):.4f}, "
                f"est={m['estimate']:.4f}, bias={m['bias']:+.4f}"
            )

        sigma_metrics = metrics.get('sigmas', {})
        if sigma_metrics:
            print("\nSIGMAS:")
            for name, m in sigma_metrics.items():
                if name == "sigma_omega":
                    true_value = self.true_sigma_omega
                elif name == "sigma_eta":
                    true_value = self.true_sigma_eta
                elif name == "sigma_nu":
                    true_value = self.true_sigma_nu
                else:
                    true_value = None
                print(
                    f"  {name}: true={float(true_value) if true_value is not None else np.nan:.4f}, "
                    f"est={m['estimate']:.4f}, bias={m['bias']:+.4f}, "
                    f"95% CI=[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]"
                )

        print("\nDiagnostics:")
        print(f"  R-hat beta:  {metrics['rhat_beta']:.4f}")
        print(f"  R-hat gamma: {metrics['rhat_gamma']:.4f}")
        print(f"  R-hat kappa: {metrics['rhat_kappa']:.4f}")
        col = self.results.get("collinearity_diagnostics")
        if col:
            print(
                f"  Collinearity: max|corr|={col.get('max_abs_corr')}, "
                f"cond={col.get('condition_number')}, max VIF={col.get('max_vif')}"
            )
        print("="*60 + "\n")
