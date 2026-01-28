from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .mcmc import MCMCConfig, TaskBMHSampler
from .sim import TaskBDGP, TaskBDataGenerator
from .specs import TASKB_SPECS


def _summarize_samples(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": np.nan, "sd": np.nan, "p2p5": np.nan, "p50": np.nan, "p97p5": np.nan}
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "p2p5": float(np.percentile(x, 2.5)),
        "p50": float(np.percentile(x, 50)),
        "p97p5": float(np.percentile(x, 97.5)),
    }


def run_compare(
    *,
    out_dir: str | Path,
    spec_names: List[str],
    N_values: List[int],
    n_rep: int = 3,
    seed: int = 123,
    dgp_template: Optional[TaskBDGP] = None,
    mcmc_config: Optional[MCMCConfig] = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dgp_template is None:
        dgp_template = TaskBDGP()
    if mcmc_config is None:
        mcmc_config = MCMCConfig(n_iterations=8000, burn_in=4000, thinning=10, n_chains=2, stage=1)

    rows: List[Dict] = []

    for spec_name in spec_names:
        if spec_name not in TASKB_SPECS:
            raise ValueError(f"Unknown spec '{spec_name}'")

        for N in N_values:
            for r in range(n_rep):
                np.random.seed(seed + 10000 * r + 37 * N)

                dgp_kwargs = asdict(dgp_template)
                dgp_kwargs["N_obs"] = int(N)
                dgp_kwargs["spec_name"] = str(spec_name)
                # Use per-spec default true betas / delta unless explicitly overridden
                dgp_kwargs["beta"] = None
                dgp_kwargs["delta"] = None
                dgp = TaskBDGP(**dgp_kwargs)

                gen = TaskBDataGenerator(dgp)
                auctions, summary = gen.generate()

                cfg = MCMCConfig(**asdict(mcmc_config))
                cfg.kappa_init = float(dgp.kappa)
                cfg.delta_init = float(dgp.delta)
                cfg.sigma_nu_fixed = float(dgp.sigma_nu)
                cfg.sigma_eta_fixed = float(dgp.sigma_eta)

                sampler = TaskBMHSampler(
                    auctions,
                    spec_name=spec_name,
                    config=cfg,
                    misreporting_mode=dgp.misreporting_mode,
                )
                results = sampler.run()

                beta = results["beta_samples"]
                gamma = results["gamma_samples"]
                kappa = results["kappa_samples"]
                delta = results["delta_samples"]
                sigma_omega = results["sigma_omega_samples"]

                row = {
                    "spec": spec_name,
                    "N_obs": N,
                    "rep_id": r,
                    "keep_rate_pct": summary.get("keep_rate_pct", np.nan),
                    "pct_one_sided": 100.0 * summary.get("n_one_sided", 0) / float(summary.get("n_observed", 1)),
                    "true_gamma": dgp.gamma,
                    "true_kappa": dgp.kappa,
                    "true_sigma_omega": dgp.sigma_omega,
                    "true_delta": dgp.delta,
                    "acc_bstar": results.get("acc_bstar", np.nan),
                    "acc_gamma": results.get("acc_gamma", np.nan),
                    "acc_kappa": results.get("acc_kappa", np.nan),
                    "acc_delta": results.get("acc_delta", np.nan),
                    "rhat_gamma": results.get("rhat_gamma", np.nan),
                    "rhat_kappa": results.get("rhat_kappa", np.nan),
                    "rhat_delta": results.get("rhat_delta", np.nan),
                    "rhat_beta_max": np.nan,
                }
                rhat_beta = np.asarray(results.get("rhat_beta", np.array([])), dtype=float)
                if rhat_beta.size and np.any(np.isfinite(rhat_beta)):
                    row["rhat_beta_max"] = float(np.nanmax(rhat_beta))

                row.update({f"post_gamma_{k}": v for k, v in _summarize_samples(gamma).items()})
                row.update({f"post_kappa_{k}": v for k, v in _summarize_samples(kappa).items()})
                row.update({f"post_delta_{k}": v for k, v in _summarize_samples(delta).items()})
                row.update({f"post_sigma_omega_{k}": v for k, v in _summarize_samples(sigma_omega).items()})

                beta_names = results.get("beta_names", [f"beta_{j}" for j in range(beta.shape[1])])
                for j, name in enumerate(beta_names):
                    summ = _summarize_samples(beta[:, j])
                    for k, v in summ.items():
                        row[f"post_{name}_{k}"] = v
                    row[f"true_{name}"] = float(dgp.beta[j]) if dgp.beta is not None and j < len(dgp.beta) else np.nan

                col = results.get("collinearity_diagnostics", {})
                row["max_abs_corr"] = col.get("max_abs_corr", np.nan)
                row["cond_X"] = col.get("condition_number", np.nan)
                row["max_vif"] = col.get("max_vif", np.nan)

                if beta.shape[1] >= 3 and kappa.size:
                    theta_type = beta[:, 1]
                    theta_spr = beta[:, 2]
                    row["corr_theta_spr_kappa"] = float(np.corrcoef(theta_spr, kappa)[0, 1])
                    row["corr_theta_type_theta_spr"] = float(np.corrcoef(theta_type, theta_spr)[0, 1])
                else:
                    row["corr_theta_spr_kappa"] = np.nan
                    row["corr_theta_type_theta_spr"] = np.nan

                if spec_name == "cand3_type_shift_admission" and delta.size and beta.shape[1] >= 2:
                    row["corr_delta_theta_type"] = float(np.corrcoef(delta, beta[:, 1])[0, 1])
                    row["corr_delta_c"] = float(np.corrcoef(delta, beta[:, 0])[0, 1])
                else:
                    row["corr_delta_theta_type"] = np.nan
                    row["corr_delta_c"] = np.nan

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "compare.csv", index=False)
    return df
