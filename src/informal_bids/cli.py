"""New minimal CLI entrypoints for the clean-slate Task B refactor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .compare import run_compare
from .mcmc import MCMCConfig, TaskBMHSampler
from .sim import TaskBDGP, TaskBDataGenerator
from .specs import TASKB_SPECS


def task_b_run() -> None:
    parser = argparse.ArgumentParser(description="Run one Task B simulation + MCMC fit (clean-slate refactor).")
    parser.add_argument("--spec", default="cand1_type_spr", choices=sorted(TASKB_SPECS.keys()))
    parser.add_argument("--N", type=int, default=100, help="Number of observed auctions (reach formal stage).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="outputs/run_v1")
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--burn", type=int, default=4000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))

    dgp = TaskBDGP(N_obs=int(args.N), spec_name=str(args.spec))
    auctions, summary = TaskBDataGenerator(dgp).generate()

    cfg = MCMCConfig(
        n_iterations=int(args.iters),
        burn_in=int(args.burn),
        thinning=int(args.thin),
        n_chains=int(args.chains),
        stage=int(args.stage),
        kappa_init=float(dgp.kappa),
        log_every=int(args.log_every),
    )
    cfg.delta_init = float(dgp.delta)
    cfg.sigma_nu_fixed = float(dgp.sigma_nu)
    cfg.sigma_eta_fixed = float(dgp.sigma_eta)
    sampler = TaskBMHSampler(auctions, spec_name=str(args.spec), config=cfg, misreporting_mode=dgp.misreporting_mode)
    results = sampler.run()

    (out_dir / "summary.txt").write_text(
        f"spec={args.spec}\\nN_obs={args.N}\\nkeep_rate_pct={summary.get('keep_rate_pct')}\\n"
        f"rhat_gamma={results.get('rhat_gamma')}\\nrhat_kappa={results.get('rhat_kappa')}\\n",
        encoding="utf-8",
    )


def task_b_compare() -> None:
    parser = argparse.ArgumentParser(description="Run Task B comparison suite (clean-slate refactor).")
    parser.add_argument("--out", type=str, default="outputs/compare_v1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--N", type=str, default="20,50,100,200", help="Comma-separated N values (observed auctions).")
    parser.add_argument(
        "--specs",
        type=str,
        default="legacy_moments_k4,legacy_depth_k2,cand1_type_spr,cand2_type_spr_depth_z,cand3_type_shift_admission,cand4_type_spr_prec_z",
        help="Comma-separated spec names.",
    )
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--burn", type=int, default=4000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    N_values = [int(x.strip()) for x in args.N.split(",") if x.strip()]
    spec_names = [x.strip() for x in args.specs.split(",") if x.strip()]

    run_compare(
        out_dir=args.out,
        spec_names=spec_names,
        N_values=N_values,
        n_rep=int(args.rep),
        seed=int(args.seed),
        dgp_template=TaskBDGP(),
        mcmc_config=MCMCConfig(
            n_iterations=int(args.iters),
            burn_in=int(args.burn),
            thinning=int(args.thin),
            n_chains=int(args.chains),
            stage=int(args.stage),
            log_every=int(args.log_every),
        ),
    )
