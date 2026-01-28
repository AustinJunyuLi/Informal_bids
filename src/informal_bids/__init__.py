"""
Informal Bids — clean-slate refactor branch.

This branch rebuilds the Task B (two-stage) simulation + estimation codebase
around a spec-driven cutoff/admission interface.

Public entrypoints:
- `informal_bids.cli:task_b_run`
- `informal_bids.cli:task_b_compare`
"""

from .compare import run_compare
from .mcmc import MCMCConfig, TaskBMHSampler
from .sim import TaskBDGP, TaskBDataGenerator
from .specs import TASKB_SPECS

__version__ = "0.3.0"

__all__ = [
    "TaskBDGP",
    "TaskBDataGenerator",
    "MCMCConfig",
    "TaskBMHSampler",
    "TASKB_SPECS",
    "run_compare",
]

