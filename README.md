# Informal Bids

Simulation and MCMC estimation for informal bid admission cutoffs.
This repository currently runs **simulation-only** experiments (no real data yet).

## Quick start

1) Create/activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Install the package in editable mode (recommended for development).

```bash
pip install -e .
```

3) Run baselines

```bash
task-a-baseline
task-b-baseline
```

4) Run sensitivity analyses

```bash
task-a-sensitivity
task-b-sensitivity
```

If you do not want to install the package, you can also run via module path:

```bash
PYTHONPATH=src python -m informal_bids.cli
```

Outputs are written to `outputs/` by default.

## Structure

```
src/informal_bids/              # Core package code (Python)
docs/                           # Meeting notes + shared model documentation
external/                       # Reference papers / Matlab code / packages
contrib/                        # Personal scratch space (non-canonical)
data/                           # (Future) raw/processed data
outputs/                        # Generated plots/results (ignored)
reports/                        # LaTeX reports / PDFs
```

## Reproducibility

- Random seeds are set inside each task module.
- Numba is enabled by default; the first run compiles kernels and may be slower.

## Notes

- Current sampling is conditional on reaching the formal stage (simulation).
- Incomplete-auction handling is under discussion with supervisor.
