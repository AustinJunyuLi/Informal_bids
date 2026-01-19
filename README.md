# Informal Bids

Simulation and MCMC estimation for informal bid admission cutoffs in M&A style auctions.
This repository currently runs simulation-only experiments (no real data yet).

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

3) Run baselines.

```bash
task-a-baseline
task-b-baseline
```

4) Run sensitivity analyses.

```bash
task-a-sensitivity
task-b-sensitivity
```

If you do not want to install the package, you can also run via module path:

```bash
PYTHONPATH=src python -m informal_bids.cli
```

Outputs are written to `outputs/` by default.

## What the code does

- Task A: estimate a single constant admission cutoff b* from admit/reject data.
- Task B: estimate type-specific cutoffs b*_S and b*_F for two bidder types.

Observed-sample definition (current simulation setup): N denotes auctions that reach the
formal stage (at least one admitted). All-reject auctions are generated but treated as
unobserved and excluded. All-admit auctions are retained via one-sided upper bounds.

## Model and estimation (Task A)

Admission rule:

- Bidder j is admitted in auction i iff b^I_ij >= b*_i.

Interval restrictions from observed admits/rejects:

- L_i = max rejected bid
- U_i = min admitted bid
- b*_i in [L_i, U_i] for two-sided cases
- If all admitted: only an upper bound (L_i = -inf, U_i = min admitted)
- If all rejected: only a lower bound (U_i = +inf) and the auction is unobserved

Parametric cutoff model:

- b*_i = X_i' beta + nu_i
- nu_i ~ N(0, sigma^2)
- nu_i in [L_i - X_i' beta, U_i - X_i' beta]

Gibbs sampling with data augmentation (current baseline):

1) Sample nu_i for each auction from a truncated normal with bounds implied by L_i/U_i
2) Update beta using a conjugate normal posterior
3) Update sigma^2 using a conjugate inverse-gamma posterior

Annotated core loop (conceptual):

```python
for t in range(n_iterations):
    # Step 1: sample latent nu_i subject to interval bounds
    for i in range(N):
        xb = X[i] @ beta
        lower = L[i] - xb
        upper = U[i] - xb
        nu[i] = sample_truncated_normal(0.0, sigma, lower, upper)
        b_star[i] = xb + nu[i]

    # Step 2: beta | b_star ~ Normal (conjugate update)
    V_post = inv(V0_inv + (X.T @ X) / (sigma ** 2))
    beta_post = V_post @ (V0_inv @ beta_prior_mean + (X.T @ b_star) / (sigma ** 2))
    beta = mvn(beta_post, V_post)

    # Step 3: sigma^2 | nu ~ InvGamma (conjugate update)
    a_post = a_prior + N / 2
    b_post = b_prior + 0.5 * sum(nu ** 2)
    sigma = sqrt(invgamma(a_post, b_post))
```

MATLAB pseudocode equivalent:

```matlab
for t = 1:n_iterations
    % Step 1: sample nu_i from truncated normal
    for i = 1:N
        xb = X(i,:) * beta;
        lower = L(i) - xb;
        upper = U(i) - xb;
        nu(i) = sample_truncnorm(0, sigma, lower, upper);
        b_star(i) = xb + nu(i);
    end

    % Step 2: update beta (normal-normal)
    V_post = inv(V0_inv + (X'*X) / sigma^2);
    mu_post = V_post * (V0_inv * mu_prior + (X'*b_star) / sigma^2);
    beta = mvnrnd(mu_post, V_post);

    % Step 3: update sigma^2 (inverse-gamma)
    a_post = a_prior + N/2;
    b_post = b_prior + sum(nu.^2)/2;
    sigma_sq = 1 / gamrnd(a_post, 1/b_post);
    sigma = sqrt(sigma_sq);
end
```

## Task B: type-specific cutoffs

Task B uses two bidder types (S and F) and separate cutoffs for each type. The sampler
runs independent Gibbs updates for (mu_S, sigma_S) and (mu_F, sigma_F) using the type-specific
bounds (L_S, U_S) and (L_F, U_F). Auctions with zero admitted overall are dropped; auctions
with one-sided bounds are retained for the relevant type.

## Code architecture

### Module overview

| Module | Purpose | MATLAB equivalent | Key deps |
| --- | --- | --- | --- |
| `cli.py` | Entry points for baselines and sensitivity | `main.m` | config, data, samplers, analysis, visualization, sensitivity |
| `data.py` | DGP params, auction data classes, generators | struct definitions | utils |
| `data_io.py` | Optional CSV loader for real data | data loader scripts | data |
| `samplers.py` | Gibbs samplers for Task A and B | main estimation script | data, utils, numba_kernels |
| `analysis.py` | Bias/RMSE/CI/coverage metrics | post-processing | numpy |
| `visualization.py` | Diagnostic and interval plots | plotting scripts | matplotlib |
| `sensitivity.py` | Sensitivity sweeps (sample size, cutoffs) | batch runner | data, samplers |
| `config.py` | Output paths and plot styling | config file | matplotlib, seaborn |
| `utils.py` | Truncated normal, R-hat, covariates | helper functions | scipy |
| `numba_kernels.py` | JIT-compiled fast loops | MEX/Coder | numba |

## Python and MATLAB cribsheet

### Syntax mapping

| MATLAB | Python |
| --- | --- |
| `% comment` | `# comment` |
| `function y = foo(x)` | `def foo(x): return y` |
| `end` | indentation (no end keyword) |
| `if cond ... end` | `if cond: ...` |
| `for i = 1:N ... end` | `for i in range(N): ...` |
| `a(1)` | `a[0]` |
| `a(1:3)` | `a[0:3]` or `a[:3]` |
| `a(end)` | `a[-1]` |
| `[a; b]` | `np.vstack([a, b])` |
| `[a, b]` | `np.hstack([a, b])` |
| `zeros(N, M)` | `np.zeros((N, M))` |
| `rand(N, 1)` | `np.random.rand(N)` |
| `randn(N, 1)` | `np.random.randn(N)` |
| `A * B` (matrix) | `A @ B` |
| `A .* B` (elemwise) | `A * B` |

### Key libraries

| Python library | MATLAB equivalent | Used for |
| --- | --- | --- |
| `numpy` | built-in arrays | arrays, linear algebra |
| `scipy.stats` | Statistics Toolbox | distributions (invgamma, truncnorm) |
| `matplotlib` | plotting | figures, subplots |
| `pandas` | tables | data frames, CSV I/O |
| `numba` | MEX/Coder | JIT compilation |

### Dataclasses (Python) vs structs (MATLAB)

```python
from dataclasses import dataclass

@dataclass
class MCMCConfig:
    n_iterations: int = 20000
    burn_in: int = 10000
    n_chains: int = 3
```

```matlab
config = struct();
config.n_iterations = 20000;
config.burn_in = 10000;
config.n_chains = 3;
```

## Data generation details (Task A)

DGP parameters (intercept-only baseline):

- N: number of observed (formal-stage) auctions
- J: bidders per auction
- mu_v: mean valuation
- sigma_v: valuation std dev
- b_star: true admission cutoff

Auction data structure (per auction):

- bids: all informal bids
- admitted: boolean mask
- L_i: max rejected bid (or -inf if all admitted)
- U_i: min admitted bid (or +inf if all rejected)
- is_complete: True if both bounds are finite

Generator logic (TaskADataGenerator):

1) Draw valuations v_ij = mu_v + epsilon_ij, epsilon ~ N(0, sigma_v^2)
2) Set bids = valuations (truthful informal bids)
3) Draw covariates X_i and cutoff b*_i = X_i' beta + optional shock
4) Admit if bid >= b*_i
5) Drop all-reject auctions (unobserved)
6) Compute L_i/U_i bounds for retained auctions

## Analysis and plots

Metrics computed for Task A and B:

- Posterior mean (mu_hat)
- Bias, RMSE
- 95% credible interval and coverage
- R-hat (Gelman-Rubin) for convergence checks

Visualization outputs:

- Trace plots (chain evolution)
- Posterior histograms
- Interval plots of [L_i, U_i] across auctions

## Reporting and figure management

All reports should cite figures directly from the centralized `outputs/` tree.
Do not create per-report `figures/` folders under `reports/`. When writing LaTeX,
reference paths like `../outputs/baseline/task_a/...` or `../outputs/sensitivity/...`.

Example (from a report under `reports/18-12-25/`):

```tex
\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{../../outputs/baseline/task_a/task_a_diagnostics.png}
  \caption{Task A diagnostics.}
\end{figure}
```

## Numba acceleration

`numba_kernels.py` provides JIT-compiled versions of the intercept-only samplers. The
first run compiles kernels and may take 10-30 seconds. Subsequent runs are fast. If needed,
set `NUMBA_DISABLE_JIT=1` to force pure Python (slower but avoids compile delays).

## Running the code

### Baselines

```bash
task-a-baseline
task-b-baseline
```

### Sensitivity analyses

```bash
task-a-sensitivity
task-b-sensitivity
```

### Alternative (no package install)

```bash
PYTHONPATH=src python -m informal_bids.cli
```

## Windows setup

1) Install Python 3.9+ (3.11 recommended) from python.org
2) Check "Add Python to PATH" during installation
3) Open Command Prompt and verify:

```bat
python --version
```

4) Create and activate a virtual environment:

```bat
cd C:\Users\YourName\Documents\informal_bids
python -m venv .venv
.venv\Scripts\activate
```

5) Install dependencies:

```bat
pip install -r requirements.txt
pip install -e .
```

6) Run baselines or sensitivity analyses (same commands as above)

## VS Code setup (recommended)

1) Install VS Code from code.visualstudio.com
2) Install the Microsoft Python extension
3) File -> Open Folder -> select the `informal_bids` folder
4) Select interpreter: Ctrl+Shift+P -> "Python: Select Interpreter" -> `.venv`
5) Run scripts using the play button or the integrated terminal

## Command-line usage (PowerShell basics)

```powershell
cd C:\Users\YourName\Documents\informal_bids
.venv\Scripts\activate

task-a-baseline
task-b-baseline

task-a-sensitivity
task-b-sensitivity

deactivate
```

If CLI entry points are not available:

```powershell
set PYTHONPATH=src
python -m informal_bids.cli
```

Or run a specific entry point:

```powershell
python -c "from informal_bids.cli import task_a_baseline; task_a_baseline()"
```

## Troubleshooting

- "python" not recognized: reinstall Python and check "Add Python to PATH".
- "pip" not recognized: use `python -m pip` instead of `pip`.
- Import errors: make sure the virtual environment is active and `pip install -e .` ran.
- Numba compilation slow: first run is expected; set `NUMBA_DISABLE_JIT=1` to bypass.
- Plots not visible: outputs are saved under `outputs/` (check `outputs/baseline/...`).
- Out of memory: reduce `n_iterations` or increase `thinning` in `MCMCConfig`.

## Structure

```
src/informal_bids/              # Core package code (Python)
meeting/                        # Meeting notes (by date)
contrib/austin/                 # Austin scratch/drafts (non-canonical)
contrib/alex/                   # Alex scratch/drafts + reference materials
contrib/alex/external/          # Alex-provided papers / Matlab reference code
data/                           # (Future) raw/processed data
outputs/                        # Generated plots/results (ignored)
reports/                        # LaTeX reports / PDFs
```

## Summary

This codebase implements Gibbs sampling with data augmentation to estimate informal-bid
admission cutoffs under interval restrictions. It provides Task A (single cutoff) and
Task B (type-specific cutoffs) simulations, sensitivity analyses, and a full set of
diagnostics and plots for evaluating estimator performance.
