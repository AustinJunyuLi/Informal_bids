# Informal Bids

Selection-aware Bayesian estimation for two-stage M&A auctions.

> **This branch (`two_stage`)** builds on `cond`: Task A uses the selection-aware MH correction,
> and this branch implements the Jan 14, 2026 Task B two-stage simulation/estimation (including
> staged variance relaxation and an optional moments-based cutoff). The naive baseline lives on
> branch `basic`.

---

## Table of Contents

1. [The Selection Problem](#1-the-selection-problem)
2. [Economic Model](#2-economic-model)
3. [Selection-Corrected Likelihood](#3-selection-corrected-likelihood)
4. [MCMC Implementation](#4-mcmc-implementation)
5. [The kappa Reparametrization](#5-the-kappa-reparametrization)
6. [Cutoff Specifications](#6-cutoff-specifications)
7. [Identification Analysis](#7-identification-analysis)
8. [Staged Variance Relaxation](#8-staged-variance-relaxation)
9. [Data Structures](#9-data-structures)
10. [Quick Start](#10-quick-start)
11. [Module Reference](#11-module-reference)
12. [Convergence Diagnostics](#12-convergence-diagnostics)
13. [Project Structure](#13-project-structure)
14. [Known Issues and Limitations](#14-known-issues-and-limitations)
15. [References](#15-references)

---

## 1. The Selection Problem

### What It Is

Researchers observe auctions that reach the formal stage. Define the selection event:

```
S_i = 1  <=>  exists j such that b^I_ij >= b*_i
```

Auctions with `S_i = 0` (all informal bids below cutoff) are unobserved and excluded from the sample.

### Why It Matters

A naive estimator maximizes:

```
L_naive(theta) = prod_{i: S_i=1} p(data_i | theta)
```

This treats all observed auctions as representative. As N grows:
- Posterior concentrates around a pseudo-true value theta* != theta_0
- Credible intervals shrink around the wrong value
- Coverage rates collapse (e.g., 100% at N=20 -> 0% at N=500)

### Mathematical Definition

For an auction with J bidders and cutoff b*:

**Task A (truthful bidding):**
```
Pr(S=1 | b*, mu_v, sigma_v, J) = 1 - Phi((b* - mu_v)/sigma_v)^J
```

**Task B (with misreporting):**
```
Pr(S=1 | b*, gamma, sigma_nu, J, kappa) = 1 - Phi((b*/lambda_I - gamma)/sigma_nu)^J
```

where `lambda_I` is the informal-stage bid multiplier.

**Code:**
- `utils.py:139` - `selection_prob_at_least_one_exceeds_cutoff()`
- `utils.py:190` - `selection_prob_reaches_formal_stage()`

---

## 2. Economic Model

### 2.1 Valuation Process

Bidder j in auction i has:

```
v_ij = gamma + nu_ij,    nu_ij ~ N(0, sigma_nu^2)     [initial valuation]
u_ij = v_ij + eta_ij,    eta_ij ~ N(0, sigma_eta^2)   [post-due-diligence valuation]
```

**Derivation:** The bidder's prior valuation v_ij is drawn from a normal distribution centered at gamma (common mean). After due diligence, new information eta_ij updates the valuation to u_ij.

### 2.2 Bidding Equations

**Informal bids (with misreporting):**
```
b^I_ij = lambda_I(J, kappa) * v_ij
```

**Formal bids (standard first-price shading):**
```
b^F_ij = lambda_f * u_ij = (1 - 1/J) * u_ij    [admitted bidders only]
```

**Derivation:** In a first-price auction with J symmetric bidders and private values drawn from F, the equilibrium bid is:
```
b(v) = v - integral_0^v [F(x)/F(v)]^{J-1} dx
```

For uniform or normal valuations with sufficient competition, this approximates `b(v) = (1 - 1/J) * v`.

The informal stage adds a misreporting factor because bids are non-binding. See Section 5 for the kappa reparametrization.

**Code:** `data.py:162-171` - `lambda_f`, `lambda_i` properties of `TaskBDGPParameters`

### 2.3 Admission Rule and Interval Bounds

**Admission rule:**
```
Bidder j admitted  <=>  b^I_ij >= b*_i
```

**Interval bounds from observed admission:**
- `L_i = max(rejected bids)` - highest bid that was rejected
- `U_i = min(admitted bids)` - lowest bid that was admitted
- `b*_i in [L_i, U_i]` for two-sided cases

**Three cases:**
| Case | Description | Bounds |
|------|-------------|--------|
| Two-sided | Some admitted, some rejected | L_i, U_i both finite |
| One-sided upper | All admitted | L_i = -inf, U_i = min admitted |
| All-reject | None admitted | Auction unobserved (S_i = 0) |

**Code:** `data.py:354-391` - `TaskADataGenerator.generate_auction_data()`, `data.py:460-471` - bounds computation in `TaskBDataGenerator`

### 2.4 Cutoff Model

The latent cutoff follows a linear model:

```
b*_i = X_i' beta_cutoff + omega_i,    omega_i ~ N(0, sigma_omega^2)
```

where X_i includes an intercept and potentially moments of informal bids.

---

## 3. Selection-Corrected Likelihood

### 3.1 The Conditional Likelihood Principle

For observed auctions, the correct likelihood conditions on selection:

```
p(data_i | theta, S_i=1) = p(data_i | theta) / Pr(S_i=1 | theta)
```

**Derivation:** By Bayes' rule,
```
p(data_i | theta, S_i=1) = p(data_i, S_i=1 | theta) / Pr(S_i=1 | theta)
                         = p(data_i | theta) * Pr(S_i=1 | data_i, theta) / Pr(S_i=1 | theta)
```

Since S_i is deterministic given the data (selection is determined by max bid vs cutoff):
```
Pr(S_i=1 | data_i, theta) = 1    for observed auctions
```

Taking logs:
```
log p(data_i | theta, S_i=1) = log p(data_i | theta) - log Pr(S_i=1 | theta)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^
                                                        selection penalty
```

### 3.2 Selection Probability Formula (Task A)

**Setting:** Informal bids equal valuations, `b^I_ij = v_ij ~ N(mu_v, sigma_v^2)`.

**Derivation:**
```
Pr(S=1 | b*) = 1 - Pr(all bids < b*)
             = 1 - prod_{j=1}^J Pr(v_ij < b*)
             = 1 - Phi((b* - mu_v)/sigma_v)^J
```

**Behavior:**
- b* -> -inf: Pr(S=1) -> 1 (low cutoff, almost certainly someone exceeds)
- b* -> +inf: Pr(S=1) -> 0 (high cutoff, unlikely anyone exceeds)

**Code:** `utils.py:139-174` - `selection_prob_at_least_one_exceeds_cutoff()`

### 3.3 Selection Probability Formula (Task B)

**Setting:** Informal bids involve misreporting, `b^I_ij = lambda_I * v_ij`.

**Derivation:**
```
S_i = 1  <=>  exists j: b^I_ij >= b*
         <=>  exists j: lambda_I * v_ij >= b*
         <=>  exists j: v_ij >= b* / lambda_I
```

Therefore:
```
Pr(S=1 | b*) = 1 - Pr(all valuations < b*/lambda_I)
             = 1 - Phi((b*/lambda_I - gamma)/sigma_nu)^J
```

**Code:** `utils.py:190-225` - `selection_prob_reaches_formal_stage()`

### 3.4 The Corrected Posterior Kernel

The posterior kernel for the latent cutoff is:

```
pi(b | L_i, U_i, mu, sigma_omega, mu_v, sigma_v, J) propto
    phi((b-mu)/sigma_omega)                    [cutoff model]
  * 1/(1 - Phi((b-mu_v)/sigma_v)^J)            [selection penalty]
  * I{L_i <= b <= U_i}                         [interval restriction]
```

This is **not** a truncated normal because the selection penalty `1/Pr(S=1|b)` varies with b. This necessitates Metropolis-Hastings sampling.

---

## 4. MCMC Implementation

### 4.1 Independence MH for Cutoffs

**Why MH is needed:** The selection-corrected kernel is not a standard distribution. We use independence MH with the naive truncated normal as proposal.

**Algorithm:**
```
1. Propose:  b_prop ~ TruncNormal(X_i' beta, sigma_omega, [L_i, U_i])
2. Compute acceptance ratio:
      alpha = min(1, Pr(S=1 | b_old) / Pr(S=1 | b_prop))
3. Accept b_prop with probability alpha; else keep b_old
```

**Derivation of acceptance ratio:**

For target pi(b) and proposal q(b), the MH ratio is:
```
alpha = min(1, [pi(b_prop) * q(b_old|b_prop)] / [pi(b_old) * q(b_prop|b_old)])
```

For independence proposal (q doesn't depend on current state):
```
q(b_old|b_prop) = q(b_old)
q(b_prop|b_old) = q(b_prop)
```

The target and proposal share the truncated normal factor:
```
pi(b) propto phi((b-mu)/sigma) * [1/Pr(S=1|b)] * I{L <= b <= U}
q(b) propto phi((b-mu)/sigma) * I{L <= b <= U}
```

So:
```
pi(b_prop)/pi(b_old) * q(b_old)/q(b_prop)
  = [1/Pr(S=1|b_prop)] / [1/Pr(S=1|b_old)]
  = Pr(S=1|b_old) / Pr(S=1|b_prop)
```

**Intuition:**
- If b_prop > b_old: Selection less likely -> ratio > 1 -> always accept
- If b_prop < b_old: Selection more likely -> ratio < 1 -> sometimes reject

This asymmetry shifts the equilibrium distribution toward higher cutoffs, correcting selection bias.

**Code:** `samplers.py:178-196` - Task A cutoff update, `samplers.py:545-579` - Task B cutoff update

### 4.2 Task A Update Steps (3 Blocks)

```
for t in range(n_iterations):
    # Step 1: Update latent cutoffs b*_i via independence MH
    for i in range(N):
        b_prop = sample_truncated_normal(X_i' beta, sigma_omega, L_i, U_i)
        p_old = selection_prob_at_least_one_exceeds_cutoff(b_star[i], ...)
        p_prop = selection_prob_at_least_one_exceeds_cutoff(b_prop, ...)
        alpha = min(1, p_old / p_prop)
        if rand() < alpha:
            b_star[i] = b_prop

    # Step 2: Update beta (conjugate normal)
    V_post = inv(V0_inv + X'X / sigma^2)
    beta_post = V_post * (V0_inv * beta_prior + X'b_star / sigma^2)
    beta ~ N(beta_post, V_post)

    # Step 3: Update sigma^2 (conjugate inverse-gamma)
    a_post = a_prior + N/2
    b_post = b_prior + sum((b_star - X*beta)^2)/2
    sigma^2 ~ InvGamma(a_post, b_post)
```

**Code:** `samplers.py:127-222` - `TaskAMCMCSampler.run_chain()`

### 4.3 Task B Update Steps (7 Blocks)

| Block | Parameter | Update Method |
|-------|-----------|---------------|
| 1 | b*_i (latent cutoffs) | Independence MH with selection penalty |
| 2 | beta_cutoff | Conjugate normal regression |
| 3 | sigma_omega^2 | Conjugate inverse-gamma |
| 4 | gamma | Random-walk MH with selection penalty |
| 5 | sigma_nu^2 (Stage 3) | Inverse-gamma proposal with selection-aware accept/reject |
| 6 | sigma_eta^2 (Stages 2-3) | Conjugate inverse-gamma (admitted bidders only) |
| 7 | kappa | Random-walk MH |

**Code:** `samplers.py:467-812` - `TaskBMCMCSampler.run_chain()`

### 4.4 Numba Kernels

For performance, the innermost loops are JIT-compiled with Numba:

| Function | Purpose | Location |
|----------|---------|----------|
| `task_a_run_chain_intercept()` | Full Task A chain (intercept-only) | `numba_kernels.py:155-231` |
| `task_b_update_b_star()` | Task B cutoff MH update | `numba_kernels.py:270-308` |
| `task_b_logpost_gamma()` | Log-posterior for gamma | `numba_kernels.py:312-343` |
| `task_b_logpost_kappa()` | Log-posterior for kappa | `numba_kernels.py:395-445` |

---

## 5. The kappa Reparametrization

### 5.1 The Constraint Problem

The informal-stage multiplier must satisfy:
```
lambda_I = (1 - 1/J) + alpha > 0
```

where alpha is the additive misreporting parameter. This requires:
```
alpha > -(1 - 1/J) = -lambda_f
```

Enforcing this constraint during MCMC (e.g., via truncation) leads to boundary issues.

### 5.2 The Transformation

We reparametrize using:
```
lambda_I = (1 - 1/J) * exp(kappa) = lambda_f * exp(kappa)
```

Define `tilde_alpha = lambda_I / lambda_f - 1 = exp(kappa) - 1`.

| kappa | exp(kappa) | tilde_alpha | Interpretation |
|-------|------------|-------------|----------------|
| 0 | 1 | 0 | No misreporting |
| > 0 | > 1 | > 0 | Overbidding |
| < 0 | < 1 | < 0 | Underbidding |

Now kappa is unconstrained and can be sampled via standard random-walk MH.

### 5.3 Relationship to Meeting Notes

The meeting notes use an additive parameterization:
```
lambda_I = (1 - 1/J) + alpha_additive
```

The relationship:
```
alpha_additive = lambda_I - lambda_f = lambda_f * (exp(kappa) - 1)
tilde_alpha = exp(kappa) - 1
```

**Code:** `utils.py:28-45` - `informal_bid_multiplier()`, `utils.py:48-66` - `misreporting_measures()`

---

## 6. Cutoff Specifications

### 6.1 Intercept-Only (Baseline)

```
b*_i = c + omega_i
X_i = [1]
beta_cutoff = [c]
```

**Code:** `cutoff_spec = "intercept"` in `TaskBDGPParameters`

### 6.2 Moments-Based (Meeting Specification)

```
b*_i = c + theta_1 * m_{i,1} + theta_2 * m_{i,2} + theta_3 * m_{i,3} + omega_i
```

where:
- `m_{i,1} = b^{I(1)}_i` (max bid)
- `m_{i,2} = (b^{I(1)}_i + b^{I(2)}_i)/2` (top-2 average)
- `m_{i,3} = (b^{I(1)}_i + b^{I(2)}_i + b^{I(3)}_i)/3` (top-3 average)

```
X_i = [1, m_{i,1}, m_{i,2}, m_{i,3}]
beta_cutoff = [c, theta_1, theta_2, theta_3]
```

**Code:** `cutoff_spec = "moments_k4"`, feature computation at `utils.py:93-118`

### 6.3 Depth-Based (Better Identified)

```
b*_i = c + theta_1 * (b^{I(2)}_i + b^{I(3)}_i)/2 + theta_2 * (b^{I(2)}_i - b^{I(3)}_i) + omega_i
```

where:
- Runner-up mean: `(b^{I(2)}_i + b^{I(3)}_i)/2`
- Runner-up gap: `b^{I(2)}_i - b^{I(3)}_i`

```
X_i = [1, depth_mean_23, depth_gap_23]
beta_cutoff = [c, theta_1, theta_2]
```

**Code:** `cutoff_spec = "depth_k2"`, feature names at `utils.py:69-79`

---

## 7. Identification Analysis

### 7.1 What Is Identified and From What

| Parameter | Identified From | Data Used |
|-----------|-----------------|-----------|
| gamma (mean valuation) | Formal bid distribution | b^F_ij = (1-1/J) * u_ij |
| tilde_alpha (misreporting) | Informal/formal bid ratio | b^I_ij / b^F_ij relationship |
| Cutoff params | Admission pattern | Where cutoff falls vs bid distribution |

### 7.2 Why Moments Cutoff Fails

**Problem 1: Collinearity**

The moments are highly correlated (all include the max bid):

| Feature | Formula |
|---------|---------|
| m_1 | b^{I(1)} |
| m_2 | (b^{I(1)} + b^{I(2)})/2 |
| m_3 | (b^{I(1)} + b^{I(2)} + b^{I(3)})/3 |

Diagnostics show:
- Max pairwise correlation: |rho| > 0.9
- Condition number: > 6
- Variance Inflation Factors: One VIF > 10

**Problem 2: Mechanical Cancellation**

The selection condition with max bid in cutoff:
```
S_i = 1  <=>  b^{I(1)}_i >= b*_i = c + theta_1 * b^{I(1)}_i + ...
```

Rearranging:
```
(1 - theta_1) * b^{I(1)}_i >= c + theta_2 * m_2 + theta_3 * m_3 + omega_i
```

As theta_1 -> 1, the max bid's role collapses, producing weak identification.

**Symptom:** Erratic coefficient recovery that doesn't improve with sample size.

**Code:** `analysis.py:12-97` - `compute_collinearity_diagnostics()`

### 7.3 Why Depth-Based Works Better

The depth-based cutoff excludes the max bid from covariates:
```
b*_i = c + theta_1 * (runner-up mean) + theta_2 * (runner-up gap) + omega_i
```

Selection condition:
```
b^{I(1)}_i >= c + theta_1 * (b^{I(2)} + b^{I(3)})/2 + theta_2 * (b^{I(2)} - b^{I(3)}) + omega_i
```

The max bid appears **only on the left side**, avoiding mechanical cancellation.

Collinearity diagnostics: VIF near 1 (vs 10+ for moments cutoff).

---

## 8. Staged Variance Relaxation

To assess identification incrementally, estimation proceeds in stages:

| Stage | sigma_nu | sigma_eta | Estimated Parameters |
|-------|----------|-----------|---------------------|
| 1 | Fixed | Fixed | gamma, tilde_alpha, cutoff params, sigma_omega |
| 2 | Fixed | Estimated | + sigma_eta |
| 3 | Estimated | Estimated | All parameters including sigma_nu |

**Configuration:**
```python
MCMCConfig(
    task_b_stage=1,  # or 2, 3
    task_b_sigma_nu_fixed=0.2,
    task_b_sigma_eta_fixed=0.1,
)
```

**Code:** `data.py:207` - `task_b_stage` field, `samplers.py:401-415` - `_stage_sigmas()`

---

## 9. Data Structures

### 9.1 DGP Parameters

**TaskADGPParameters** (`data.py:34-80`)

| Field | Type | Description |
|-------|------|-------------|
| `N` | int | Number of observed auctions |
| `J` | int | Bidders per auction |
| `mu_v` | float | Mean valuation |
| `sigma_v` | float | Valuation std dev |
| `b_star` | float | Baseline cutoff (intercept) |
| `beta` | ndarray | Cutoff coefficients (including intercept) |
| `sigma_b` | float | Cutoff shock std dev |

**TaskBDGPParameters** (`data.py:84-185`)

| Field | Type | Description |
|-------|------|-------------|
| `N` | int | Number of observed auctions |
| `J` | int | Bidders per auction |
| `gamma` | float | Mean valuation |
| `sigma_nu` | float | Valuation shock std dev |
| `sigma_eta` | float | Due diligence shock std dev |
| `kappa` | float | Unconstrained misreporting parameter |
| `misreporting_mode` | str | "scale" (default) or "shift" |
| `cutoff_spec` | str | "intercept", "moments_k4", or "depth_k2" |
| `beta_cutoff` | ndarray | Cutoff coefficients |
| `sigma_omega` | float | Cutoff shock std dev |

### 9.2 MCMCConfig

**MCMCConfig** (`data.py:193-247`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_iterations` | int | 20000 | Total MCMC iterations |
| `burn_in` | int | 10000 | Burn-in iterations |
| `thinning` | int | 10 | Thinning interval |
| `n_chains` | int | 3 | Number of parallel chains |
| `task_b_stage` | int | 1 | Estimation stage (1, 2, or 3) |
| `task_b_sigma_nu_fixed` | float | None | Fixed sigma_nu value |
| `task_b_sigma_eta_fixed` | float | None | Fixed sigma_eta value |
| `task_b_use_numba` | bool | True | Enable Numba acceleration |

### 9.3 Auction Data Classes

**TaskAAuctionData** (`data.py:255-271`)

| Field | Type | Description |
|-------|------|-------------|
| `auction_id` | int | Unique identifier |
| `X_i` | ndarray | Cutoff covariates |
| `bids` | ndarray | All bids in auction |
| `admitted` | ndarray | Boolean mask |
| `L_i` | float | Lower bound |
| `U_i` | float | Upper bound |
| `is_complete` | bool | Both bounds finite |

**TaskBAuctionData** (`data.py:284-301`)

| Field | Type | Description |
|-------|------|-------------|
| `auction_id` | int | Unique identifier |
| `X_i` | ndarray | Cutoff covariates |
| `informal_bids` | ndarray | Shape (J,) |
| `admitted` | ndarray | Shape (J,), boolean |
| `formal_bids` | ndarray | Shape (J,), NaN for rejected |
| `L_i` | float | Lower bound |
| `U_i` | float | Upper bound |
| `is_complete` | bool | Both bounds finite |
| `n_bidders` | int | Number of bidders |

### 9.4 Generators

| Class | Location | Purpose |
|-------|----------|---------|
| `TaskADataGenerator` | `data.py:308-406` | Generate Task A simulated data |
| `TaskBDataGenerator` | `data.py:409-498` | Generate Task B simulated data |
| `RealDataLoader` | `data_io.py:21-146` | Load real data from CSV |

---

## 10. Quick Start

### 10.1 Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 10.2 CLI Commands

| Command | Description |
|---------|-------------|
| `task-a-baseline` | Task A single cutoff estimation |
| `task-a-sensitivity` | Task A sample size sensitivity |
| `task-b-baseline` | Task B moments cutoff baseline |
| `task-b-baseline-depth` | Task B depth-based cutoff baseline |
| `task-b-sensitivity` | Task B moments cutoff sensitivity |
| `task-b-sensitivity-depth` | Task B depth cutoff sensitivity |
| `task-b-intercept-baseline` | Task B intercept-only (Stages 1-3) |
| `task-b-intercept-sensitivity` | Task B intercept-only sensitivity |

Outputs are written to `outputs/baseline/` and `outputs/sensitivity/`.

### 10.3 Programmatic Example

```python
import numpy as np
from informal_bids import (
    TaskADGPParameters,
    TaskADataGenerator,
    TaskAMCMCSampler,
    TaskAResultsAnalyzer,
    MCMCConfig,
)

# Set up DGP
dgp = TaskADGPParameters(N=100, J=3, mu_v=1.3, sigma_v=0.2, b_star=1.4)

# Generate data
generator = TaskADataGenerator(dgp)
auctions, stats = generator.generate_auction_data()
print(f"Keep rate: {stats['keep_rate_pct']:.1f}%")

# Run MCMC
config = MCMCConfig(n_iterations=10000, burn_in=5000, n_chains=2)
sampler = TaskAMCMCSampler(auctions, config, bid_mu=dgp.mu_v, bid_sigma=dgp.sigma_v)
results = sampler.run()

# Analyze
analyzer = TaskAResultsAnalyzer(results, dgp.b_star)
metrics = analyzer.compute_metrics()
print(f"Bias: {metrics['bias']:.4f}, Coverage: {metrics['coverage']}")
```

---

## 11. Module Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `data.py` | DGP parameters, data classes, generators | `TaskADGPParameters`, `TaskBDGPParameters`, `MCMCConfig`, `TaskAAuctionData`, `TaskBAuctionData`, `TaskADataGenerator`, `TaskBDataGenerator` |
| `data_io.py` | Real data loading | `RealDataLoader` |
| `samplers.py` | MCMC samplers | `TaskAMCMCSampler`, `TaskBMCMCSampler` |
| `analysis.py` | Results analysis and metrics | `TaskAResultsAnalyzer`, `TaskBResultsAnalyzer`, `compute_parameter_metrics()`, `compute_collinearity_diagnostics()` |
| `visualization.py` | Publication-quality plots | `TaskAVisualizer`, `TaskBVisualizer` |
| `sensitivity.py` | Batch sensitivity analyses | `TaskASensitivityAnalysis`, `TaskBSensitivityAnalysis` |
| `utils.py` | Statistical utilities | `selection_prob_at_least_one_exceeds_cutoff()`, `selection_prob_reaches_formal_stage()`, `sample_truncated_normal()`, `gelman_rubin()`, `informal_bid_multiplier()`, `misreporting_measures()`, `compute_cutoff_features()`, `cutoff_feature_names()` |
| `numba_kernels.py` | JIT-compiled MCMC loops | `task_a_run_chain_intercept()`, `task_b_update_b_star()`, `task_b_logpost_gamma()`, `task_b_logpost_kappa()` |
| `config.py` | Paths and plotting setup | `configure_plotting()`, path constants |
| `cli.py` | Command-line entry points | `task_a_baseline()`, `task_b_baseline()`, etc. |

---

## 12. Convergence Diagnostics

### Gelman-Rubin R-hat

Compares within-chain and between-chain variance across M independent chains:

```
R-hat = sqrt(V-hat / W)
```

where V-hat is the estimated marginal posterior variance and W is the within-chain variance.

| R-hat | Interpretation |
|-------|----------------|
| < 1.01 | Excellent convergence |
| 1.01-1.05 | Good convergence |
| 1.05-1.10 | Acceptable; longer chains may help |
| > 1.10 | Convergence concern; investigate |

**Code:** `utils.py:295-326` - `gelman_rubin()`

### MH Acceptance Rate Interpretation

| Rate | Interpretation |
|------|----------------|
| > 90% | Selection pressure weak; naive proposal nearly correct |
| 50-90% | Meaningful selection correction active |
| < 50% | Severe selection; naive proposal poor approximation |

Baseline simulations (~65% conversion) show acceptance rates of 70-80%.

### Effective Sample Size (ESS)

Accounts for autocorrelation:
```
ESS = n / (1 + 2 * sum_{k=1}^inf rho_k)
```

**Target:** ESS > 400 for reliable posterior summaries.

---

## 13. Project Structure

```
Informal_bids/
├── src/informal_bids/         # Core package
│   ├── __init__.py            # Public API exports
│   ├── data.py                # DGP parameters, data classes, generators
│   ├── data_io.py             # Real data loading
│   ├── samplers.py            # MCMC samplers
│   ├── analysis.py            # Results analysis
│   ├── visualization.py       # Plotting
│   ├── sensitivity.py         # Batch analyses
│   ├── utils.py               # Statistical utilities
│   ├── numba_kernels.py       # JIT-compiled functions
│   ├── config.py              # Configuration and paths
│   └── cli.py                 # Command-line entry points
├── meeting/                   # Meeting notes (by date)
├── reports/                   # LaTeX reports / PDFs
│   └── 14-01-26/
│       └── analysis_report.pdf
├── contrib/
│   └── alex/                  # External papers and materials
├── scripts/                   # Utility scripts
├── data/                      # (Future) real data
├── outputs/                   # Generated plots/results (git-ignored)
│   ├── baseline/
│   │   ├── task_a/
│   │   └── task_b/
│   │       ├── intercept/
│   │       ├── moments/
│   │       └── depth/
│   └── sensitivity/
│       ├── task_a/
│       └── task_b/
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 14. Known Issues and Limitations

### Simulation-Only Validation

All results are Monte Carlo simulations. The DGP may not capture:
- Heterogeneous bidder types (strategic vs financial)
- Endogenous participation
- Multi-round informal bidding
- Target-specific unobserved heterogeneity

### Fixed Number of Bidders

Simulations fix J=3. Variable J requires:
- Modified selection probability formula
- Potentially modeling participation decision
- J-dependent bid shading

### Computational Considerations

MH-within-Gibbs is slower than pure Gibbs:
- Selection probability evaluation required for each cutoff draw
- 70-80% acceptance means some wasted draws
- N cutoff updates per iteration dominates runtime

Mitigation: Numba JIT compilation. For N > 500, consider parallelization.

### Prior Sensitivity

Weakly informative priors used throughout. When identification is weak (moments cutoff), results may be prior-sensitive.

### One-Sided Bounds at High Conversion

When b* << mu_v (very high conversion):
- Most auctions are all-admit (one-sided upper bounds only)
- Limited information content
- Coverage may deteriorate despite high conversion

This is a data environment issue, not a selection correction failure.

---

## 15. References

- **Selection correction:** Heckman (1979), "Sample Selection Bias as a Specification Error"
- **Cheap talk:** Kartik (2009), "Strategic Communication with Lying Costs"
- **MCMC diagnostics:** Gelman & Rubin (1992), "Inference from Iterative Simulation"
- **Data augmentation:** Tanner & Wong (1987), "The Calculation of Posterior Distributions by Data Augmentation"
