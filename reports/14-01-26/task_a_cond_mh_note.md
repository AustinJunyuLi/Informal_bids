# Task A (Jan 14, 2026): Conditional-Probability + MH Implementation Note

## Goal (from meeting 2026-01-14)
Fix the formal-stage selection problem in the baseline debugging case (`b^I = v`) by using the correct conditional likelihood,
implemented as a selection-penalized cutoff draw (MH-within-Gibbs). Then run conversion diagnostics at higher conversion rates
(lower true cutoffs).

## What was implemented (branch `cond`)
- **Selection-aware cutoff update (Task A)**: replaced the naive truncated-normal cutoff draw with an **independence MH** step.
  - Proposal: naive truncated normal (same as the old Gibbs step).
  - Acceptance probability (as in meeting notes):
    \[
    \alpha = \min\left(1, \frac{\Pr(S=1\mid b_{old})}{\Pr(S=1\mid b_{prop})}\right).
    \]
  - Baseline selection probability (debugging case):
    \[
    \Pr(S=1\mid b) = 1 - \Phi\left(\frac{b-\mu_v}{\sigma_v}\right)^J
    \]
    using **fixed DGP inputs** `(mu_v, sigma_v, J)`.
- **Numba**: added a JIT kernel for the intercept-only Task A MH-within-Gibbs loop.
- **Sensitivity output**: Task A sensitivity now records conversion diagnostics (`n_initiated`, `keep_rate_pct`, etc.) and
  runs the meeting cutoffs **`b* = 1.2` and `b* = 1.1`**.

## How to reproduce
- Baseline (b*=1.4):
  - `task-a-baseline`
- Sensitivity (b*=1.2 and b*=1.1):
  - `task-a-sensitivity`

Artifacts are written under `outputs/` (not tracked).

## Quick comparison to the Dec 18 report (`reports/18-12-25/analysis_report.pdf`)
Baseline Task A (N=100, J=3, mu_v=1.3, sigma_v=0.2, b*=1.4):
- **Dec 18 report (naive / basic)**: posterior mean around **1.387** with CI roughly **[1.367, 1.406]** (downward bias).
- **`cond` baseline (selection-aware MH)**: posterior mean **1.4024** with CI **[1.3830, 1.4214]**.

Sensitivity (headline):
- **Basic (b*=1.4)** showed severe undercoverage as N grows (coverage collapses by N=500).
- **Cond+MH conversion diagnostics**:
  - `b*=1.2`: high conversion (~96–97% keep rate) and coverage near nominal in the generated runs.
  - `b*=1.1`: extremely high conversion (~99.6% keep rate) but many all-admit one-sided bounds; coverage deteriorates at
    larger N in the generated runs.

## Notes / known issues
- Very high conversion (b*=1.1) produces a large share of one-sided (all-admit) auctions, which reduces information content.
  Remaining undercoverage there may reflect that data environment more than selection bias per se.

## Next step
Proceed to **Task B (Jan 14, 2026 two-stage simulation + estimation)** using the selection-aware structure as the baseline
building block.
