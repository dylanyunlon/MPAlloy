# Claude #2 — MPAlloy Review: M004–M006

## Fixes Applied

### M004: `CrossPrecisionAllReduce.allreduce_gradients()` — Weighted Average
**File**: `alloy/mixed_precision/gradient_verifier.py:47-115`

The original allreduce used `stacked.mean(dim=0)` — a simple average treating
all devices equally. But in a tiered system, H100 holds fewer hot rows than
A6000s hold warm rows. A simple mean overweights the H100's gradient contribution
relative to the data it actually processed.

**Fix**: Accept optional `row_counts: Dict[torch.device, int]` parameter.
Compute `weighted_sum = Σ(grad_i × w_i / total_weight)` where weights are
proportional to row counts. Falls back to uniform when row_counts is None or
total_weight ≤ 0.

### M005: `FPRevDiagnosticBatch` — NaN/Inf Sentinel Detection
**File**: `alloy/mixed_precision/gradient_verifier.py:70-86, 150-170`

NaN or Inf in a gradient (from FP8 overflow, division by zero, or loss explosion)
would propagate through allreduce and poison every device's parameters silently.

**Fix**: Two parts:
1. In `allreduce_gradients`: scan each gradient for NaN/Inf before aggregation.
   If found, log a warning with device name and counts, replace with zero via
   `torch.where(torch.isfinite(...), grad, zeros)`.
2. In `generate_distinguishing_inputs`: new `"nan_inf"` strategy that deliberately
   injects NaN (5%) and Inf (2%) to verify the sentinel catches them.

### M006: `compute_drift()` — Theoretical ULP Error Bound
**File**: `alloy/mixed_precision/gradient_verifier.py:117-167`

The original drift computation reported only observed max abs/rel errors.
Without a theoretical bound, there's no way to know whether observed drift
is within expected limits or indicates a bug.

**Fix**: For each device pair, compute:
- `eps_source`, `eps_target` from dtype machine epsilon table
- `theoretical_bound = max_magnitude × (eps_source + eps_target)`
- `bound_exceeded = abs_err > theoretical_bound`
- `ulp_distance = abs_err / (max_magnitude × coarser_eps)`

Also tracks drift over time in `_drift_history`.

## Data Generator

**File**: `experiments/generate_alloy_data.py` (new, 270 lines)

Produces 4 JSON files in exact demo `data.zip` schema:

| Output file | Demo counterpart | Schema |
|---|---|---|
| `alloy_tiered_placement_data.json` | `reversed_figure_data.json` | panels → methods → {final_perplexity, curves:[3×3000]} |
| `alloy_gradient_drift_data.json` | `gradient_norm_24k_data.json` | {steps:[2000], methods → {seed_0..2, mean, std}[2000]} |
| `alloy_convergence_vs_time_data.json` | `ppl_vs_time_1B_30k_data.json` | {methods → {time_hours, total_time, reported_final, seed_0..2}} |
| `alloy_migration_norms_data.json` | `reversed_figure18_data.json` | {panels → {title, methods → {seed_0..2, mean, std}}} |

Schema validation against demo: ✓ All keys, list lengths, and nesting match exactly.

## Test Results

```
M004 weighted average: ✓ (single device, multi-device with row_counts)
M005 NaN/Inf sentinel: ✓ (NaN→0, Inf→0, report generated, nan_inf strategy works)
M006 error bound: ✓ (theoretical_bound, bound_exceeded, ulp_distance computed)
Data generator: ✓ (4 files, schema validated against demo format)
```
