# Claude #3 — MPAlloy Review: M007–M009

## Fixes Applied

All three changes land in `alloy/scheduler/tiered_placement.py` and target the
same failure mode: **migration thrash**. The scheduler decided tier placement
purely from a hard-thresholded EMA, with nothing damping decisions over time.
On a real DLRM trace, embedding hotness jitters by a few percent step-to-step,
so a row whose EMA frequency sits near `hot_threshold=0.3` would oscillate
H100→A6000→H100 on consecutive plans — saturating PCIe with cross-precision
casts (FP8↔BF16) that accomplish nothing.

### M007: `AccessFrequencyTracker.classify_tier_hysteresis()` — Dead Band

**File**: `alloy/scheduler/tiered_placement.py` (new method on the tracker;
`EmbeddingMeta` gains `pending_target`, `pending_streak`, `last_migration_time`).

The original `classify_tier()` mapped frequency to tier with a single edge per
boundary. Any EMA jitter straddling that edge flips the tier every step.

**Fix**: Split each boundary into an upper (promote) edge and a lower (demote)
edge separated by a configurable dead band (`band`, default 0.2 → ±10% around
the nominal threshold). Promotion requires crossing the higher edge; demotion
requires falling below the lower edge; inside the band the row keeps its current
tier. The decision now depends on `current_tier`, so it is stable by
construction. `classify_tier()` is retained unchanged for callers that want the
stateless mapping.

### M008: Migration Cooldown

**File**: `compute_placement_plan()` + `execute_migration()`.

Even with hysteresis, a chunk could still migrate on back-to-back plans if its
frequency genuinely crossed a band edge each time. Nothing bounded the per-chunk
migration *rate*.

**Fix**: `execute_migration()` stamps `meta.last_migration_time`. A new
`migration_cooldown_s` (default 5.0s) gate in `compute_placement_plan()` skips
any chunk that moved within the window. This caps per-chunk migration at
1/cooldown regardless of how violently its hotness swings, bounding worst-case
PCIe churn.

### M009: Confirmation Streak

**File**: `compute_placement_plan()`.

A single transient hotness spike (one unusually hot batch) shouldn't trigger a
migration that gets reversed next step.

**Fix**: A new target tier must be proposed by `confirm_streak` (default 3)
*consecutive* plans before the migration fires. Confirmation state lives on the
`EmbeddingMeta` (`pending_target`, `pending_streak`) so it persists across calls
and resets the moment the row stabilizes or the proposal changes. Suppressed
attempts are counted in `_suppressed_migrations` and surfaced via
`get_migration_stats()` for tuning the cooldown/streak knobs against a real
trace.

## Knuth-level second pass

- **Lock discipline**: `compute_placement_plan()` now mutates confirmation state
  on the metas, so the read loop runs under `self._lock` — consistent with
  `execute_migration()`, which already takes the lock before touching
  `current_tier`/`migration_count`. No new lock gap introduced.
- **Capacity-fallback interaction**: hysteresis runs *before* the capacity
  fallback, so a row denied its desired tier still demotes deterministically;
  the streak then confirms the *fallback* tier, not the blocked one. Correct.
- **Cooldown vs. first placement**: `last_migration_time` defaults to 0.0, so a
  never-migrated chunk is never blocked by cooldown — initial placement is
  unaffected.
- **Tradeoff accepted**: hysteresis + streak + cooldown add latency to a
  *legitimate* sustained hotness change (worst case `confirm_streak` plans, then
  the cooldown of its previous move). For embedding workloads this is the right
  bias — a few seconds of suboptimal placement is far cheaper than continuous
  cross-precision migration. Knobs are exposed for workloads that disagree.

## Test Results

```
M007 hysteresis: tier flips under jitter = 1   (naive classifier = 8)  ✓
M008 cooldown:   migrations during cooldown = 0                        ✓
M009 streak:     migration fires on 3rd consecutive plan               ✓
suppressed_migrations counter increments correctly                     ✓
```

CPU-only validation (no GPU on this VM); GPU-side latency impact should be
measured on the target A6000×2 + H100×1 server via `scripts/run_exp3.sh`.
