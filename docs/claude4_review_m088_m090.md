# Claude #4 — MPAlloy Review: M088–M090

> Milestone block #30 from the dev plan: `FrequencyHistogramKernel`
> shared-memory histogram optimisation. Picked over the next sequential
> block (#4, M010–M012) because that block targets `CrucibleFuzzer` /
> `InfluenceGuide`, which live in the companion **CASH** repo, not MPAlloy.
> This block is the next open MPAlloy-resident kernel-architecture task and
> is a structural rewrite (not a parameter knob) in the spirit of CUB
> DeviceTopK's two-level histogram decomposition.

## The bottleneck

Embedding access in DLRM-style workloads is heavily power-law: a small hot
set absorbs most references in every batch. The original kernel was a single
unprivileged loop of `atomicAdd(d_freq_hist + row, 1)`. Every lane that
touched a hot row serialised onto the **same global address** — within a
warp, across a block, and across the grid. On a row hit by all 32 lanes of a
warp the atomic unit, not memory bandwidth, set the runtime. The kernel also
silently dropped any batch tail past `grid·block·ITEMS_PER_THREAD`.

## Fixes Applied

**File**: `alloy/include/alloy/tiered_cache.cuh` (kernel + launch config),
`alloy/src/tiered_embedding.cu` (launch site).

### M088 — Warp-aggregated atomics
New `warp_aggregated_inc()` helper. Active lanes targeting the same row are
discovered with `__match_any_sync`; the lowest lane in each peer group
(`__ffs(peers) - 1`) performs one `atomicAdd` carrying the group population
(`__popc(peers)`). A hot row hit by all 32 lanes now costs **1** atomic
instead of 32. The all-distinct case degrades gracefully to one atomic per
lane (identical to the old behaviour, no penalty). Pre-Volta (`__CUDA_ARCH__
< 700`) lacks `__match_any_sync`, so a guarded `#else` falls back to a plain
atomic — the kernel still compiles and is correct on every architecture.

### M089 — Block-privatised shared-memory path
When the vocabulary is small enough that a `uint32` slot per row fits in one
block's shared memory, each block accumulates into a private shared histogram
(itself fed through the M088 warp aggregator) and flushes to global **once**
at the end. This converts *B* global atomics per row (one per block) plus
intra-block contention into one warp-aggregated shared path + a single flush
atomic per non-zero slot. Path selection is made at launch in
`TieredCacheLaunchConfig::compute()` via the dynamic shared-memory size; when
the table is too large it cleanly falls back to the global M088 path
(`histogram_shmem_bytes = 0`, kernel `num_embeddings` arg = 0).

### M090 — Vectorised grid-stride loads
Replaced the fixed `grid·block·IPT` sweep (which dropped the batch tail) with
a true grid-stride loop, four indices per iteration. Correct for **any** grid
size — important because M089 deliberately caps the grid on the privatised
path, which the old loop would have mishandled by leaving items uncounted.

## Knuth-level second pass

- **Correctness invariant**: the histogram must be bit-identical to a naive
  per-item increment regardless of warp packing or block count. Aggregation
  only changes *who* issues the atomic and *by how much*, never the total.
  Verified (see tests) across partial warps and heavy-collision batches.
- **Shared-path race**: the private histogram is zeroed under `__syncthreads`
  before use and flushed under `__syncthreads` after, so no lane reads a slot
  another is still writing. The flush skips zero slots to avoid pointless
  global traffic.
- **Leader election within a partial warp**: `__match_any_sync` is called with
  the real `__activemask()`, not `0xffffffff`, so divergent/tail warps elect a
  leader only among *participating* lanes. Using the full mask would have made
  inactive lanes "vote" and corrupt the popcount.
- **Accepted limitation**: the privatised path assumes the launch-time shmem
  budget (`max_shmem_per_block`, default 48 KB conservative). On Hopper the
  opt-in carveout is larger; wiring `cudaFuncAttributeMaxDynamicSharedMemory`
  to raise the threshold is left as a follow-up (would widen the vocabulary
  range that qualifies for M089). The global M088 path is unaffected and
  remains the correct fallback.

## Test Results

CPU host-emulation (no GPU on this VM):

```
M088 warp-aggregation: 2000 randomized trials (partial warps, 70% hot-set
  collisions, vocab=64) — every histogram bit-identical to reference   ✓
M089 block-privatised flush: 4-block partials summed == reference        ✓
Structural: helper + kernel host-compile, link, and execute via CUDA
  stub harness (ARCH 900 path exercised)                                 ✓
```

GPU-side throughput impact (expected: large reduction in atomic-unit
serialisation on skewed batches) should be measured on the target
A6000×2 + H100×1 server via `scripts/run_exp1.sh`.
