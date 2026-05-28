#pragma once

#ifndef ALLOY_TIERED_CACHE_CUH
#define ALLOY_TIERED_CACHE_CUH

// Alloy Tiered Embedding Cache
//
// Extends HypeReca's two-level (GPU-pool / CPU-DRAM) embedding hierarchy
// into a four-level placement with per-tier native precision:
//
//   Tier 0  H100  HBM3   hot    FP8  (E4M3)
//   Tier 1  A6000 GDDR6  warm   BF16
//   Tier 2  CPU   DRAM   cold   FP32
//   Tier 3  SSD          arch.  FP32
//
// The key data structure is a per-row frequency histogram maintained
// entirely in device memory.  A dedicated histogram-only kernel runs
// first over the full batch of lookup indices (mirroring CUB's
// DeviceTopK histogram-pass separation), then a fused classify+migrate
// kernel partitions rows across tiers.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900
#include <cuda_fp8.h>
#define ALLOY_HAS_FP8 1
#else
#define ALLOY_HAS_FP8 0
#endif
#else
#define ALLOY_HAS_FP8 0
#endif

#include <cstdint>
#include <cstddef>

namespace Alloy {

// ─────────────────────────────────────────────────────────────────
//  Tier classification
// ─────────────────────────────────────────────────────────────────

enum class Tier : uint8_t {
    H100_HBM3   = 0,   // hot  — FP8
    A6000_GDDR6 = 1,   // warm — BF16
    CPU_DRAM    = 2,   // cold — FP32
    SSD_ARCHIVE = 3,   // archived
    NUM_TIERS   = 4
};

// Thresholds are expressed as EMA frequency values.
// classify_row() is evaluated per-row, per-batch.
struct TierThresholds {
    float hot;     // freq >= hot  → Tier 0 (H100)
    float warm;    // freq >= warm → Tier 1 (A6000)
    // everything else → Tier 2 or 3
};

__device__ __forceinline__
Tier classify_row(float freq, const TierThresholds& th)
{
    if (freq >= th.hot)  return Tier::H100_HBM3;
    if (freq >= th.warm) return Tier::A6000_GDDR6;
    return Tier::CPU_DRAM;
}

// ─────────────────────────────────────────────────────────────────
//  Pass 0 — histogram-only kernel
//
//  Counts how many times each embedding row is referenced in the
//  current batch.  The histogram is atomically accumulated into a
//  per-row frequency array that persists across batches (with EMA
//  decay applied once per training step).
//
//  This is a *separate* kernel from the subsequent filter-and-place
//  pass, following the same decomposition as CUB DeviceTopK:
//  isolating the histogram pass lets us pick an occupancy-optimal
//  grid size independently of the filter kernel.
//
//  ── Why the naive version was a bottleneck ──────────────────────
//  Embedding access in DLRM-style workloads is heavily power-law:
//  a handful of "hot" rows absorb a large fraction of every batch.
//  A plain `atomicAdd(d_freq_hist + row, 1)` therefore serialises
//  *every* lane that touched a hot row onto the *same* global
//  address — within a warp, across a block, and across the grid.
//  At a 32× collision rate on a hot row the atomic unit, not memory
//  bandwidth, sets the kernel's runtime.
//
//  The rewrite below removes that serialisation in two stages, in
//  the spirit of CUB's two-level (privatised → global) histogram:
//
//    M088  Warp-level aggregation.  Lanes in a warp that target the
//          *same* row are discovered with __match_any_sync; one
//          elected leader performs a single atomicAdd carrying the
//          whole group's count.  A hot row hit by all 32 lanes now
//          costs 1 atomic instead of 32.
//
//    M089  Block-level privatisation (opt-in fast path).  When the
//          vocabulary is small enough to fit a uint32 slot per row
//          in shared memory, each block accumulates into a private
//          shared histogram and flushes it to global once at the
//          end — converting B global atomics per row into 1 per
//          block that touched it.  Selected at launch via the
//          dynamic shared-memory size; falls back to the global
//          warp-aggregated path (M088) when it would not fit.
//
//    M090  Vectorised, grid-stride index loads.  The old kernel
//          processed exactly grid·block·IPT items and silently
//          dropped any batch tail beyond that.  The loop below is a
//          true grid-stride loop (correct for *any* grid size) and
//          loads four indices per iteration to hide load latency.
// ─────────────────────────────────────────────────────────────────

// ---- M088 helper: one warp-aggregated increment to a uint32 histogram ----
//
//  All *active* lanes call this with their own `slot`.  Lanes sharing
//  a slot are grouped via __match_any_sync; the lowest lane in each
//  group adds the group's population in a single atomic.  Works for a
//  partial warp (uses the real active mask) and for the degenerate
//  all-distinct case (every group has size 1 → identical to a plain
//  atomicAdd, no correctness penalty).
//
//  Pre-Volta architectures lack __match_any_sync; there we fall back
//  to a plain atomic so the kernel still compiles and runs correctly.
__device__ __forceinline__
void warp_aggregated_inc(uint32_t* __restrict__ hist, uint32_t slot)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const unsigned active = __activemask();
    // Mask of lanes (within `active`) whose slot equals ours.
    const unsigned peers  = __match_any_sync(active, slot);
    const int      group  = __popc(peers);                 // group size
    const int      lane   = threadIdx.x & 31;
    // Elect the lowest set lane in the peer group as the leader.
    const int      leader = __ffs(peers) - 1;
    if (lane == leader)
    {
        atomicAdd(hist + slot, static_cast<uint32_t>(group));
    }
#else
    atomicAdd(hist + slot, 1u);
#endif
}

template <int BLOCK_THREADS = 256, int ITEMS_PER_THREAD = 4>
__global__ void FrequencyHistogramKernel(
    const size_t* __restrict__ d_indices,    // [batch_size]
    uint32_t*     __restrict__ d_freq_hist,  // [num_embeddings] — global histogram
    size_t                     batch_size,
    size_t                     num_embeddings = 0)  // needed only for the M089 path
{
    // M089: dynamic shared memory.  Size > 0 ⇒ caller decided the whole
    // vocabulary fits and wants the block-privatised path.  Size == 0 ⇒
    // global warp-aggregated path only.
    extern __shared__ uint32_t s_hist[];
    const bool use_shared = (num_embeddings > 0);

    if (use_shared)
    {
        // Zero the private histogram cooperatively.
        for (size_t i = threadIdx.x; i < num_embeddings; i += BLOCK_THREADS)
            s_hist[i] = 0u;
        __syncthreads();
    }

    // M090: true grid-stride loop, four indices per step.
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK_THREADS * ITEMS_PER_THREAD;
    for (size_t base = (static_cast<size_t>(blockIdx.x) * BLOCK_THREADS * ITEMS_PER_THREAD)
                       + threadIdx.x;
         base < batch_size + stride;            // +stride so the tail iteration runs
         base += stride)
    {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const size_t pos = base + static_cast<size_t>(i) * BLOCK_THREADS;
            if (pos < batch_size)
            {
                const uint32_t slot = static_cast<uint32_t>(d_indices[pos]);
                if (use_shared)
                    warp_aggregated_inc(s_hist, slot);       // M088 into shared
                else
                    warp_aggregated_inc(d_freq_hist, slot);  // M088 into global
            }
        }
    }

    // M089: flush the block-private histogram to global once.
    if (use_shared)
    {
        __syncthreads();
        for (size_t i = threadIdx.x; i < num_embeddings; i += BLOCK_THREADS)
        {
            const uint32_t c = s_hist[i];
            if (c != 0u) atomicAdd(d_freq_hist + i, c);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  EMA decay kernel
//
//  Applied once per training step BEFORE the histogram kernel.
//  new_freq[i] = old_freq[i] * decay_factor
//
//  Grid-stride loop, vectorized 4-wide.
// ─────────────────────────────────────────────────────────────────

__global__ void FrequencyDecayKernel(
    uint32_t* __restrict__ d_freq_hist,
    float                  decay_factor,
    size_t                 num_embeddings)
{
    // We decay in-place using integer multiply-shift:
    //   new = (old * decay_num) >> decay_shift
    // For decay_factor ≈ 0.95 we use (243, 8) → 243/256 ≈ 0.9492
    const uint32_t decay_num   = static_cast<uint32_t>(decay_factor * 256.0f);
    const uint32_t decay_shift = 8u;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_embeddings;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        const uint32_t old_val = d_freq_hist[i];
        d_freq_hist[i] = (old_val * decay_num) >> decay_shift;
    }
}

// ─────────────────────────────────────────────────────────────────
//  Tier buffer resolution
//
//  A single lookup replacing repeated switch-case blocks.
//  The buffer table lives in constant memory per-launch.
// ─────────────────────────────────────────────────────────────────

template <typename DataType>
struct TierBufferTable {
    const DataType* __restrict__ bufs[3];  // indexed by Tier enum

    __device__ __forceinline__
    const DataType* operator[](Tier t) const { return bufs[static_cast<int>(t)]; }
};

// ─────────────────────────────────────────────────────────────────
//  process_range — generic strided element processor
//
//  Mirrors CUB's process_range: iterates over a tile of batch
//  indices and invokes a caller-supplied lambda on each (row, tid)
//  pair.  The lambda captures all strategy-specific logic; the
//  iteration skeleton is shared across strategies.
// ─────────────────────────────────────────────────────────────────

template <int BLOCK_THREADS, typename F>
__device__ __forceinline__
void process_range(
    const size_t* __restrict__ d_indices,
    size_t                     batch_size,
    size_t                     tile_offset,
    F&&                        f)
{
    for (size_t tid = tile_offset + threadIdx.x;
         tid < batch_size;
         tid += static_cast<size_t>(gridDim.x) * BLOCK_THREADS)
    {
        f(d_indices[tid], tid);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Migration double-buffer
//
//  Replaces raw pointer management with CUB's DoubleBuffer pattern.
//  Current() is the migration queue being filled; Alternate() holds
//  the previous step's queue being drained by ExecuteMigrationsKernel.
//  At each step boundary: selector ^= 1.
// ─────────────────────────────────────────────────────────────────

struct MigrationEntry {
    size_t row_idx;
    Tier   src_tier;
    Tier   dst_tier;
};

struct MigrationQueueDoubleBuffer {
    MigrationEntry* bufs[2];
    uint32_t*       counts[2];
    int             selector;

    __host__ __device__ MigrationEntry* Current()   { return bufs[selector]; }
    __host__ __device__ MigrationEntry* Alternate() { return bufs[selector ^ 1]; }
    __host__ __device__ uint32_t*       CurrentCount()   { return counts[selector]; }
    __host__ __device__ uint32_t*       AlternateCount() { return counts[selector ^ 1]; }
};

// ─────────────────────────────────────────────────────────────────
//  Pass 1 — fused classify-and-scatter kernel
//
//  Restructured following CUB DeviceTopK's three-lambda pattern:
//
//    f_in_place:       row is already at the correct tier.
//                      Serve from current buffer, no migration needed.
//                      (analogous to f_early_stop: the "done" path)
//
//    f_serve_enqueue:  row is at the wrong tier but migration queue
//                      has capacity.  Serve from current buffer AND
//                      enqueue a migration entry for async processing.
//                      (analogous to f_with_out_buf: filter + histogram)
//
//    f_serve_only:     row is at the wrong tier but migration queue
//                      is full (budget exhausted this step).  Serve
//                      from current buffer, skip migration enqueue.
//                      (analogous to f_no_out_buf: histogram only)
//
//  All three lambdas are passed to the same process_range template.
//  The appropriate lambda is selected based on runtime state, just
//  as CUB selects based on early_stop / out_buf / nullptr.
// ─────────────────────────────────────────────────────────────────

template <typename DataType, int BLOCK_THREADS = 256>
__global__ void ClassifyAndScatterKernel(
    const size_t*       __restrict__ d_indices,
    size_t                           batch_size,
    size_t                           embedding_dim,
    const uint32_t*     __restrict__ d_freq_hist,
    const uint8_t*      __restrict__ d_row_tier,
    TierThresholds                   thresholds,
    TierBufferTable<DataType>        tier_bufs,
    DataType*           __restrict__ d_output,
    MigrationEntry*     __restrict__ d_migration_queue,
    uint32_t*           __restrict__ d_migration_count,
    uint32_t                         migration_budget)
{
    const size_t tile_offset = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS;

    // Shared gather helper: copy one embedding row from src tier to output
    auto gather_row = [&](const size_t row, const size_t out_idx, const Tier current) {
        const DataType* __restrict__ src = tier_bufs[current];
        if (src == nullptr) return;
        const DataType* __restrict__ row_ptr = src + row * embedding_dim;
        DataType*       __restrict__ out_ptr = d_output + out_idx * embedding_dim;
        for (size_t d = 0; d < embedding_dim; ++d)
            out_ptr[d] = row_ptr[d];
    };

    // Lambda for row already at correct tier: just serve, no migration.
    // Analogous to CUB f_early_stop: the "we're done" fast path.
    auto f_in_place = [&](size_t row, size_t out_idx) {
        const Tier current = static_cast<Tier>(d_row_tier[row]);
        gather_row(row, out_idx, current);
    };

    // Lambda for row at wrong tier, migration queue has capacity.
    // Serve from current tier AND enqueue migration.
    // Analogous to CUB f_with_out_buf: filter + write candidates + histogram.
    auto f_serve_enqueue = [&](size_t row, size_t out_idx) {
        const float freq    = static_cast<float>(d_freq_hist[row]);
        const Tier  target  = classify_row(freq, thresholds);
        const Tier  current = static_cast<Tier>(d_row_tier[row]);
        gather_row(row, out_idx, current);
        if (target != current)
        {
            const uint32_t slot = atomicAdd(d_migration_count, 1u);
            d_migration_queue[slot] = {row, current, target};
        }
    };

    // Lambda for row at wrong tier, migration budget exhausted.
    // Serve from current tier, skip enqueue.
    // Analogous to CUB f_no_out_buf: histogram only, no output write.
    auto f_serve_only = [&](size_t row, size_t out_idx) {
        const Tier current = static_cast<Tier>(d_row_tier[row]);
        gather_row(row, out_idx, current);
    };

    // Choose and invoke the appropriate lambda.
    // Read current migration count to decide if we have budget.
    // This mirrors CUB's if(early_stop) / else if(out_buf) / else dispatch.
    if (migration_budget == 0)
    {
        // No migrations allowed this step — fast path
        process_range<BLOCK_THREADS>(d_indices, batch_size, tile_offset, f_in_place);
    }
    else
    {
        // Check if migration queue still has capacity
        const uint32_t current_count = *d_migration_count;
        if (current_count < migration_budget)
        {
            process_range<BLOCK_THREADS>(d_indices, batch_size, tile_offset, f_serve_enqueue);
        }
        else
        {
            process_range<BLOCK_THREADS>(d_indices, batch_size, tile_offset, f_serve_only);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Migration execution kernel
//
//  Processes the migration queue built by ClassifyAndScatterKernel.
//  Uses TierBufferTable for single-lookup buffer resolution
//  (replacing the two switch-case blocks in the previous version).
//
//  The queue comes from MigrationQueueDoubleBuffer.Alternate():
//  the previous step's enqueued entries, which can now be drained
//  while the current step fills Current() — the DoubleBuffer
//  pattern from CUB ensures no race between producer and consumer.
//
//  Precision conversion rules:
//    Tier 0 (H100)  ↔ FP8  E4M3
//    Tier 1 (A6000) ↔ BF16
//    Tier 2 (CPU)   ↔ FP32
// ─────────────────────────────────────────────────────────────────

template <typename SrcType, typename DstType>
__device__ __forceinline__
DstType precision_cast(SrcType val)
{
    return static_cast<DstType>(static_cast<float>(val));
}

template <typename DataType>
struct MutableTierBufferTable {
    DataType* __restrict__ bufs[3];

    __device__ __forceinline__
    DataType* operator[](Tier t) const { return bufs[static_cast<int>(t)]; }
};

template <typename DataType, int BLOCK_THREADS = 128>
__global__ void ExecuteMigrationsKernel(
    const MigrationEntry* __restrict__ d_queue,
    uint32_t                           queue_len,
    size_t                             embedding_dim,
    MutableTierBufferTable<DataType>   tier_bufs,
    uint8_t*              __restrict__ d_row_tier)
{
    const uint32_t tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (tid >= queue_len) return;

    const MigrationEntry entry = d_queue[tid];
    const size_t base = entry.row_idx * embedding_dim;

    // Single-lookup buffer resolution via table (no switch-case)
    DataType* __restrict__ src = tier_bufs[entry.src_tier];
    DataType* __restrict__ dst = tier_bufs[entry.dst_tier];

    if (src == nullptr || dst == nullptr) return;

    for (size_t d = 0; d < embedding_dim; ++d)
    {
        dst[base + d] = src[base + d];
    }

    d_row_tier[entry.row_idx] = static_cast<uint8_t>(entry.dst_tier);
}

// ─────────────────────────────────────────────────────────────────
//  Tier capacity tracking
// ─────────────────────────────────────────────────────────────────

struct TierCapacity {
    size_t capacity_rows;       // max rows this tier can hold
    size_t occupied_rows;       // currently occupied
    size_t bytes_per_row;       // embedding_dim × sizeof(tier_dtype)
    float  pcie_bandwidth_gbps; // measured PCIe bandwidth

    __host__ __device__
    float utilization() const {
        return (capacity_rows > 0)
            ? static_cast<float>(occupied_rows) / static_cast<float>(capacity_rows)
            : 0.0f;
    }

    __host__ __device__
    float migration_latency_ms(size_t num_rows) const {
        const float bytes = static_cast<float>(num_rows * bytes_per_row);
        return (pcie_bandwidth_gbps > 0.0f)
            ? bytes / (pcie_bandwidth_gbps * 1e6f)  // GB/s → bytes/ms
            : 0.0f;
    }
};

// ─────────────────────────────────────────────────────────────────
//  Kernel launch helpers
// ─────────────────────────────────────────────────────────────────

struct TieredCacheLaunchConfig {
    int histogram_grid_size;
    int histogram_block_size;
    int classify_grid_size;
    int classify_block_size;
    int migrate_grid_size;
    int migrate_block_size;

    // M089: shared-memory histogram path. histogram_shmem_bytes > 0 means
    // the whole vocabulary fits in one block's shared memory, so the
    // privatised path should be launched with this dynamic shmem size and
    // num_embeddings passed to the kernel. 0 means use the global
    // warp-aggregated path (kernel's num_embeddings arg stays 0).
    size_t histogram_shmem_bytes;
    bool   histogram_use_shared;

    static TieredCacheLaunchConfig compute(
        size_t batch_size,
        size_t num_embeddings,
        int    sm_count,
        size_t max_shmem_per_block = 48u * 1024u)  // conservative default (48 KB)
    {
        constexpr int HIST_BLOCK  = 256;
        constexpr int HIST_IPT    = 4;
        constexpr int CLASS_BLOCK = 256;
        constexpr int MIG_BLOCK   = 128;

        TieredCacheLaunchConfig cfg;
        cfg.histogram_block_size = HIST_BLOCK;
        cfg.histogram_grid_size  = static_cast<int>(
            (batch_size + HIST_BLOCK * HIST_IPT - 1) / (HIST_BLOCK * HIST_IPT));
        // Cap at 4× SM occupancy
        if (cfg.histogram_grid_size > sm_count * 4)
            cfg.histogram_grid_size = sm_count * 4;
        if (cfg.histogram_grid_size < 1)
            cfg.histogram_grid_size = 1;

        // M089: pick the privatised path only when a uint32 slot per row
        // fits in shared memory. Above that we'd thrash, so use the global
        // warp-aggregated path (M088) instead.
        const size_t needed = num_embeddings * sizeof(uint32_t);
        if (num_embeddings > 0 && needed <= max_shmem_per_block)
        {
            cfg.histogram_use_shared  = true;
            cfg.histogram_shmem_bytes = needed;
            // With privatisation, more blocks only add flush traffic; cap
            // the grid so each block still sees a meaningful slice of input.
            const int by_occupancy = sm_count * 2;
            if (by_occupancy > 0 && cfg.histogram_grid_size > by_occupancy)
                cfg.histogram_grid_size = by_occupancy;
        }
        else
        {
            cfg.histogram_use_shared  = false;
            cfg.histogram_shmem_bytes = 0;
        }

        cfg.classify_block_size = CLASS_BLOCK;
        cfg.classify_grid_size  = static_cast<int>(
            (batch_size + CLASS_BLOCK - 1) / CLASS_BLOCK);

        cfg.migrate_block_size = MIG_BLOCK;
        cfg.migrate_grid_size  = 1;  // set after queue_len is known

        return cfg;
    }
};

}  // namespace Alloy

#endif  // ALLOY_TIERED_CACHE_CUH
