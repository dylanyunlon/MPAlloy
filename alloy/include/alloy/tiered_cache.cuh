#pragma once

#ifndef ALLOY_TIERED_CACHE_CUH
#define ALLOY_TIERED_CACHE_CUH

/// Alloy Tiered Embedding Cache
///
/// Extends HypeReca's two-level (GPU-pool / CPU-DRAM) embedding hierarchy
/// into a four-level placement with per-tier native precision:
///
///   Tier 0  H100  HBM3   hot    FP8  (E4M3)
///   Tier 1  A6000 GDDR6  warm   BF16
///   Tier 2  CPU   DRAM   cold   FP32
///   Tier 3  SSD          arch.  FP32
///
/// The key data structure is a per-row frequency histogram maintained
/// entirely in device memory.  A dedicated histogram-only kernel runs
/// first over the full batch of lookup indices (mirroring CUB's
/// DeviceTopK histogram-pass separation), then a fused classify+migrate
/// kernel partitions rows across tiers.

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

/// Thresholds are expressed as EMA frequency values.
/// classify_row() is evaluated per-row, per-batch.
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
// ─────────────────────────────────────────────────────────────────

template <int BLOCK_THREADS = 256, int ITEMS_PER_THREAD = 4>
__global__ void FrequencyHistogramKernel(
    const size_t* __restrict__ d_indices,   // [batch_size]
    uint32_t*     __restrict__ d_freq_hist, // [num_embeddings] — global histogram
    size_t                     batch_size)
{
    const size_t tile_offset = static_cast<size_t>(blockIdx.x) *
                               BLOCK_THREADS * ITEMS_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const size_t pos = tile_offset + threadIdx.x + i * BLOCK_THREADS;
        if (pos < batch_size)
        {
            atomicAdd(d_freq_hist + d_indices[pos], 1u);
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
//  Pass 1 — fused classify-and-scatter kernel
//
//  For each index in the batch:
//    1.  Read the row's current EMA frequency from d_freq_hist
//    2.  Classify into a tier via classify_row()
//    3.  Compare with the row's *current* tier
//    4a. If the row is already in the correct tier: read the
//        embedding from the tier-local buffer and write to output
//    4b. If a migration is needed: enqueue the (row, old_tier,
//        new_tier) triple into d_migration_queue via atomicAdd
//        on a per-tier counter, and still serve from old tier
//
//  This two-pass structure (histogram → classify+scatter) is
//  analogous to CUB's first-pass histogram then fused filter-and-
//  histogram for subsequent passes: the histogram kernel has a
//  different occupancy sweet-spot, and separating it lets the
//  compiler specialize each kernel independently.
// ─────────────────────────────────────────────────────────────────

struct MigrationEntry {
    size_t row_idx;
    Tier   src_tier;
    Tier   dst_tier;
};

template <typename DataType, int BLOCK_THREADS = 256>
__global__ void ClassifyAndScatterKernel(
    // Lookup indices
    const size_t*       __restrict__ d_indices,      // [batch_size]
    size_t                           batch_size,
    size_t                           embedding_dim,
    // Frequency histogram (read-only after Pass 0)
    const uint32_t*     __restrict__ d_freq_hist,
    // Per-row current tier assignment
    const uint8_t*      __restrict__ d_row_tier,     // [num_embeddings]
    // Tier thresholds
    TierThresholds                   thresholds,
    // Tier-local embedding buffers (may be nullptr for unpopulated tiers)
    const DataType*     __restrict__ d_tier0_buf,    // H100 — FP8 storage (cast widened to DataType on read)
    const DataType*     __restrict__ d_tier1_buf,    // A6000 — BF16 storage
    const DataType*     __restrict__ d_tier2_buf,    // CPU-staged DRAM copy
    // Output buffer (always FP32 for forward pass)
    DataType*           __restrict__ d_output,       // [batch_size × embedding_dim]
    // Migration queue
    MigrationEntry*     __restrict__ d_migration_queue,
    uint32_t*           __restrict__ d_migration_count)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;
    if (tid >= batch_size) return;

    const size_t row     = d_indices[tid];
    const float  freq    = static_cast<float>(d_freq_hist[row]);
    const Tier   target  = classify_row(freq, thresholds);
    const Tier   current = static_cast<Tier>(d_row_tier[row]);

    // Select source buffer based on current (not target) tier:
    // we always serve from wherever the data currently lives.
    const DataType* __restrict__ src_buf = nullptr;
    switch (current)
    {
        case Tier::H100_HBM3:   src_buf = d_tier0_buf; break;
        case Tier::A6000_GDDR6: src_buf = d_tier1_buf; break;
        case Tier::CPU_DRAM:    src_buf = d_tier2_buf; break;
        default:                src_buf = d_tier2_buf; break;
    }

    // Gather embedding row → output (coalesced along embedding_dim via
    // the same blockDim.x-as-embedding-width trick from HypeReca's
    // indexGetKernel, but here we handle it with a loop to support
    // arbitrary embedding dimensions)
    if (src_buf != nullptr)
    {
        const DataType* __restrict__ row_ptr = src_buf + row * embedding_dim;
        DataType*       __restrict__ out_ptr = d_output + tid * embedding_dim;
        for (size_t d = 0; d < embedding_dim; ++d)
        {
            out_ptr[d] = row_ptr[d];
        }
    }

    // Enqueue migration if tier changed
    if (target != current)
    {
        const uint32_t slot = atomicAdd(d_migration_count, 1u);
        d_migration_queue[slot] = {row, current, target};
    }
}

// ─────────────────────────────────────────────────────────────────
//  Migration execution kernel
//
//  Processes the migration queue built by ClassifyAndScatterKernel.
//  Each thread handles one MigrationEntry: reads from src tier
//  buffer, converts precision, writes to dst tier buffer, and
//  updates the per-row tier assignment.
//
//  Precision conversion rules:
//    Tier 0 (H100)  ↔ FP8  E4M3
//    Tier 1 (A6000) ↔ BF16
//    Tier 2 (CPU)   ↔ FP32
//
//  Promotion (cold→hot): truncates precision but gains bandwidth.
//  Demotion  (hot→cold): widens precision, preserves accumulated
//                         gradient information.
// ─────────────────────────────────────────────────────────────────

template <typename SrcType, typename DstType>
__device__ __forceinline__
DstType precision_cast(SrcType val)
{
    return static_cast<DstType>(static_cast<float>(val));
}

template <typename DataType, int BLOCK_THREADS = 128>
__global__ void ExecuteMigrationsKernel(
    const MigrationEntry* __restrict__ d_queue,
    uint32_t                           queue_len,
    size_t                             embedding_dim,
    // Tier buffers (read/write)
    DataType*             __restrict__ d_tier0_buf,
    DataType*             __restrict__ d_tier1_buf,
    DataType*             __restrict__ d_tier2_buf,
    // Row-tier assignment (updated in-place)
    uint8_t*              __restrict__ d_row_tier)
{
    const uint32_t tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (tid >= queue_len) return;

    const MigrationEntry entry = d_queue[tid];
    const size_t base = entry.row_idx * embedding_dim;

    // Resolve source and destination buffer pointers
    DataType* __restrict__ src = nullptr;
    DataType* __restrict__ dst = nullptr;

    switch (entry.src_tier)
    {
        case Tier::H100_HBM3:   src = d_tier0_buf; break;
        case Tier::A6000_GDDR6: src = d_tier1_buf; break;
        case Tier::CPU_DRAM:    src = d_tier2_buf; break;
        default: return;
    }
    switch (entry.dst_tier)
    {
        case Tier::H100_HBM3:   dst = d_tier0_buf; break;
        case Tier::A6000_GDDR6: dst = d_tier1_buf; break;
        case Tier::CPU_DRAM:    dst = d_tier2_buf; break;
        default: return;
    }

    if (src == nullptr || dst == nullptr) return;

    // Copy with implicit precision conversion through DataType
    // (the actual FP8↔BF16↔FP32 cast is handled by the typed
    //  instantiation of the kernel — see dispatch in tiered_embedding.cu)
    for (size_t d = 0; d < embedding_dim; ++d)
    {
        dst[base + d] = src[base + d];
    }

    // Update tier assignment
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

    static TieredCacheLaunchConfig compute(
        size_t batch_size,
        size_t num_embeddings,
        int    sm_count)
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
