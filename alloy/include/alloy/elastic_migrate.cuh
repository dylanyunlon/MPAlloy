#pragma once

#ifndef ALLOY_ELASTIC_MIGRATE_CUH
#define ALLOY_ELASTIC_MIGRATE_CUH

// Alloy Elastic Migration Engine
//
// Adapts mTuner's four-operation model to the tiered embedding cache:
//
//   gather     pull hot rows from lower tier to higher tier
//   discard    evict cold rows from higher tier to lower tier
//   execute    compute on rows at their current location
//   checkpoint save dirty rows to CPU DRAM for fault tolerance
//
// The key difference from mTuner's original design (which manages
// full tensors) is that Alloy operates on *row ranges* within a
// shared embedding table.  A single table can have hot rows on
// H100, warm rows on A6000, and cold rows on CPU simultaneously.
//
// Migration is PCIe-aware: gather/discard operations are batched
// into PCIe-Gen-sized transfers to amortise per-transfer overhead.
// The DoubleBuffer pattern (from CUB) is used to overlap migration
// of batch N with training on batch N-1.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace Alloy {

// ─────────────────────────────────────────────────────────────────
//  Row range descriptor
// ─────────────────────────────────────────────────────────────────

struct RowRange {
    size_t  table_id;
    size_t  row_start;
    size_t  row_count;
    uint8_t current_tier;  // Tier enum value
    uint8_t target_tier;
    bool    dirty;         // modified since last checkpoint
};

// ─────────────────────────────────────────────────────────────────
//  DoubleBuffer for migration pipelining
//
//  While rows in buffer[selector] are being migrated over PCIe,
//  newly scheduled migrations accumulate in buffer[selector ^ 1].
//  At sync points the selector flips (exactly as CUB DoubleBuffer).
// ─────────────────────────────────────────────────────────────────

template <typename T>
struct MigrationDoubleBuffer {
    T*  buffers[2];
    int selector;

    __host__ __device__ T* Current()   { return buffers[selector]; }
    __host__ __device__ T* Alternate() { return buffers[selector ^ 1]; }

    void flip() { selector ^= 1; }

    static MigrationDoubleBuffer alloc(size_t num_elements)
    {
        MigrationDoubleBuffer db;
        db.selector = 0;
        cudaMalloc(&db.buffers[0], num_elements * sizeof(T));
        cudaMalloc(&db.buffers[1], num_elements * sizeof(T));
        return db;
    }

    void free()
    {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
    }
};

// ─────────────────────────────────────────────────────────────────
//  Gather kernel — tier promotion (cold → hot)
//
//  Reads rows from a lower-tier buffer, casts to the higher tier's
//  native precision, and writes to the higher-tier buffer.
//
//  The kernel uses the same (blockDim.x = embedding_dim) pattern as
//  HypeReca's indexCopyKernel for coalesced access, but adds:
//    1. Precision conversion during the copy
//    2. Staging through the DoubleBuffer to overlap with PCIe
//
//  Template parameter NarrowType is the target tier's storage type
//  (e.g., __nv_bfloat16 for A6000, or fp8 for H100).
// ─────────────────────────────────────────────────────────────────

template <typename WideType, typename NarrowType>
__global__ void GatherPromoteKernel(
    const WideType*   __restrict__ d_src,       // lower-tier buffer (wider precision)
    NarrowType*       __restrict__ d_dst,       // higher-tier buffer (narrower precision)
    const size_t*     __restrict__ d_src_rows,  // row indices in source
    const size_t*     __restrict__ d_dst_rows,  // row indices in destination
    size_t                         num_rows,
    size_t                         embedding_dim)
{
    // Grid: one block per row, blockDim.x = embedding_dim (capped at 1024)
    // For embedding_dim > 1024 we loop over chunks.
    const size_t row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const size_t src_row   = d_src_rows[row_idx];
    const size_t dst_row   = d_dst_rows[row_idx];
    const size_t src_base  = src_row * embedding_dim;
    const size_t dst_base  = dst_row * embedding_dim;

    for (size_t d = threadIdx.x; d < embedding_dim; d += blockDim.x)
    {
        // Widen → float → narrow (two-step cast for precision control)
        const float val = static_cast<float>(d_src[src_base + d]);
        d_dst[dst_base + d] = static_cast<NarrowType>(val);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Discard kernel — tier demotion (hot → cold)
//
//  The reverse of gather: reads from higher tier in narrow precision,
//  widens to the lower tier's precision, writes out.
//
//  Demotion *widens* precision (FP8 → BF16 → FP32), preserving all
//  information accumulated in the narrow format.  This is lossless
//  from the narrow type's perspective.
// ─────────────────────────────────────────────────────────────────

template <typename NarrowType, typename WideType>
__global__ void DiscardDemoteKernel(
    const NarrowType* __restrict__ d_src,       // higher-tier buffer (narrow)
    WideType*         __restrict__ d_dst,       // lower-tier buffer (wide)
    const size_t*     __restrict__ d_src_rows,
    const size_t*     __restrict__ d_dst_rows,
    size_t                         num_rows,
    size_t                         embedding_dim)
{
    const size_t row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const size_t src_base = d_src_rows[row_idx] * embedding_dim;
    const size_t dst_base = d_dst_rows[row_idx] * embedding_dim;

    for (size_t d = threadIdx.x; d < embedding_dim; d += blockDim.x)
    {
        // Narrow → float → wide (lossless from narrow's perspective)
        const float val = static_cast<float>(d_src[src_base + d]);
        d_dst[dst_base + d] = static_cast<WideType>(val);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Execute kernel — in-place gradient application at current tier
//
//  Applies optimizer step to rows at whatever tier they currently
//  reside on.  This replaces HypeReca's single-device SGDKernel
//  with a tier-aware version that marks rows dirty for checkpointing.
//
//  The grid-stride pattern matches SGDKernel in sgd.cu, but with
//  an added dirty-bit write.
// ─────────────────────────────────────────────────────────────────

template <typename DataType, int STRIDE = 128>
__global__ void TieredSGDKernel(
    DataType*         __restrict__ d_param,       // embedding rows at current tier
    const DataType*   __restrict__ d_grad,        // gradient buffer
    DataType                       lr,            // learning rate (in tier-native type)
    const size_t*     __restrict__ d_active_rows, // which rows to update
    size_t                         num_active,
    size_t                         embedding_dim,
    uint8_t*          __restrict__ d_dirty_flags)  // per-row dirty bit
{
    size_t active_idx = static_cast<size_t>(blockIdx.x) * blockDim.x * STRIDE
                      + threadIdx.x;

    for (; active_idx < num_active;
         active_idx += static_cast<size_t>(gridDim.x) * blockDim.x * STRIDE)
    {
        for (size_t s = 0; s < STRIDE && (active_idx + s * blockDim.x) < num_active; ++s)
        {
            const size_t ai  = active_idx + s * blockDim.x;
            const size_t row = d_active_rows[ai];
            const size_t base = row * embedding_dim;

            // SGD update: param -= grad * lr
            for (size_t d = 0; d < embedding_dim; ++d)
            {
                d_param[base + d] -= d_grad[base + d] * lr;
            }

            // Mark dirty for checkpoint
            d_dirty_flags[row] = 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Checkpoint kernel — snapshot dirty rows to CPU staging buffer
//
//  Scans the dirty-flag array, gathers dirty rows into a compact
//  staging buffer, then the host copies the staging buffer to CPU
//  DRAM via cudaMemcpyAsync.
//
//  Two-phase design:
//    Phase 1: count dirty rows (parallel prefix sum → atomicAdd)
//    Phase 2: compact copy dirty rows to staging buffer
//
//  This avoids copying the entire tier to CPU every step.
// ─────────────────────────────────────────────────────────────────

template <typename DataType>
__global__ void CheckpointCompactKernel(
    const DataType*   __restrict__ d_tier_buf,      // tier embedding buffer
    const uint8_t*    __restrict__ d_dirty_flags,   // [num_rows]
    DataType*         __restrict__ d_staging_buf,    // compact output
    size_t*           __restrict__ d_staging_rows,   // which rows are in staging
    uint32_t*         __restrict__ d_staging_count,  // atomic counter
    size_t                         num_rows,
    size_t                         embedding_dim)
{
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < num_rows;
         row += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        if (d_dirty_flags[row])
        {
            const uint32_t slot = atomicAdd(d_staging_count, 1u);
            d_staging_rows[slot] = row;

            // Copy row to staging
            const size_t src_base = row * embedding_dim;
            const size_t dst_base = slot * embedding_dim;
            for (size_t d = 0; d < embedding_dim; ++d)
            {
                d_staging_buf[dst_base + d] = d_tier_buf[src_base + d];
            }
        }
    }
}

// Reset dirty flags after checkpoint
__global__ void ClearDirtyFlagsKernel(
    uint8_t* __restrict__ d_dirty_flags,
    size_t                num_rows)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_rows;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        d_dirty_flags[i] = 0;
    }
}

// ─────────────────────────────────────────────────────────────────
//  Migration batch scheduler
//
//  Groups pending migrations by (src_tier, dst_tier) and caps each
//  batch to fit within the PCIe transfer budget for that tier pair.
//
//  PCIe Gen4 x16: ~25 GB/s usable → ~25 MB per millisecond
//  PCIe Gen5 x16: ~50 GB/s usable → ~50 MB per millisecond
//
//  Given a per-step latency budget (e.g., 2ms), we compute:
//    max_rows = budget_ms × pcie_bandwidth / bytes_per_row
// ─────────────────────────────────────────────────────────────────

struct MigrationBatchConfig {
    float  budget_ms;         // max latency per step for migration
    float  pcie_gen4_gbps;    // measured Gen4 bandwidth
    float  pcie_gen5_gbps;    // measured Gen5 bandwidth
    size_t embedding_dim;
    size_t bytes_per_element; // sizeof(DataType)

    size_t max_rows_gen4() const
    {
        const float bytes_per_row = static_cast<float>(embedding_dim * bytes_per_element);
        return static_cast<size_t>(budget_ms * pcie_gen4_gbps * 1e6f / bytes_per_row);
    }

    size_t max_rows_gen5() const
    {
        const float bytes_per_row = static_cast<float>(embedding_dim * bytes_per_element);
        return static_cast<size_t>(budget_ms * pcie_gen5_gbps * 1e6f / bytes_per_row);
    }
};

}  // namespace Alloy

#endif  // ALLOY_ELASTIC_MIGRATE_CUH
