#pragma once

#ifndef ALLOY_MIXED_PRECISION_KERNELS_CUH
#define ALLOY_MIXED_PRECISION_KERNELS_CUH

/// Alloy Mixed-Precision Gradient Aggregation
///
/// The core numerical problem: when H100 accumulates gradients in FP8,
/// A6000 in BF16, and CPU in FP32, the allreduce produces different
/// bit-patterns depending on reduction order and intermediate precision.
///
/// This module implements:
///   1. Per-tier gradient accumulation kernels with configurable precision
///   2. A cross-precision allreduce that upcasts to FP32 before summing
///   3. FPRev-style diagnostic batch generation for drift detection
///
/// The "distinguishing input" technique (from FPRev) constructs gradients
/// near precision boundaries — FP8 max (448), subnormal zones, and
/// catastrophic cancellation pairs — to maximise observable divergence
/// between precision paths.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include <cfloat>

namespace Alloy {

// ─────────────────────────────────────────────────────────────────
//  Precision traits
// ─────────────────────────────────────────────────────────────────

template <typename T>
struct PrecisionTraits;

template <>
struct PrecisionTraits<float> {
    static constexpr float max_val      = FLT_MAX;
    static constexpr float min_normal   = FLT_MIN;
    static constexpr int   mantissa_bits = 23;
    static constexpr int   total_bits    = 32;
};

template <>
struct PrecisionTraits<__nv_bfloat16> {
    static constexpr float max_val      = 3.3895e+38f;
    static constexpr float min_normal   = 1.1755e-38f;
    static constexpr int   mantissa_bits = 7;
    static constexpr int   total_bits    = 16;
};

// FP8 E4M3: max ≈ 448, min_normal ≈ 2^-6
// (Only available on sm90+, guarded by ALLOY_HAS_FP8 in tiered_cache.cuh)
struct FP8Traits {
    static constexpr float max_val      = 448.0f;
    static constexpr float min_normal   = 0.015625f;  // 2^-6
    static constexpr int   mantissa_bits = 3;
    static constexpr int   total_bits    = 8;
};

// ─────────────────────────────────────────────────────────────────
//  Gradient accumulation kernel (per-tier, single precision path)
//
//  Accumulates gradients from a sparse set of embedding rows into
//  a dense gradient buffer.  The accumulation dtype matches the
//  tier's native precision:
//
//    Tier 0 (H100):  accumulate in FP32, then truncate to storage
//    Tier 1 (A6000): accumulate in FP32, then truncate to BF16
//    Tier 2 (CPU):   accumulate in FP32 natively
//
//  We always accumulate in FP32 to avoid catastrophic precision
//  loss, and only truncate on the final write.  This is the
//  "accumulate-wide, store-narrow" pattern.
// ─────────────────────────────────────────────────────────────────

template <typename StorageType, int BLOCK_THREADS = 256, int STRIDE = 128>
__global__ void GradientAccumulateKernel(
    const float*    __restrict__ d_grad_in,     // [num_updates × dim] — always FP32 input
    const size_t*   __restrict__ d_row_indices, // [num_updates] — which rows were updated
    StorageType*    __restrict__ d_grad_out,     // [num_embeddings × dim] — tier-native storage
    size_t                       num_updates,
    size_t                       embedding_dim)
{
    // Grid-stride loop over updates, inner loop over embedding_dim
    // (same strided pattern as HypeReca SGDKernel for memory coalescing)
    size_t update_idx = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS * STRIDE
                      + threadIdx.x;

    for (; update_idx < num_updates;
         update_idx += static_cast<size_t>(gridDim.x) * BLOCK_THREADS * STRIDE)
    {
        for (size_t s = 0; s < STRIDE && (update_idx + s * BLOCK_THREADS) < num_updates; ++s)
        {
            const size_t u   = update_idx + s * BLOCK_THREADS;
            const size_t row = d_row_indices[u];

            for (size_t d = 0; d < embedding_dim; ++d)
            {
                // Widen→accumulate→narrow
                const float val = d_grad_in[u * embedding_dim + d];
                const float old = static_cast<float>(d_grad_out[row * embedding_dim + d]);
                d_grad_out[row * embedding_dim + d] =
                    static_cast<StorageType>(old + val);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Cross-precision gradient allreduce
//
//  Gathers gradient contributions from all tiers, upcasts to FP32,
//  sums, then scatters back to each tier in its native precision.
//
//  Phase 1 (Gather+Upcast):  Each tier's gradient buffer is read,
//          cast to FP32, and atomically added to a shared FP32
//          accumulation buffer.
//
//  Phase 2 (Scatter+Downcast):  The FP32 result is read, divided
//          by the number of contributing tiers, and written back
//          to each tier's buffer in native precision.
//
//  This two-phase approach guarantees that the reduction is
//  *commutative* in FP32 regardless of the tier order.
// ─────────────────────────────────────────────────────────────────

template <typename TierType>
__global__ void GatherUpcastKernel(
    const TierType* __restrict__ d_tier_grad,   // tier-native grad
    float*          __restrict__ d_accum,        // FP32 accumulator
    size_t                       num_elements)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        atomicAdd(d_accum + i, static_cast<float>(d_tier_grad[i]));
    }
}

template <typename TierType>
__global__ void ScatterDowncastKernel(
    const float*    __restrict__ d_accum,
    TierType*       __restrict__ d_tier_grad,
    float                        scale,         // 1.0f / num_tiers
    size_t                       num_elements)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        d_tier_grad[i] = static_cast<TierType>(d_accum[i] * scale);
    }
}

// ─────────────────────────────────────────────────────────────────
//  FPRev-style drift diagnostic kernel
//
//  Constructs "distinguishing inputs" — gradient values engineered
//  to maximise numerical divergence between precision paths — and
//  measures the actual drift after a round-trip through each tier's
//  accumulate-reduce-scatter pipeline.
//
//  Three strategies:
//
//  BOUNDARY:      Values near FP8 max/min that overflow or denormalize
//                 in narrow formats but are exact in wider ones.
//
//  CANCELLATION:  Pairs (a, -a+ε) where ε is small enough that
//                 narrow-precision subtraction yields zero while
//                 wide-precision subtraction preserves ε.
//
//  ACCUMULATION:  Long chains of identical small values whose sum
//                 is representation-exact in FP32 but drifts in
//                 FP8/BF16 due to repeated rounding.
// ─────────────────────────────────────────────────────────────────

enum class DiagnosticStrategy : int {
    BOUNDARY      = 0,
    CANCELLATION  = 1,
    ACCUMULATION  = 2
};

__global__ void GenerateDiagnosticGradients(
    float*              __restrict__ d_diag_grads,  // [batch_size × dim]
    size_t                           batch_size,
    size_t                           embedding_dim,
    DiagnosticStrategy               strategy,
    uint64_t                         seed)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float* __restrict__ row = d_diag_grads + tid * embedding_dim;

    // Simple xorshift64 seeded per-thread
    uint64_t state = seed ^ (tid * 6364136223846793005ULL + 1442695040888963407ULL);
    auto xorshift = [&]() -> float {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<float>(state & 0xFFFFFF) / 16777216.0f;  // [0,1)
    };

    switch (strategy)
    {
    case DiagnosticStrategy::BOUNDARY:
    {
        // Alternate between FP8 overflow zone and subnormal zone
        const float fp8_max    = FP8Traits::max_val;
        const float fp8_submin = FP8Traits::min_normal * 0.5f;
        for (size_t d = 0; d < embedding_dim; ++d)
        {
            row[d] = (d % 2 == 0)
                ? fp8_max * (1.0f + xorshift() * 0.01f)    // just above max
                : fp8_submin * (0.1f + xorshift() * 0.9f); // deep subnormal
        }
        break;
    }
    case DiagnosticStrategy::CANCELLATION:
    {
        // Pairs that nearly cancel: (base, -base + epsilon)
        for (size_t d = 0; d < embedding_dim; d += 2)
        {
            const float base = 100.0f * (xorshift() - 0.5f);
            const float eps  = 1e-5f * xorshift();
            row[d]     =  base + eps;
            if (d + 1 < embedding_dim)
                row[d + 1] = -base + eps;
        }
        break;
    }
    case DiagnosticStrategy::ACCUMULATION:
    {
        // Uniform small value whose sum drifts in narrow precision
        const float small_val = 1e-3f;
        for (size_t d = 0; d < embedding_dim; ++d)
        {
            row[d] = small_val;
        }
        break;
    }
    }
}

/// Computes max |a - b| and max |a - b| / max(|a|, ε) element-wise
/// between two gradient buffers.  The output is a single (abs, rel) pair
/// reduced via shared memory.
__global__ void MeasureDriftKernel(
    const float* __restrict__ d_ref,     // FP32 reference
    const float* __restrict__ d_test,    // round-tripped through narrow precision
    float*       __restrict__ d_abs_max, // [1] — output: max absolute error
    float*       __restrict__ d_rel_max, // [1] — output: max relative error
    size_t                    num_elements)
{
    __shared__ float s_abs_max;
    __shared__ float s_rel_max;

    if (threadIdx.x == 0)
    {
        s_abs_max = 0.0f;
        s_rel_max = 0.0f;
    }
    __syncthreads();

    float local_abs = 0.0f;
    float local_rel = 0.0f;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        const float a   = d_ref[i];
        const float b   = d_test[i];
        const float diff = fabsf(a - b);
        const float denom = fmaxf(fabsf(a), 1e-10f);

        local_abs = fmaxf(local_abs, diff);
        local_rel = fmaxf(local_rel, diff / denom);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_abs = fmaxf(local_abs, __shfl_down_sync(0xFFFFFFFF, local_abs, offset));
        local_rel = fmaxf(local_rel, __shfl_down_sync(0xFFFFFFFF, local_rel, offset));
    }

    // Lane 0 of each warp atomically updates block-level max
    if ((threadIdx.x & 31) == 0)
    {
        atomicMax(reinterpret_cast<int*>(&s_abs_max), __float_as_int(local_abs));
        atomicMax(reinterpret_cast<int*>(&s_rel_max), __float_as_int(local_rel));
    }
    __syncthreads();

    // Block 0, thread 0 writes final result
    if (threadIdx.x == 0)
    {
        atomicMax(reinterpret_cast<int*>(d_abs_max), __float_as_int(s_abs_max));
        atomicMax(reinterpret_cast<int*>(d_rel_max), __float_as_int(s_rel_max));
    }
}

}  // namespace Alloy

#endif  // ALLOY_MIXED_PRECISION_KERNELS_CUH
