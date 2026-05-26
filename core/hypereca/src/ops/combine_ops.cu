#include <dress/combine_ops.h>
#include <dress/logging.h>
#include "combine_ops_impl.cuh"

namespace Dress {
void combineDedupedByCSRForward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr, const size_t ptr_offset,
        size_t nnz, const void* lookup,
        DType val_t, const void* val,
        void* out, cudaStream_t stream,
        const float alpha, const float beta) {
    DISPATCH_KNV(([&] {
        auto row_per_blk = 1024 / item_sz;
        dim3 grid_sz((n_rows - 1) / row_per_blk + 1);
        dim3 blk_sz(item_sz, row_per_blk);
        combineDedupedByCSRForwardKernel<KeyType, ValType>
                <<<grid_sz, blk_sz, 0, stream>>>
                (n_rows, (const KeyType*)ptr, ptr_offset,
                 (const KeyType*)lookup, (const ValType*)val,
                 (ValType*)out, alpha, beta);
    }));
}

void combineDedupedByCSRBackward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr, const size_t ptr_offset,
        size_t nnz, const void* lookup,
        DType val_t, const void* val,
        void* out, cudaStream_t stream,
        const float alpha, const float beta) {
    DISPATCH_KNV(([&] {
        cudaMemsetAsync(out, 0, sizeof(ValType) * nnz * item_sz, stream);
        auto row_per_blk = 1024 / item_sz;
        dim3 grid_sz((n_rows - 1) / row_per_blk + 1);
        dim3 blk_sz(item_sz, row_per_blk);
        combineDedupedByCSRBackwardKernel<KeyType, ValType>
                <<<grid_sz, blk_sz, 0, stream>>>
                (n_rows, (const KeyType*)ptr, ptr_offset,
                 (const KeyType*)lookup, (const ValType*)val,
                 (ValType*)out, alpha);
    }));
}

};  // namespace Dress
