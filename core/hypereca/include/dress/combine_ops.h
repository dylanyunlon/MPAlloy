#pragma once
#ifndef DRESS_COMBINE_OPS_H
#define DRESS_COMBINE_OPS_H

#include <dress/spec.h>

#define DISPATCH_KNV(__fn__) \
    DISPATCH_INT(key_t, KeyType, ([&]() { \
        DISPATCH_FLOAT(val_t, ValType, __fn__); \
    }));

namespace Dress {

void combineByCSRForward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr,
        DType val_t, const void* val, void* out,
        const float alpha=1., const float beta=0.);

void combineByCSRBackward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr,
        DType val_t, const void* grad, void* out,
        const float alpha=1., const float beta=0.);

#ifdef DRESS_USE_CUDA

#ifndef __NVCC__
struct cudaStream_t;
#endif  // __NVCC__

void combineDedupedByCSRForward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr, const size_t ptr_offset,
        size_t nnz, const void* lookup,
        DType val_t, const void* val,
        void* out, cudaStream_t stream,
        const float alpha=1., const float beta=0.);
void combineDedupedByCSRBackward(size_t nout, size_t item_sz,
        DType key_t, const void* ptr, const size_t ptr_offset,
        size_t nnz, const void* lookup,
        DType val_t, const void* val,
        void* out, cudaStream_t stream,
        const float alpha=1., const float beta=0.);

#endif  // DRESS_USE_CUDA

};  // namespace Dress

#endif  // DRESS_COMBINE_OPS_H
