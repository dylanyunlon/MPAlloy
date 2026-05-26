template<typename KeyType, typename ValType>
__global__ void combineDedupedByCSRForwardKernel(
        size_t n_rows,
        const KeyType* ptr, const size_t ptr_offset,
        const KeyType* lookup, const ValType* val,
        ValType* out, float alpha, float beta) {
    auto row_offset = threadIdx.x;
    auto row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    auto item_sz = blockDim.x;
    if (row_idx >= n_rows) {
        return;
    }

    size_t i0 = ptr[row_idx] - ptr_offset, i1 = ptr[row_idx + 1] - ptr_offset;
    ValType res = 0.;
    for (size_t i = i0; i < i1; ++i) {
        auto idx = lookup[i];
        res += val[idx * item_sz + row_offset];
    }

    ValType* out_ptr = out + row_idx * item_sz;
    if (alpha != 1.) {
        res *= alpha;
    }
    if (beta != 0.) {
        res += beta * out_ptr[row_offset];
    }
    out_ptr[row_offset] = res;
}


template<typename KeyType, typename ValType>
__global__ void combineDedupedByCSRBackwardKernel(
        size_t n_rows,
        const KeyType* ptr, const size_t ptr_offset,
        const KeyType* lookup, const ValType* val,
        ValType* out, float alpha) {
    auto row_offset = threadIdx.x;
    auto row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    auto item_sz = blockDim.x;
    if (row_idx >= n_rows) {
        return;
    }
    ValType v = val[row_idx * item_sz + row_offset] * alpha;
    size_t i0 = ptr[row_idx] - ptr_offset, i1 = ptr[row_idx + 1] - ptr_offset;
    for (size_t i = i0; i < i1; ++i) {
        auto idx = lookup[i];
        atomicAdd(out + idx * item_sz + row_offset, v);
    }
}
