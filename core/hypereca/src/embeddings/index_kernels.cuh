template<typename DataType>
__global__ void indexPutKernel(size_t n, DataType* out,
        const DataType* in, const size_t* idx) {
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) {
        return;
    }
    out[idx[row_idx] * blockDim.x + threadIdx.x] = in[row_idx * blockDim.x + threadIdx.x];
}

template<typename DataType>
__global__ void indexGetKernel(size_t n, DataType* out,
        const DataType* in, const size_t* idx) {
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) {
        return;
    }
    out[row_idx * blockDim.x + threadIdx.x] = in[idx[row_idx] * blockDim.x + threadIdx.x];
}

template<typename DataType>
__global__ void indexCopyKernel(size_t n, DataType* out, const size_t* out_idx,
        const DataType* in, const size_t* in_idx) {
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) {
        return;
    }
    out[out_idx[row_idx] * blockDim.x + threadIdx.x] =
        in[in_idx[row_idx] * blockDim.x + threadIdx.x];
}

