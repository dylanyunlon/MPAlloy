#include "sgd.h"

#include <dress/logging.h>

namespace Dress {

const size_t sgd_stride = 128;

template<typename DataType>
__global__ void SGDKernel(size_t n, DataType* param,
        const DataType* grad, DataType lr) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * sgd_stride;
    if (idx >= n) {
        return;
    }
    for (; idx < n; idx += gridDim.x * blockDim.x * sgd_stride) {
        for (size_t o = 0, i = idx; o < sgd_stride && i < n; ++o, i += blockDim.x) {
            param[i] -= grad[i] * lr;
        }
    }
}

void SGD::step(size_t tid, size_t size, const Tensor& grad,
        cudaStream_t stream) {
    DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
        dim3 block_dim(512);
        dim3 grid_dim(80);
        auto grad_ptr = (DataType*)grad.getPtr();
        auto param_ptr = (DataType*)this->ptr_[tid];
        SGDKernel<DataType><<<block_dim, grid_dim, 0, stream>>>(size,
                param_ptr, grad_ptr, this->lr_);
    }));
}
};  // namespace Dress
