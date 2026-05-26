#include <dress/logging.h>
#include <dress/tensor.h>

#include <cstring>

#ifdef DRESS_USE_CUDA
#include <cuda_runtime.h>
#endif  // DRESS_USE_CUDA

namespace Dress {

Tensor Tensor::clone() const {
    if (this->dev_ == CPU) {
        // TODO: use better alloc and memcpy
        void* ptr = malloc(this->sz_);
        memcpy(ptr, this->ptr_, this->sz_);
        return Tensor(ptr, this->sz_, this->dev_);
    } else {
        throw "Unimplemented";
    }
}

Tensor Tensor::to(Device d, int idx) const {
#ifndef DRESS_USE_CUDA
    throw "Unimplemented";
#else
    if (this->dev_ == CPU && d == CUDA) {
        cudaSetDevice(idx);
        char* p_cuda;
        cudaMalloc(&p_cuda, this->sz_);
        cudaMemcpy(p_cuda, this->ptr_, this->sz_, cudaMemcpyHostToDevice);
        CUDA_SAFE_CHECK;
        return Tensor(p_cuda, this->sz_, CUDA);
    } else if (this->dev_ == CUDA && d == CPU) {
        void* ptr = malloc(this->sz_);
        cudaMemcpy(ptr, this->ptr_, this->sz_, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CHECK;
        return Tensor(ptr, this->sz_, CPU);
    }
    throw "Unimplemented";
#endif  // DRESS_USE_CUDA
}

Tensor Tensor::to(Device d, int idx, cudaStream_t stream) const {
#ifdef DRESS_USE_CUDA
    if (this->dev_ == CPU && d == CUDA) {
        cudaSetDevice(idx);
        char* p_cuda;
        cudaMalloc(&p_cuda, this->sz_);
        cudaMemcpyAsync(p_cuda, this->ptr_, this->sz_,
                cudaMemcpyHostToDevice, stream);
        return Tensor(p_cuda, this->sz_, CUDA);
    } else if (this->dev_ == CUDA && d == CPU) {
        void* ptr = malloc(this->sz_);
        cudaMemcpyAsync(ptr, this->ptr_, this->sz_,
                cudaMemcpyDeviceToHost, stream);
        return Tensor(ptr, this->sz_, CPU);
    }
#endif
    throw "Unimplemented";
}

void Tensor::copy(const Tensor& other) {
    this->sz_ = other.sz_;
    if (this->dev_ == CPU && other.dev_ == CPU) {
        memcpy(this->ptr_, other.ptr_, this->sz_);
    }
#ifdef DRESS_USE_CUDA
    if (this->dev_ == CPU && other.dev_ == CUDA) {
        cudaMemcpy(this->ptr_, other.ptr_, this->sz_, cudaMemcpyDeviceToHost);
    }
    if (this->dev_ == CUDA && other.dev_ == CPU) {
        cudaMemcpy(this->ptr_, other.ptr_, this->sz_, cudaMemcpyHostToDevice);
    }
#endif  // DRESS_USE_CUDA
}
void Tensor::copy(const Tensor& other, cudaStream_t stream) {
    this->sz_ = other.sz_;
#ifdef DRESS_USE_CUDA
    if (this->dev_ == CPU && other.dev_ == CUDA) {
        cudaMemcpyAsync(this->ptr_, other.ptr_, this->sz_,
                cudaMemcpyDeviceToHost, stream);
    }
    if (this->dev_ == CUDA && other.dev_ == CPU) {
        cudaMemcpyAsync(this->ptr_, other.ptr_, this->sz_,
                cudaMemcpyHostToDevice, stream);
    }
#endif  // DRESS_USE_CUDA
}

void Tensor::printSpec() {
    PINFO << "Tensor spec (" << this->ptr_ << "): size=" << this->sz_
        << " dev=" << this->dev_ << std::endl;
}

};  // namespace Dress
