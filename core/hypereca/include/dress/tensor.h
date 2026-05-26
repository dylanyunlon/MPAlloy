#pragma once

#ifndef DRESS_TENSOR_H
#define DRESS_TENSOR_H

#include <cstdlib>

#include <dress/spec.h>

#ifdef DRESS_USE_CUDA
#include <cuda_runtime.h>
#else
struct cudaStream_t;
#endif  // DRESS_USE_CUDA

namespace Dress {

class Tensor {
private:
    size_t sz_;
    void* ptr_;
    Device dev_;

public:
    Tensor(void* ptr=0, size_t sz=0, Device dev=CPU):
        ptr_(ptr), sz_(sz), dev_(dev) {}

    Tensor(const Tensor& other):
        sz_(other.sz_), dev_(other.dev_), ptr_(other.ptr_) {}
    ~Tensor() {}

    static Tensor create(size_t sz) {
        void* p = malloc(sz);
        return Tensor(p, sz);
    }
    static Tensor create(size_t sz, DType dtype) {
        void* p = malloc(sz * typeSize(dtype));
        return Tensor(p, sz * typeSize(dtype));
    }

    void* getPtr() {
        return this->ptr_;
    }
    const void* getPtr() const {
        return this->ptr_;
    }

    void copy(const Tensor& other);
    void copy(const Tensor& other, cudaStream_t stream);

    inline void free() {
        if (this->dev_ == CPU) {
            ::free(this->ptr_);
        }
#ifdef DRESS_USE_CUDA
        if (this->dev_ == CUDA) {
            cudaFree(this->ptr_);
        }
#endif  // DRESS_USE_CUDA
    }

    inline Device device() const {
        return this->dev_;
    }
    inline size_t size() const {
        return this->sz_;
    }
    inline void resize(size_t sz) {
        this->sz_ = sz;
    }

    void printSpec();

    Tensor clone() const;
    Tensor to(Device d, int idx=0) const;
    Tensor to(Device d, int idx, cudaStream_t stream) const;
};

};  // namespace Dress

#endif // DRESS_TENSOR_H
