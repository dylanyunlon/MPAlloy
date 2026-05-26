#pragma once

#ifndef DRESS_OPTIMIZER_H
#define DRESS_OPTIMIZER_H

#include <dress/tensor.h>

namespace Dress {

struct OptimizerOption {
    enum OptimizerType {
        SGD
    };

    OptimizerType type;
    double lr;
    DType dtype;
    OptimizerOption(OptimizerType type_=SGD, double lr_=0., DType dtype_=FP32):
        type(type_), lr(lr_), dtype(dtype_) {}
};

class Optimizer {
public:
    static Optimizer* create(const OptimizerOption& opt);

    virtual size_t addTensor(Tensor& t) = 0; // returns an index `tid`
    virtual void step(size_t tid, size_t offset, size_t size,
            const Tensor& grad, size_t grad_offset=0) = 0;
#ifdef DRESS_USE_CUDA
    virtual void step(size_t tid, size_t size, const Tensor& grad,
            cudaStream_t stream) = 0;
#endif  // DRESS_USE_CUDA
    virtual void setLR(double lr) = 0;
};

};  // namespace Dress

#endif  // DRESS_OPTIMIZER_H
