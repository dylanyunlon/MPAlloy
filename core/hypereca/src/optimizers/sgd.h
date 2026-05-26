#pragma once

#ifndef DRESS_SGD_H
#define DRESS_SGD_H

#include <dress/optimizer.h>
#include <vector>

namespace Dress {

class SGD: public Optimizer {
private:
    DType dtype_;
    double lr_;
    std::vector<void*> ptr_;
    size_t tid_;

public:
    SGD(DType dtype, double lr): dtype_(dtype), lr_(lr), tid_(0) {}

    size_t addTensor(Tensor& t) override;
    void step(size_t tid, size_t offset, size_t size,
            const Tensor& grad, size_t grad_offset=0) override;

#ifdef DRESS_USE_CUDA
    void step(size_t tid, size_t size, const Tensor& grad,
            cudaStream_t stream) override;
#endif  // DRESS_USE_CUDA

    void setLR(double lr) override;
};

};  // namespace Dress


#endif  // DRESS_SGD_H

