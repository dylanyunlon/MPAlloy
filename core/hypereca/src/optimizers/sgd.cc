#include "sgd.h"

#include <dress/config.h>
#include <dress/logging.h>

#include <algorithm>


namespace Dress {

size_t SGD::addTensor(Tensor& t) {
    // TODO: assert data type match
    auto tid = ptr_.size();
    ptr_.push_back((void*)t.getPtr());
    return tid;
}

template<class DataType>
void stepSGD(DataType* param, DataType* grad, size_t n, DataType lr) {
    for (size_t i = 0; i < n; ++i) {
        param[i] -= grad[i] * lr;
    }
}

void SGD::step(size_t tid, size_t offset, size_t size,
        const Tensor& grad, size_t grad_offset) {
    DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
        stepSGD<DataType>(((DataType*)this->ptr_[tid]) + offset, 
                ((DataType*)grad.getPtr()) + grad_offset,
                size, this->lr_);
    }));
}

void SGD::setLR(double lr) {
    this->lr_ = lr;
}
};  // namespace Dress
