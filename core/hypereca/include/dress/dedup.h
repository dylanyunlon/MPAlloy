#pragma once
#ifndef DRESS_DEDUP_H
#define DRESS_DEDUP_H

#include <dress/tensor.h>

namespace Dress {

struct DedupOutput {
    size_t nnz;
    Tensor val, lookup;
};

DedupOutput dedupKeys(const Tensor& keys, DType key_t);
DedupOutput noDedupKeys(const Tensor& keys, DType key_t);

};  // namespace Dress

#endif  // DRESS_DEDUP_H
