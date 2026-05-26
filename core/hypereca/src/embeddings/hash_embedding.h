#pragma once

#ifndef DRESS_HASH_EMBEDDING_H
#define DRESS_HASH_EMBEDDING_H

#include "naive_embedding.h"

namespace Dress {

class HashEmbedding: public NaiveEmbedding {
public:
    HashEmbedding(size_t item_sz, DType ktype=INT32, DType dtype=FP32):
            NaiveEmbedding(item_sz, ktype, dtype) {}

    uint64_t prefetch(Tensor uid, size_t batch_size, Tensor* out=0) override;
};

};

#endif  // DRESS_HASH_EMBEDDING_H
