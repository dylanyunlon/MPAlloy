#pragma once

#ifndef DRESS_EMBEDDING_H
#define DRESS_EMBEDDING_H

#include <utility>

#include <dress/tensor.h>
#include <dress/optimizer.h>

namespace Dress {

typedef u_int64_t key_t;

struct EmbeddingOption {
    enum Type {
        Host, Hypra, Naive, Hash
    };

    Type type;
    size_t item_sz;
    DType ktype, dtype;
    EmbeddingOption(size_t sz_=0, DType ktype_=INT32, DType dtype_=FP32):
        item_sz(sz_), ktype(ktype_), dtype(dtype_) {}
};

class Embedding {
protected:
    size_t item_sz_;
    DType ktype_, dtype_;
    Optimizer* opt_;
    EmbeddingOption::Type type_;

public:
    /*
     * Prefetch generates an query id (qid) for each batch.
     * Use qid as a token for pull and push operations.
     * Update applies all the pushed gradients to the embedding.
     */
    Embedding(size_t item_sz, DType ktype=INT32, DType dtype=FP32,
            Optimizer* opt=0):
        item_sz_(item_sz), ktype_(ktype), dtype_(dtype), opt_(opt) {}

    virtual uint64_t prefetch(Tensor idx, size_t batch_size, Tensor* out=0) = 0;
    virtual void pull(uint64_t qid, Tensor* out=0) = 0;
    virtual void push(uint64_t qid, const Tensor& grad) = 0;
    virtual void release(uint64_t qid) = 0;
    virtual void update() = 0;
    virtual void reshard() = 0;

    virtual size_t getBatchSize(uint64_t qid) = 0;

public:
    static Embedding* create(const EmbeddingOption& opt);

    inline EmbeddingOption::Type type() {
        return this->type_;
    }

    inline size_t itemBytes() {
        return this->item_sz_ * typeSize(this->dtype_);
    }

    // shortcuts
    inline uint64_t pullTo(Tensor idx, size_t batch_size, Tensor& out) {
        auto qid = this->prefetch(idx, batch_size, &out);
        this->pull(qid);
        return qid;
    }
    inline Tensor pullOut(uint64_t qid, size_t batch_size) {
        Tensor t = Tensor::create(item_sz_ * batch_size, dtype_);
        this->pull(qid, &t);
        return t;
    }
    inline std::pair<Tensor, uint64_t>  pullOut(Tensor idx, size_t batch_size) {
        Tensor t = Tensor::create(item_sz_ * batch_size, dtype_);
        auto qid = this->pullTo(idx, batch_size, t);
        return std::pair<Tensor, uint64_t>(t, qid);
    }

    // optimizer
    void setOptimizer(Optimizer* opt) {
        this->opt_ = opt;
    }
    Optimizer* getOptimizer() {
        return this->opt_;
    }
};

}; // namespace Dress

#endif  // DRESS_EMBEDDING_H

