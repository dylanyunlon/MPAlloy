#pragma once

#ifndef DRESS_NAIVE_EMBEDDING_H
#define DRESS_NAIVE_EMBEDDING_H

#include <dress/embedding.h>
#include <dress/config.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>

#include "naive_embedding_server.h"
#include "host_embedding.h"

namespace Dress {

class NaiveEmbedding: public Embedding {
protected:
    std::unordered_map<uint64_t, QueryEntry> qrys_;
    std::atomic_ullong qry_ptr_;
    std::mutex qry_mtx_;

    NaiveEmbeddingServer* server_;

    void globalLookup(const Tensor& uid, QueryEntry& qry);
    void init();
    void destroy();
    void reshard() override;

public:
    NaiveEmbedding(size_t item_sz, DType ktype=INT32, DType dtype=FP32):
            Embedding(item_sz, ktype, dtype),
            qry_ptr_(0) {
        this->init();
    }

    ~NaiveEmbedding() {
        this->destroy();
    }

    uint64_t prefetch(Tensor uid, size_t batch_size, Tensor* out=0) override;
    void pull(uint64_t qid, Tensor* out=0) override;
    void release(uint64_t qid) override;
    void push(uint64_t qid, const Tensor& grad) override;
    void update() override;
    size_t getBatchSize(uint64_t qid) override;
};

}; // namespace Dress

#endif  // DRESS_NAIVE_EMBEDDING_H
