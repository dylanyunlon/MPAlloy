#pragma once

#ifndef DRESS_HOST_EMBEDDING_H
#define DRESS_HOST_EMBEDDING_H

#include <dress/embedding.h>
#include <dress/config.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>

#include "host_embedding_server.h"

namespace Dress {

struct QueryEntry {
    uint64_t qid;
    uint64_t *idx;
    std::vector<uint64_t> rank_ptr;
    uint64_t *rank_ridx;
    size_t batch_size;
    Tensor* out;
    std::vector<uint64_t> packs;
    char* buf;
};

class HostEmbedding: public Embedding {
protected:
    std::unordered_map<uint64_t, QueryEntry> qrys_;
    std::atomic_ullong qry_ptr_;
    std::mutex qry_mtx_;

    HostEmbeddingServer* server_;

    void globalLookup(const Tensor& uid, QueryEntry& qry);
    void init();
    void destroy();

    void pullExchange(QueryEntry& qry, char* buf);
    void pushExchange(QueryEntry& qry, char* buf);

public:
    HostEmbedding(size_t item_sz, DType ktype=INT32, DType dtype=FP32):
            Embedding(item_sz, ktype, dtype),
            qry_ptr_(0) {
        this->init();
    }

    ~HostEmbedding() {
        this->destroy();
    }

    uint64_t prefetch(Tensor uid, size_t batch_size, Tensor* out=0) override;
    void pull(uint64_t qid, Tensor* out=0) override;
    void release(uint64_t qid) override;
    void push(uint64_t qid, const Tensor& grad) override;
    void update() override;
    size_t getBatchSize(uint64_t qid) override;
    void reshard() override;
};

}; // namespace Dress

#endif  // DRESS_HOST_EMBEDDING_H
