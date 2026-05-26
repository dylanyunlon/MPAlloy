/*
 * Hypra Embedding: hybrid parallel embedding layer
 * Hot items are stored in data-parallel mode on GPUs, and cold ones are in
 * host memory.
 */
#pragma once

#include "host_embedding.h"

namespace Dress {

struct Buffer {
    void* ptr=0;
    size_t sz=0;
};

struct QueryData {
    Tensor cp_from, cp_to;
    std::vector<Tensor> cp_froms, cp_tos;
    size_t sz;
};

class HypraEmbedding: public HostEmbedding {
private:
    std::vector<Buffer> host_buf_, cuda_buf_;
    std::unordered_map<key_t, uint64_t> freq_keys_;
    std::vector<Tensor> gpu_pool_, gpu_grad_;
    std::vector<size_t> opt_tid_;

    int gpu_count_;

    std::mutex qry_data_mtx_;
    std::unordered_map<uint64_t, QueryData> qry_data_;

protected:
    void init();
    void destroy();

    void* getBuffer(size_t sz, bool host, int idx);

public:
    HypraEmbedding(size_t item_sz, DType ktype=INT32, DType dtype=FP32):
        HostEmbedding(item_sz, ktype, dtype) {
        this->init();
    }
    ~HypraEmbedding() {
        this->destroy();
    }

    uint64_t prefetch(Tensor uid, size_t batch_size, Tensor* out=0) override;
    void pull(uint64_t qid, Tensor* out=0) override;
    void push(uint64_t qid, const Tensor& grad) override;
    void update() override;
    void reshard() override;
};

};  // namespace Dress
