#pragma once

#ifndef DRESS_HOST_EMBEDDING_SERVER_H
#define DRESS_HOST_EMBEDDING_SERVER_H

#include <dress/optimizer.h>
#include <dress/config.h>

#include "../thread_pool.hh"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <thread>

#ifdef DRESS_USE_UCWO
#include <ucwo.hh>
#endif  // DRESS_USE_UCWO

namespace Dress {

const key_t flag_skip = 1ull << 30;

#define GEN_CHUNK_INFO(_idx_) \
    auto target_rank = _idx_ / this->chunk_sz_ % this->world_size; \
    auto chunk_idx = _idx_ / this->chunk_sz_ / this->world_size; \
    auto chunk_offset = _idx_ % this->chunk_sz_;

struct MappingEntry {
    std::unordered_map<key_t, uint64_t> map;
    std::shared_mutex mtx;
};

struct QueryPack {
    uint64_t *idx;
    size_t sz;
    Tensor grad;
    int flags; // 0x1 Pulled 0x2 Pushed 0x4 Updated
};

struct EmbeddingChunk {
    Tensor param, grad;
    size_t opt_tid;
};

struct TensorPool {
    std::atomic_ullong ptr;
    std::vector<Tensor> pool;
    std::vector<Tensor> grad;
    std::vector<size_t> opt_tid;
    std::shared_mutex mtx;

    TensorPool(): ptr(0) {}
    Tensor& get(int i) {
        // mtx.lock_shared();
        auto& t = pool[i];
        // mtx.unlock_shared();
        return t;
    }
    size_t size() {
        return pool.size();
    }
};

using handle_t = std::function<void()>;

class HostEmbeddingServer {
protected:
    size_t item_sz_;
    DType ktype_, dtype_;
    std::unordered_map<uint64_t, QueryPack> pcks_;
    std::atomic_ullong pck_ptr_;
    std::mutex pck_mtx_;
    std::queue<uint64_t> upd_pcks_;

    /*
     * First lookup chunk and offset by a few maps.
     * Use last `map_bits_` bits of id as direct mapping.
     */
    size_t map_bits_;
    uint64_t map_mask_;
    size_t chunk_sz_;
    MappingEntry* maps_;

    std::shared_ptr<TensorPool> pool_;

    inline size_t ptr2global(size_t idx, size_t rank) {
        return idx % this->chunk_sz_ +
            (idx / this->chunk_sz_ * this->world_size + rank) * this->chunk_sz_;
    }

    std::thread* server_thr_;
    ThreadPool<handle_t>* prepool_;
    ThreadPool<handle_t>* comppool_;
    int rank, world_size, tag_mask;

#ifdef DRESS_USE_UCWO
    UCWO::World* world;
#endif  // DRESS_USE_UCWO

protected:
    void init();
    void destroy();

    void handleLookup(int source, int tag, size_t sz);
    void handleInsert(int source, int tag, size_t sz);
    void handleRegister(int source, int tag, size_t sz);
    void handlePull(int source, int tag, uint64_t pckid);
    void handlePush(int source, int tag, uint64_t pckid);
    static void serverThreadFunc(HostEmbeddingServer* e);

public:
    inline int getRank() { return this->rank; }
    inline int getWorldSize() { return this->world_size; }
    inline int getTargetRank(uint64_t idx) {
        GEN_CHUNK_INFO(idx);
        return target_rank;
    }
    inline size_t itemBytes() {
        return this->item_sz_ * typeSize(this->dtype_);
    }
    inline void maskTag(int& tag) {
        tag &= this->tag_mask;
    }

public:
    HostEmbeddingServer(size_t item_sz, DType ktype, DType dtype):
            item_sz_(item_sz), ktype_(ktype), dtype_(dtype),
            pck_ptr_(0) {
        this->init();
    }

    ~HostEmbeddingServer() {
        this->destroy();
    }

    template<typename KeyType>
    void localLookup(KeyType*, size_t batch_size, uint64_t* idx,
            int preferred_rank);
    void localLookup(Tensor uid, size_t batch_size, uint64_t* idx,
            int preferred_rank=-1);

    uint64_t registerPck(const QueryPack& pck);
    void pullLocal(uint64_t pckid, char* out);
    void pushLocal(uint64_t pckid, char* grad);
    void update(Optimizer* opt);
    void createChunks();
#ifdef DRESS_USE_UCWO
    UCWO::Worker* getWorker();
    void pullRemote(UCWO::Worker* w, uint64_t idx, char* out);
#endif  // DRESS_USE_UCWO
};


const int max_query_len    = 16;
const int tag_query        = 1;
const uint64_t query_key_lookup = 2;
const uint64_t query_insert     = 3;
const uint64_t query_register   = 4;
const uint64_t query_pull       = 5;
const uint64_t query_push       = 6;
const uint64_t query_terminate  = 7;

const int idx_rank_bits        = 8;
const uint64_t idx_rank_mask   = (1ul << idx_rank_bits) - 1ul;
const uint64_t mask_not_found  = 1ull << 63;

}; // namespace Dress

#endif  // DRESS_HOST_EMBEDDING_SERVER_H

