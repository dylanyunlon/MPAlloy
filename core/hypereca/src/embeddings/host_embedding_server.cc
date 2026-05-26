#include "host_embedding_server.h"

#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <unistd.h>
#include <assert.h>

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

#include <dress/tensor.h>
#include <dress/profiler.h>

#include "../ops/initializer.h"

namespace Dress {

static const auto nth_loader = cfgi()->getInt("nth_loader");
static const auto nth_prefetch = cfgi()->getInt("nth_prefetch");
static const auto nth_embedding = cfgi()->getInt("nth_embedding");
static const auto nth_optimizer = cfgi()->getInt("nth_optimizer");


void HostEmbeddingServer::init() {
    this->chunk_sz_ = cfgi()->getInt("host_emb_chunk_sz");
    this->pool_ = std::make_shared<TensorPool>();
    this->prepool_ = new ThreadPool<handle_t>(nth_loader);
    this->comppool_ = new ThreadPool<handle_t>(nth_optimizer / nth_embedding);

#ifdef DRESS_USE_MPI
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &flag);
    } else {
        MPI_Query_thread(&flag);
    }
    if (flag != MPI_THREAD_MULTIPLE) {
        throw "MPI should be initialized with MPI_THREAD_MULTIPLE for DrESS";
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

    MPI_Aint* ub_ptr;
    int isset;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub_ptr, &isset);
    this->tag_mask = *ub_ptr;

    this->server_thr_ = new std::thread(HostEmbeddingServer::serverThreadFunc, this);

#ifdef DRESS_USE_UCWO
    this->world = new UCWO::World(MPI_COMM_WORLD);
    int nth = cfgi()->getInt("ucwo_emb_worker");
    for (int i = 0; i < nth; ++i) {
        this->world->newWorker(false);
    }
#endif  // DRESS_USE_UCWO

#else
    this->rank = 0;
    this->world_size = 1;
#endif  // DRESS_USE_MPI

    pfri()->setRank(this->rank);

    this->map_bits_ = cfgi()->getInt("host_emb_map_bits");
    this->map_mask_ = (1u << this->map_bits_) - 1u;
    size_t global_map_count = 1u << this->map_bits_;
    size_t local_map_count = global_map_count / this->world_size;
    if (this->rank < global_map_count % this->world_size) {
        ++local_map_count;
    }
    if (local_map_count > 0) {
        this->maps_ = new MappingEntry[local_map_count];
    }

    int n_pre_create = cfgi()->getInt("host_emb_pre_create");
    while (n_pre_create--) {
        auto numel = this->chunk_sz_ * this->item_sz_;
#ifdef DRESS_USE_UCWO
        auto ptr = this->world->expose(0, numel * typeSize(this->dtype_));
        auto t = Tensor(ptr);
#else
        auto t = Tensor::create(numel, this->dtype_);
#endif  // DRESS_USE_UCWO
        // TODO: Customize initialization
        DISPATCH_FLOAT(this->dtype_, FPType, ([&] {
            uniformFill<FPType>(t.getPtr(), numel, -.05, .05);
        }));
        this->pool_->pool.push_back(t);
        t = Tensor::create(numel, this->dtype_);
        this->pool_->grad.push_back(t);
    }
}

void HostEmbeddingServer::destroy() {
    delete [] this->maps_;
    delete this->prepool_;
    delete this->comppool_;
#ifdef DRESS_USE_MPI
    uint64_t q = query_terminate;
    MPI_Send(&q, 1, MPI_UNSIGNED_LONG_LONG, this->rank, tag_query, MPI_COMM_WORLD);
    this->server_thr_->join();
    delete this->server_thr_;

#ifdef DRESS_USE_UCWO
    delete this->world;
#endif  // DRESS_USE_UCWO
#endif  // DRESS_USE_MPI
}

void HostEmbeddingServer::serverThreadFunc(HostEmbeddingServer* e) {
#ifdef DRESS_USE_MPI
    bool endth = false;
    while (!endth) {
        uint64_t q[max_query_len];
        MPI_Status s;
        MPI_Recv(q, max_query_len, MPI_UNSIGNED_LONG_LONG, MPI_ANY_SOURCE,
                tag_query, MPI_COMM_WORLD, &s);
        switch (q[0]) {
            case query_key_lookup:
                e->handleLookup(s.MPI_SOURCE, (int)q[1], q[2]);
                break;
            case query_insert:
                e->handleInsert(s.MPI_SOURCE, (int)q[1], q[2]);
                break;
            case query_register:
                e->handleRegister(s.MPI_SOURCE, (int)q[1], q[2]);
                break;
            case query_pull:
                e->handlePull(s.MPI_SOURCE, (int)q[1], q[2]);
                break;
            case query_push:
                e->handlePush(s.MPI_SOURCE, (int)q[1], q[2]);
                break;
            case query_terminate:
                endth = true;
                break;
            default:
                throw "Unknown command";
        }
    }
#endif  // DRESS_USE_MPI
}

void HostEmbeddingServer::localLookup(Tensor uid, size_t batch_size,
        uint64_t* idx, int preferred_rank) {
    DISPATCH_INT(this->ktype_, KeyType, ([&]() {
        this->localLookup<KeyType>((KeyType*)uid.getPtr(), batch_size,
                idx, preferred_rank);
    }));
}

template<typename KeyType>
void HostEmbeddingServer::localLookup(KeyType* key_ptr, size_t batch_size,
        uint64_t* idx, int preferred_rank) {
    size_t count_not_found = 0;
    std::unordered_map<size_t, std::vector<size_t>> mp_seps;
// #pragma omp parallel for reduction(+:count_not_found) num_threads(nth_prefetch)
    for (size_t i = 0; i < batch_size; ++i) {
        key_t uid = key_ptr[i];
        auto map_uid = (uid & this->map_mask_) / this->world_size;
        mp_seps[map_uid].push_back(i);
    }

    for (auto& mp_it: mp_seps) {
        auto& mp = this->maps_[mp_it.first];
        std::vector<size_t> vnf;
        mp.mtx.lock_shared();
        for (auto& i: mp_it.second) {
            key_t uid = key_ptr[i];
            auto it = mp.map.find(uid);
            if (it != mp.map.end()) {
                idx[i] = it->second;
            } else {
                count_not_found += 1;
                vnf.push_back(i);
            }
        }
        mp.mtx.unlock_shared();
        mp_it.second = vnf;
    }
    if (count_not_found == 0) {
        return;
    }

    int tgt_rank = preferred_rank == -1 ? this->rank : preferred_rank;
    size_t base_idx;
    if (tgt_rank == this->rank) {
        base_idx = this->pool_->ptr.fetch_add(count_not_found);
    } else {
#ifdef DRESS_USE_MPI
        int qry_tag = query_insert ^ (this->rank << 10) ^ (gettid() << 4);
        this->maskTag(qry_tag);
        uint64_t q[3] = {query_insert, (uint64_t)qry_tag, count_not_found};
        MPI_Send(q, 3, MPI_UNSIGNED_LONG_LONG, tgt_rank, tag_query, MPI_COMM_WORLD);
        MPI_Recv(&base_idx, 1, MPI_UNSIGNED_LONG_LONG, tgt_rank, qry_tag,
                MPI_COMM_WORLD, 0);
#endif  // DRESS_USE_MPI
    }
    std::atomic_ullong ptr(base_idx);
// #pragma omp parallel for num_threads(nth_prefetch)
    for (auto& mp_it: mp_seps) {
        auto& mp = this->maps_[mp_it.first];
        mp.mtx.lock();
        for (auto& i: mp_it.second) {
            key_t uid = key_ptr[i];
            auto it = mp.map.find(uid);
            if (it != mp.map.end()) {
                idx[i] = it->second;
            } else {
                mp.map[uid] = idx[i] = this->ptr2global(ptr++, tgt_rank);
            }
        }
        mp.mtx.unlock();
    }
}

void HostEmbeddingServer::handleLookup(int source, int recv_tag, size_t recv_sz) {
    this->prepool_->push([=]() mutable {
        Tensor uid = Tensor::create(recv_sz, this->ktype_);
        Tensor idx = Tensor::create(recv_sz, UINT64);
        MPI_Recv(uid.getPtr(), recv_sz * typeSize(this->ktype_), MPI_BYTE,
                source, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        DISPATCH_INT(this->ktype_, KeyType, ([&]() {
            this->localLookup(uid, recv_sz, (uint64_t*)idx.getPtr(), -1);
        }));
        MPI_Send(idx.getPtr(), recv_sz, MPI_UNSIGNED_LONG_LONG,
                source, recv_tag, MPI_COMM_WORLD);
        uid.free();
        idx.free();
    });
}

void HostEmbeddingServer::handleInsert(int source, int ret_tag, size_t sz) {
    auto base = this->pool_->ptr.fetch_add(sz);
    this->prepool_->push([=]() {
        MPI_Send(&base, 1, MPI_UNSIGNED_LONG_LONG, source,
                ret_tag, MPI_COMM_WORLD);
    });
}

uint64_t HostEmbeddingServer::registerPck(const QueryPack& pck) {
    uint64_t pckid = ++this->pck_ptr_;
    std::lock_guard<std::mutex> lck(this->pck_mtx_);
    this->pcks_[pckid] = pck;
    return pckid;
}

void HostEmbeddingServer::handleRegister(int source, int recv_tag, size_t sz) {
    this->prepool_->push([=]() mutable {
        QueryPack pck;
        pck.sz = sz;
        pck.idx = new uint64_t[pck.sz];
        MPI_Recv(pck.idx, pck.sz, MPI_UNSIGNED_LONG_LONG, source,
                recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pck.flags = 0;
        auto pckid = this->registerPck(pck);
        MPI_Send(&pckid, 1, MPI_UNSIGNED_LONG_LONG, source,
                recv_tag, MPI_COMM_WORLD);
    });
}

void HostEmbeddingServer::handlePull(int source, int recv_tag, uint64_t pckid) {
    this->comppool_->push([=]() mutable {
        this->pck_mtx_.lock();
        auto pck = this->pcks_[pckid];
        this->pck_mtx_.unlock();
        char* tmp_out = new char[this->itemBytes() * pck.sz];
        this->pullLocal(pckid, tmp_out);
        MPI_Send(tmp_out, pck.sz * this->itemBytes(), MPI_CHAR,
                source, recv_tag, MPI_COMM_WORLD);
        delete [] tmp_out;
    });
}

void HostEmbeddingServer::createChunks() {
    std::lock_guard<std::shared_mutex> lck(this->pool_->mtx);

    auto numel = this->chunk_sz_ * this->item_sz_;
    while (this->pool_->ptr > this->chunk_sz_ * this->pool_->size()) {
#ifdef DRESS_USE_UCWO
        auto ptr = this->world->expose(0, numel * typeSize(this->dtype_));
        auto t = Tensor(ptr);
#else
        auto t = Tensor::create(numel, this->dtype_);
#endif  // DRESS_USE_UCWO
        // TODO: Customize initialization
        DISPATCH_FLOAT(this->dtype_, FPType, ([&] {
            uniformFill<FPType>(t.getPtr(), numel, -.05, .05);
        }));
        this->pool_->pool.push_back(t);
        t = Tensor::create(numel, this->dtype_);
        this->pool_->grad.push_back(t);
    }
}

void HostEmbeddingServer::pullLocal(uint64_t pckid, char* out) {
    this->createChunks();
    size_t batch_size = this->pcks_[pckid].sz;
    uint64_t* idx = this->pcks_[pckid].idx;
#pragma omp parallel for num_threads(nth_embedding)
    for (size_t i = 0; i < batch_size; ++i) {
        GEN_CHUNK_INFO(idx[i]);
        // TODO: better memcpy
        auto emb_ptr = (char*)this->pool_->get(chunk_idx).getPtr();
        memcpy(out + i * this->itemBytes(),
                emb_ptr + chunk_offset * this->itemBytes(),
                this->itemBytes());
    }
}

void HostEmbeddingServer::pushLocal(uint64_t pckid, char* grad) {
    std::lock_guard<std::mutex> lck(this->pck_mtx_);
    auto& pck = this->pcks_[pckid];
    pck.grad = Tensor(grad, pck.sz * this->item_sz_);
    pck.flags |= 0x2;
    this->upd_pcks_.push(pckid);
}

void HostEmbeddingServer::handlePush(int source, int push_tag, uint64_t pckid) {
    this->comppool_->push ([=]() mutable {
        // TODO: maybe not locked
        auto& pck = this->pcks_[pckid];
        char* grad = new char[pck.sz * this->itemBytes()];
        MPI_Recv(grad, pck.sz * this->itemBytes(), MPI_CHAR, source,
                push_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pck.grad = Tensor(grad);
        pck.flags |= 0x2;
        std::lock_guard<std::mutex> lck(this->pck_mtx_);
        this->upd_pcks_.push(pckid);
    });
}

template<typename DataType>
void accumGrad(DataType* out, const DataType* in, size_t sz) {
    for (size_t i = 0; i < sz; ++i) {
        out[i] += in[i];
    }
}

void HostEmbeddingServer::update(Optimizer* opt) {
    while (this->pool_->opt_tid.size() < this->pool_->size()) {
        auto& chunk = this->pool_->get(this->pool_->opt_tid.size());
        this->pool_->opt_tid.push_back(opt->addTensor(chunk));
    }
    // Assume that in each query, keys are unique (dedupd).
    while (this->upd_pcks_.size() < this->world_size) {
        std::this_thread::yield();
    }

    size_t pckn = 0;
    // TIMER_START(tupd);
    while (!this->upd_pcks_.empty()) {
        this->pck_mtx_.lock();
        auto pckid = this->upd_pcks_.front();
        this->upd_pcks_.pop();
        auto pck = this->pcks_[pckid];
        this->pcks_.erase(pckid);
        this->pck_mtx_.unlock();
#pragma omp parallel for num_threads(nth_optimizer)
        for (auto i = 0; i < pck.sz; ++i) {
            GEN_CHUNK_INFO(pck.idx[i]);
            auto tid = this->pool_->opt_tid[chunk_idx];
            auto offset = chunk_offset * this->item_sz_;
            opt->step(tid, chunk_offset * this->item_sz_, this->item_sz_,
                    pck.grad, i * this->item_sz_);
        }
        delete [] pck.idx;
        pckn += pck.sz;
    }
    // TIMER_END(tupd);
    // auto t = TIMER_GET(tupd);
    // auto bw = pckn * this->itemBytes() / t;
    // PINFO << REPORT_TIME(twait) << " Update time " << t * 1e3 << " ms; bw " << bw * 1e-9 << " GBps\n";
}

#ifdef DRESS_USE_UCWO
UCWO::Worker* HostEmbeddingServer::getWorker() {
    thread_local static UCWO::Worker* w = 0;
    static std::atomic_int wc = 0;
    if (w) {
        return w;
    }
    return w = this->world->worker(++wc);
}

void HostEmbeddingServer::pullRemote(UCWO::Worker* w, uint64_t idx, char* out) {
    GEN_CHUNK_INFO(idx);
    w->get(target_rank, chunk_idx, chunk_offset * this->itemBytes(), out,
            this->itemBytes());
}
#endif  // DRESS_USE_UCWO

};
