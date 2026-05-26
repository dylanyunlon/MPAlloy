#include "host_embedding.h"
#include <dress/profiler.h>

#include <assert.h>
#include <omp.h>
#include <unistd.h>

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

#include <cstring>
#include <algorithm>

namespace Dress {

static const auto nth_prefetch = cfgi()->getInt("nth_prefetch");
static const auto nth_embedding = cfgi()->getInt("nth_embedding");
static const auto nth_optimizer = cfgi()->getInt("nth_optimizer");

void HostEmbedding::init() {
    this->server_ = new HostEmbeddingServer(this->item_sz_,
            this->ktype_, this->dtype_);
}

void HostEmbedding::destroy() {
    delete this->server_;
}

template<typename ValType>
using Vector2d = std::vector<std::vector<ValType>>;
template<typename ValType>
using Vector3d = std::vector<Vector2d<ValType>>;

template<typename KeyType>
void classifyKeys(const KeyType* key_ptr, size_t batch_size, size_t world_size,
        Vector3d<KeyType>& keys, Vector3d<size_t>& locs,
        size_t* idx) {
// #pragma omp parallel for num_threads(nth_prefetch)
    for (size_t i = 0; i < batch_size; ++i) {
        if (key_ptr[i] & flag_skip) {
            idx[i] = key_ptr[i];
            continue;
        }
        int rank = omp_get_thread_num();
        auto target_rank = key_ptr[i] % world_size;
        keys[target_rank][rank].push_back(key_ptr[i]);
        locs[target_rank][rank].push_back(i);
    }
}

const bool disable_ring = cfgi()->getBool("disable_ring");

void HostEmbedding::globalLookup(const Tensor& uid, QueryEntry& qry) {
    int server_world_size = this->server_->getWorldSize();
    int omp_world_size = nth_prefetch;
    DISPATCH_INT(this->ktype_, KeyType, ([&]() {
        auto key_ptr = (const KeyType*)uid.getPtr();
        Vector3d<KeyType> keys(server_world_size, Vector2d<KeyType>(omp_world_size));
        Vector3d<size_t> locs(server_world_size, Vector2d<size_t>(omp_world_size));
        classifyKeys<KeyType>(key_ptr, qry.batch_size, server_world_size,
                keys, locs, qry.idx);

        for (int fi = 0, i; fi < server_world_size; ++fi) {
            if (disable_ring) {
                i = fi;
            } else {
                i = (fi + this->server_->getRank()) % server_world_size;
            }
            size_t sz = 0, offset = 0;
            for (auto& v: keys[i]) {
                sz += v.size();
            }
            if (sz == 0) {
                continue;
            }

            KeyType* kc = new KeyType[sz];
            uint64_t* idxs = new uint64_t[sz];

            for (auto& v: keys[i]) {
                std::copy(v.begin(), v.end(), kc + offset);
                offset += v.size();
                v.clear();
            }

            if (i == this->server_->getRank()) {
                this->server_->localLookup(Tensor(kc), sz, idxs);
            } else {
                int qry_tag = query_key_lookup ^ (qry.qid << 12) ^
                    (this->server_->getRank() << 8) ^ (i << 4);
                this->server_->maskTag(qry_tag);
                uint64_t qry_args[3] = {query_key_lookup, (uint64_t)qry_tag, sz};
                MPI_Send(qry_args, 3, MPI_UNSIGNED_LONG_LONG, i, tag_query, MPI_COMM_WORLD);
                MPI_Send(kc, sizeof(KeyType) * sz, MPI_BYTE, i, qry_tag,
                        MPI_COMM_WORLD);
                MPI_Recv(idxs, sz, MPI_UNSIGNED_LONG_LONG, i, qry_tag,
                        MPI_COMM_WORLD, 0);
            }

            offset = 0;
            for (auto& v: locs[i]) {
                for (size_t& l: v) {
                    qry.idx[l] = idxs[offset++];
                }
            }
            locs[i].clear();
            delete [] kc;
            delete [] idxs;
        }
    }));
}

uint64_t HostEmbedding::prefetch(Tensor uid, size_t batch_size, Tensor* out) {
    // this->server_->upd_lck.lock_shared();
    QueryEntry qry;
    qry.qid = ++this->qry_ptr_;
    qry.batch_size = batch_size;
    qry.idx = new uint64_t[batch_size];
    qry.buf = new char[batch_size * this->itemBytes()];
    qry.out = out;

    // this->server_->upd_lck.unlock_shared();
    // this->server_->upd_lck.lock_shared();

    this->globalLookup(uid, qry);

    // this->server_->upd_lck.unlock_shared();
    // this->server_->upd_lck.lock_shared();

    int server_world_size = this->server_->getWorldSize();
    qry.rank_ptr.resize(server_world_size + 1);
    auto rank_sorted = new uint64_t[qry.batch_size];
    qry.rank_ridx = new uint64_t[qry.batch_size];

    int omp_world_size = nth_prefetch;
    using klpair = std::pair<uint64_t, size_t>;
    Vector3d<klpair> kl(server_world_size, Vector2d<klpair>(omp_world_size));

    // this->server_->upd_lck.unlock_shared();
    // this->server_->upd_lck.lock_shared();
// #pragma omp parallel for num_threads(nth_prefetch)
    for (size_t i = 0; i < qry.batch_size; ++i) {
        if (qry.idx[i] & flag_skip) {
            continue;
        }
        int rank = omp_get_thread_num();
        int target_rank = this->server_->getTargetRank(qry.idx[i]);
        kl[target_rank][rank].push_back(klpair(qry.idx[i], i));
    }
    for (int i = 0; i < server_world_size; ++i) {
        qry.rank_ptr[i + 1] = 0;
        for (auto& v: kl[i]) {
            qry.rank_ptr[i + 1] += v.size();
        }
    }
    qry.rank_ptr[0] = 0;
    for (int i = 0; i < server_world_size; ++i) {
        qry.rank_ptr[i + 1] += qry.rank_ptr[i];
    }
    qry.packs.resize(server_world_size);

    // this->server_->upd_lck.unlock_shared();
    // this->server_->upd_lck.lock_shared();

// #pragma omp parallel for num_threads(nth_prefetch)
    for (int i = 0; i < server_world_size; ++i) {
        auto n_ele = qry.rank_ptr[i + 1] - qry.rank_ptr[i];
        if (n_ele == 0) {
            qry.packs[i] = -1ull;
            continue;
        }
        size_t p = 0;
        std::vector<klpair> s(qry.rank_ptr[i + 1] - qry.rank_ptr[i]);
        for (auto& v: kl[i]) {
            for (auto& klp: v) {
                s[p++] = klp;
            }
        }
        // std::sort(s.begin(), s.end());
        p = qry.rank_ptr[i];
        for (auto klp: s) {
            rank_sorted[p] = klp.first;
            qry.rank_ridx[p] = klp.second;
            ++p;
        }
    }

    // this->server_->upd_lck.unlock_shared();

    for (int fi = 0, i; fi < server_world_size; ++fi) {
        if (disable_ring) {
            i = fi;
        } else {
            i = (fi + this->server_->getRank()) % server_world_size;
        }
        auto n_ele = qry.rank_ptr[i + 1] - qry.rank_ptr[i];
        if (n_ele == 0) {
            qry.packs[i] = -1ull;
            continue;
        }
        uint64_t pack_id;
        if (i == this->server_->getRank()) {
            QueryPack pck;
            pck.sz = n_ele;
            pck.idx = new uint64_t[n_ele];
            memcpy(pck.idx, rank_sorted + qry.rank_ptr[i],
                    n_ele * sizeof(uint64_t));
            pack_id = this->server_->registerPck(pck);
        } else {
            int tag_reg = query_register ^ (qry.qid << 12) ^
                (this->server_->getRank() << 8) ^ (i << 4);
            this->server_->maskTag(tag_reg);
            uint64_t qry_args[] = {query_register, (uint64_t)tag_reg, n_ele};
            MPI_Send(qry_args, 3, MPI_UNSIGNED_LONG_LONG,
                    i, tag_query, MPI_COMM_WORLD);
            MPI_Send(rank_sorted + qry.rank_ptr[i],
                    n_ele, MPI_UNSIGNED_LONG_LONG, i, tag_reg, MPI_COMM_WORLD);
            MPI_Recv(&pack_id, 1, MPI_UNSIGNED_LONG_LONG,
                    i, tag_reg, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        qry.packs[i] = pack_id;
    }
    // PINFO << "Reduced size " << qry.rank_ptr[server_world_size] << " / " << batch_size << "\n";

    delete [] rank_sorted;
    std::lock_guard<std::mutex> lock(this->qry_mtx_);
    qrys_[qry.qid] = qry;
    return qry.qid;
}

void HostEmbedding::pullExchange(QueryEntry& qry, char* buf) {
    int server_world_size = this->server_->getWorldSize();
    for (int fi = 0, i; fi < server_world_size; ++fi) {
        i = (fi + this->server_->getRank()) % server_world_size;
        auto n_ele = qry.rank_ptr[i + 1] - qry.rank_ptr[i];
        if (n_ele == 0) {
            continue;
        }
        char* tmp_out = buf + qry.rank_ptr[i] * this->itemBytes();
        if (this->server_->getRank() == i) {
            this->server_->pullLocal(qry.packs[i], tmp_out);
        } else {
            int tag_pull = query_pull ^ (qry.qid << 12) ^
                 (this->server_->getRank()<< 8) ^ (i << 4);
            server_->maskTag(tag_pull);
            uint64_t q[] = {query_pull, (uint64_t)tag_pull, qry.packs[i]};
            MPI_Send(q, 3, MPI_UNSIGNED_LONG_LONG, i, tag_query, MPI_COMM_WORLD);
            MPI_Recv(tmp_out, n_ele * this->itemBytes(), MPI_CHAR, i, tag_pull,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void HostEmbedding::pull(uint64_t qid, Tensor* out) {
    TIMER_START(pull);
    this->qry_mtx_.lock();
    auto qry = this->qrys_[qid];
    this->qry_mtx_.unlock();

    if (out) {
        qry.out = out;
    }
    auto out_ptr = (char*)qry.out->getPtr();

#ifdef DRESS_USE_UCWO
    // TODO: get some other worker
    this->server_->createChunks();
    MPI_Barrier(MPI_COMM_WORLD);
    auto w = this->server_->getWorker();
    for (size_t i = 0; i < qry.batch_size; ++i) {
        if (qry.idx[i] & flag_skip) {
            continue;
        }
        this->server_->pullRemote(w, qry.idx[i],
                out_ptr + i * this->itemBytes());
    }
    /*
    int rank = this->server_->getRank();
    this->server_->pullLocal(qry.packs[rank], qry.buf);
    for (int j = qry.rank_ptr[rank]; j < qry.rank_ptr[rank + 1]; ++j) {
        memcpy(out_ptr + qry.rank_ridx[j] * this->itemBytes(),
                qry.buf + (j - qry.rank_ptr[rank]) * this->itemBytes(),
                this->itemBytes());
    }
    */
    w->flush();
#else
    this->pullExchange(qry, qry.buf);
    int server_world_size = this->server_->getWorldSize();
#pragma omp parallel for num_threads(nth_embedding)
    for (int j = 0; j < qry.rank_ptr[server_world_size]; ++j) {
        memcpy(out_ptr + qry.rank_ridx[j] * this->itemBytes(),
                qry.buf + j * this->itemBytes(),
                this->itemBytes());
    }
#endif  // DRESS_USE_UCWO

    TIMER_END(pull);
    // PINFO << REPORT_TIME(pull) << "\n";
}

void HostEmbedding::release(uint64_t qid) {
    auto& qry = this->qrys_[qid];

    delete [] qry.idx;
    // delete [] qry.buf;
    delete [] qry.rank_ridx;
    this->qry_mtx_.lock();
    this->qrys_.erase(qry.qid);
    this->qry_mtx_.unlock();
}

void HostEmbedding::pushExchange(QueryEntry& qry, char* buf) {
    int server_world_size = this->server_->getWorldSize();
    for (int fi = 0, i; fi < server_world_size; ++fi) {
        i = (fi + this->server_->getRank()) % server_world_size;
        auto n_ele = qry.rank_ptr[i + 1] - qry.rank_ptr[i];
        if (n_ele == 0) {
            continue;
        }
        char* tmp_grad = buf + qry.rank_ptr[i] * this->itemBytes();
        if (this->server_->getRank() == i) {
            this->server_->pushLocal(qry.packs[i], tmp_grad);
        } else {
            int tag_push = query_push ^ (qry.qid << 12) ^
                (this->server_->getRank() << 8) ^ (i << 4);
            this->server_->maskTag(tag_push);
            uint64_t q[] = {query_push, (uint64_t)tag_push, qry.packs[i]};
            MPI_Send(q, 3, MPI_UNSIGNED_LONG_LONG, i, tag_query, MPI_COMM_WORLD);
            MPI_Send(tmp_grad, n_ele * this->itemBytes(), MPI_CHAR, i, tag_push,
                    MPI_COMM_WORLD);
        }
    }
    this->release(qry.qid);
}

void HostEmbedding::push(uint64_t qid, const Tensor& grad) {
    // TIMER_START(push);
    auto& qry = this->qrys_[qid];
    char* grad_ptr = (char*)grad.getPtr();

    int server_world_size = this->server_->getWorldSize();
#pragma omp parallel for num_threads(nth_embedding)
    for (int j = 0; j < qry.rank_ptr[server_world_size]; ++j) {
        memcpy(qry.buf + j * this->itemBytes(),
                grad_ptr + qry.rank_ridx[j] * this->itemBytes(),
                this->itemBytes());
    }
    this->pushExchange(qry, qry.buf);
    // TIMER_END(push);
    // PINFO << REPORT_TIME(push) << "\n";
}

void HostEmbedding::update() {
    this->server_->update(this->opt_);
}

void HostEmbedding::reshard() {
    throw "Not implemented";
}

size_t HostEmbedding::getBatchSize(uint64_t qid) {
    if (qrys_.find(qid) == qrys_.end()) {
        return -1ull;
    }
    return qrys_[qid].batch_size;
}

}; // namespace Dress
