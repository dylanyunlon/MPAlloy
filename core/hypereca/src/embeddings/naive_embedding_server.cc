#include "naive_embedding_server.h"

#include <unistd.h>

namespace Dress {
void NaiveEmbeddingServer::localLookup(Tensor uid, size_t batch_size,
        uint64_t* idx, int preferred_rank) {
    DISPATCH_INT(this->ktype_, KeyType, ([&]() {
        this->localLookup<KeyType>((KeyType*)uid.getPtr(), batch_size,
                idx, preferred_rank);
    }));
}

template<typename KeyType>
void NaiveEmbeddingServer::localLookup(KeyType* key_ptr, size_t batch_size,
        uint64_t* idx, int preferred_rank) {
    size_t count_not_found = 0;
// #pragma omp parallel for reduction(+:count_not_found) num_threads(nth_prefetch)
    std::vector<size_t> vnf;
    for (size_t i = 0; i < batch_size; ++i) {
        key_t uid = key_ptr[i];
        auto map_uid = (uid & this->map_mask_) / this->world_size;
        auto& mp = this->maps_[map_uid];
        mp.mtx.lock_shared();
        auto it = mp.map.find(uid);
        if (it != mp.map.end()) {
            idx[i] = it->second;
        } else {
            count_not_found += 1;
            vnf.push_back(i);
        }
        mp.mtx.unlock_shared();
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
    for (auto& i: vnf) {
        key_t uid = key_ptr[i];
        auto map_uid = (uid & this->map_mask_) / this->world_size;
        auto& mp = this->maps_[map_uid];
        mp.mtx.lock();
        auto it = mp.map.find(uid);
        if (it != mp.map.end()) {
            idx[i] = it->second;
        } else {
            mp.map[uid] = idx[i] = this->ptr2global(ptr++, tgt_rank);
        }
        mp.mtx.unlock();
    }
}
};  // namespace Dress
