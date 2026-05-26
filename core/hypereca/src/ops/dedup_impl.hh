#include <dress/logging.h>
#include <dress/dedup.h>
#include <dress/config.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <omp.h>


namespace Dress {
template<typename KeyType>
DedupOutput dedupKeysMultiThreadImpl(const Tensor& keys, int n_thr) {
    size_t n = keys.size() / sizeof(KeyType);
    size_t chunk_size = (n - 1) / n_thr + 1;

    std::vector<std::vector<std::unordered_set<KeyType>>> par_keys;
    std::vector<std::unordered_map<KeyType, size_t>> rev_mapping;
    par_keys.resize(n_thr);
    rev_mapping.resize(n_thr);
    std::vector<size_t > offsets;
    offsets.resize(n_thr);

    KeyType* keys_ptr = (KeyType*)keys.getPtr();

    std::vector<std::vector<double>> tss;
    tss.resize(n_thr);
    std::vector<std::vector<size_t>> cnts;
    cnts.resize(n_thr);
    TIMER_START(t0);

    DedupOutput out;
#pragma omp parallel num_threads(n_thr)
    {
        int rank = omp_get_thread_num();
        par_keys[rank].resize(n_thr);
        auto chunk_begin = rank * chunk_size;
        auto chunk_end = std::min(chunk_begin + chunk_size, n);
        for (auto i = chunk_begin; i < chunk_end; ++i) {
            KeyType k = keys_ptr[i];
            KeyType key_idx = k % n_thr;
            par_keys[rank][key_idx].insert(k);
        }
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        auto sz_estm = par_keys[0][rank].size() * n_thr / 2;
        size_t count = 0;
        rev_mapping[rank] = std::unordered_map<KeyType, size_t>(sz_estm);
        for (int i = 0; i < n_thr; ++i) {
            for (auto& v: par_keys[i][rank]) {
                if (rev_mapping[rank].find(v) == rev_mapping[rank].end()) {
                    rev_mapping[rank][v] = count++;
                }
            }
        }
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        size_t offset = 0;
        for (int i = 0; i < rank; ++i) {
            offset += rev_mapping[i].size();
        }
        offsets[rank] = offset;
        cnts[rank].push_back(rev_mapping[rank].bucket_count());
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        if (rank == n_thr - 1) {
            out.nnz = offset + count;
            out.val = Tensor::create(out.nnz, CppTypeToDress<KeyType>());
        }
        if (rank == 0) {
            out.lookup = Tensor::create(n, UINT64);
        }
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        auto out_val_ptr = (KeyType*)out.val.getPtr();
        for (auto& v: rev_mapping[rank]) {
            out_val_ptr[offset + v.second] = v.first;
        }
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        par_keys[rank].clear();
        auto lookup_ptr = (size_t*)out.lookup.getPtr();
        for (auto i = chunk_begin; i < chunk_end; ++i) {
            KeyType k = keys_ptr[i];
            size_t key_idx = k % n_thr;
            lookup_ptr[i] = rev_mapping[key_idx][k] + offsets[key_idx];
        }
        tss[rank].push_back(TIMER_READ(t0));
#pragma omp barrier
        cnts[rank].push_back(rev_mapping[rank].size());
        rev_mapping[rank].clear();
    }
    /*
    for (size_t i = 0; i < tss[0].size(); ++i) {
        for (size_t j = 0; j < tss.size(); ++j) {
            std::cerr << std::setw(9) << std::setprecision(5) << tss[j][i] * 1e3;
        }
        std::cerr << std::endl;
    }
    for (size_t i = 0; i < cnts[0].size(); ++i) {
        for (size_t j = 0; j < cnts.size(); ++j) {
            std::cerr << std::setw(9) << cnts[j][i];
        }
        std::cerr << std::endl;
    }
    */
    return out;
}

template<typename KeyType>
DedupOutput dedupKeysMapImpl(const Tensor& keys) {
    size_t n = keys.size() / sizeof(KeyType);
    auto a = (KeyType*)keys.getPtr();
    std::unordered_map<KeyType, size_t> m;
    for (size_t i = 0; i < n; ++i) {
        if (m.find(a[i]) == m.end()) {
            m[a[i]] = m.size();
        }
    }
    DedupOutput out;
    out.nnz = m.size();
    out.val = Tensor::create(out.nnz, CppTypeToDress<KeyType>());
    out.lookup = Tensor::create(n, UINT64);
    auto val_ptr = (KeyType*)out.val.getPtr();
    for (auto& it: m) {
        val_ptr[it.second] = it.first;
    }
    auto lookup_ptr = (size_t*)out.lookup.getPtr();
    for (size_t i = 0; i < n; ++i) {
        lookup_ptr[i] = m[a[i]];
    }
    return out;
}

template<typename KeyType>
DedupOutput dedupKeysSortImpl(const Tensor& keys) {
    size_t n = keys.size() / sizeof(KeyType);
    auto a = (KeyType*)keys.getPtr();
    auto* p = new size_t[n];
    for (size_t i = 0; i < n; ++i) {
        p[i] = i;
    }
    std::sort(p, p + n, [&](const int& x, const int& y) {
        return a[x] < a[y];
    });
    DedupOutput out;
    out.val = Tensor::create(n, CppTypeToDress<KeyType>());
    out.lookup = Tensor(p, n * sizeof(size_t));

    auto val_ptr = (KeyType*)out.val.getPtr();
    val_ptr[out.nnz = 0] = a[p[0]];
    p[0] = 0;
    for (size_t i = 1; i < n; ++i) {
        if (a[p[i]] != val_ptr[out.nnz]) {
            val_ptr[++out.nnz] = a[p[i]];
        }
        p[i] = out.nnz;
    }
    ++out.nnz;

    return out;
}

};  // namespace Dress
