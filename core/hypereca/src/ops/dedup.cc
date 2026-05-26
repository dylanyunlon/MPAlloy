#include <dress/profiler.h>
#include <dress/dedup.h>

#include "dedup_impl.hh"

namespace Dress {

static const auto n_thr = cfgi()->getInt("nth_prefetch");
static const auto use_sort = (cfgi()->getString("dedup_algo") == "sort");

DedupOutput dedupKeys(const Tensor& keys, DType key_t) {
    DedupOutput res;
    DISPATCH_INT(key_t, KeyType, ([&] {
        if (n_thr > 1) {
            res = dedupKeysMultiThreadImpl<KeyType>(keys, n_thr);
        } else if (use_sort) {
            res = dedupKeysSortImpl<KeyType>(keys);
        } else {
            res = dedupKeysMapImpl<KeyType>(keys);
        }
    }));
    return res;
}

DedupOutput noDedupKeys(const Tensor& keys, DType key_t) {
    DedupOutput out;
    DISPATCH_INT(key_t, KeyType, ([&] {
        out.nnz = keys.size() / sizeof(KeyType);
        out.val = keys;
        out.lookup = Tensor::create(out.nnz, UINT64);
        size_t* p = (size_t*)out.lookup.getPtr();
        for (size_t i = 0; i < out.nnz; ++i) {
            p[i] = i;
        }
    }));
    return out;
}

};  // namespace Dress
