#include <dress/spec.h>
#include <dress/logging.h>

#include "hash_embedding.h"

namespace Dress {

static const auto nth_prefetch = cfgi()->getInt("nth_prefetch");
static const auto nth_embedding = cfgi()->getInt("nth_embedding");
static const auto nth_optimizer = cfgi()->getInt("nth_optimizer");


uint64_t HashEmbedding::prefetch(Tensor uid, size_t batch_size, Tensor* out) {
    // this->server_->upd_lck.lock_shared();
    QueryEntry qry;
    qry.qid = ++this->qry_ptr_;
    qry.batch_size = batch_size;
    qry.idx = new uint64_t[batch_size];
    qry.buf = new char[batch_size * this->itemBytes()];
    qry.out = out;

    DISPATCH_INT(this->ktype_, KeyType, ([&]() {
        auto key_ptr = (const KeyType*)uid.getPtr();
        for (size_t i = 0; i < batch_size; ++i) {
            qry.idx[i] = key_ptr[i] % 262144;
        }
    }));

    std::lock_guard<std::mutex> lock(this->qry_mtx_);
    qrys_[qry.qid] = qry;
    return qry.qid;
}

}; // namespace Dress
