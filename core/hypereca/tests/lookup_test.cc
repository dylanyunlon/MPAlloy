#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <unordered_map>

#include <cstdlib>
#include <ctime>
#include "../src/embeddings/host_embedding.h"

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

using namespace Dress;

class LookupEmbedding: public HostEmbedding {
public:
    LookupEmbedding(DType ktype):
        HostEmbedding(1, ktype, FP32) {};

    void globalLookupPub(const Tensor& uid, QueryEntry& qry) {
        this->globalLookup(uid, qry);
    }

    int getRank() {
        return this->server_->getRank();
    }
};

TEST(GLOBAL_LOOKUP, globalLookup) {
    std::shared_ptr<LookupEmbedding> emb(new LookupEmbedding(UINT64));
    const size_t n_lkups = 100;
    const size_t n_bs = 10000;

    int rank = 0;
#ifdef DRESS_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif  // DRESS_USE_MPI
    srand(114514 + rank);

    std::unordered_map<size_t, size_t> embi;

    for (size_t i_lkup = 0; i_lkup < n_lkups; ++i_lkup) {
        size_t bs = rand() % n_bs + 10;
        size_t n_ele = rand() % n_bs + 10;
        auto idx = Tensor::create(bs, UINT64);

        auto idx_ptr = (size_t*)idx.getPtr();
        for (size_t i = 0; i < bs; ++i) {
            idx_ptr[i] = (size_t)rand() % n_ele;
        }

        QueryEntry q;
        q.batch_size = bs;
        q.idx = new uint64_t[bs];
        emb->globalLookupPub(idx, q);

        for (size_t i = 0; i < bs; ++i) {
            if (embi.find(idx_ptr[i]) == embi.end()) {
                embi[idx_ptr[i]] = q.idx[i];
            } else {
                ASSERT_EQ(embi[idx_ptr[i]], q.idx[i])
                    << " Different result at rank " << emb->getRank()
                    << " lookup " << i_lkup
                    << " sample " << i << " / " << idx_ptr[i];
            }
        }
    }
#ifdef DRESS_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif  // DRESS_USE_MPI
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

