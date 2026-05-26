#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <unordered_map>

#include <cstdlib>
#include <ctime>
#include "dress/embedding.h"
#include "dress/logging.h"

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

using namespace Dress;

TEST(HOST_EMB_PULL, pullFromHostEmbedding) {
    // srand(time(0));
    const size_t embsz = 4;
    EmbeddingOption emb_opt;
    emb_opt.type = EmbeddingOption::Host;
    emb_opt.item_sz = embsz;
    emb_opt.ktype = UINT64;
    std::shared_ptr<Embedding> emb(Embedding::create(emb_opt));
    const size_t n_pulls = 100;
    const size_t n_bs = 1000;

    std::unordered_map<size_t, float*> embr;

    std::vector<void*> ptrs;
    int rank = 0;
#ifdef DRESS_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif  // DRESS_USE_MPI
    srand(114514 + rank);

    for (size_t i_pull = 0; i_pull < n_pulls; ++i_pull) {
        size_t bs = rand() % n_bs + 10;
        size_t n_ele = rand() % n_bs + 10;
        auto idx = Tensor::create(bs, UINT64);
        auto out = Tensor::create(bs * embsz, FP32);
        ptrs.push_back(idx.getPtr());
        ptrs.push_back(out.getPtr());

        auto idx_ptr = (size_t*)idx.getPtr();
        for (size_t i = 0; i < bs; ++i) {
            idx_ptr[i] = rand() % n_ele;
        }
        auto uid = emb->pullTo(idx, bs, out);
        size_t n_chked = 0;
        for (size_t i = 0; i < bs; ++i) {
            auto data_ptr = (float*)out.getPtr() + i * embsz;
            /*
            for (size_t j = 0; j < embsz; ++j) {
                fprintf(stderr, "%8.2f", data_ptr[j]);
            }
            fprintf(stderr, "\n");
            */
            if (embr.find(idx_ptr[i]) == embr.end()) {
                embr[idx_ptr[i]] = data_ptr;
            } else {
                auto ref_ptr = embr[idx_ptr[i]];
                for (size_t j = 0; j < embsz; ++j) {
                    ASSERT_EQ(data_ptr[j], ref_ptr[j])
                        << " Rank " << rank
                        << " Different result at pull " << i_pull
                        << " sample " << i << " / " << idx_ptr[i];
                }
                n_chked += 1;
            }
        }
#ifdef DRESS_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        PINFO << " Rank " << rank << " Pull " << i_pull << " checked " <<
            n_chked << " / " << bs << std::endl;
#else
        PINFO << "Pull " << i_pull << " checked " <<
            n_chked << " / " << bs << std::endl;
#endif  // DRESS_USE_MPI
    }
    for (auto& ptr: ptrs) {
        delete [] ptr;
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

