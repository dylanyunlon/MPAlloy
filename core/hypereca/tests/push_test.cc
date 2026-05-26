#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include <cstdlib>
#include <ctime>
#include "dress/embedding.h"
#include "dress/logging.h"
#include "dress/optimizer.h"

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

using namespace Dress;

TEST(HOST_EMB_PUSH, pushToHostEmbedding) {
    // srand(time(0));
    const size_t embsz = 64;
    const float lr = .1;
    EmbeddingOption emb_opt;
    emb_opt.item_sz = embsz;
    std::shared_ptr<Embedding> emb(Embedding::create(emb_opt));
    std::shared_ptr<Optimizer> opt(Optimizer::create(
                OptimizerOption(OptimizerOption::SGD, lr)));

    emb->setOptimizer(opt.get());
    const size_t n_pushs = 10;
    const size_t n_bs = 10000;

    std::unordered_map<size_t, float*> embr;

    int rank = 0, world_size = 1;
#ifdef DRESS_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif  // DRESS_USE_MPI
    srand(114514 + rank);

    for (size_t i_push = 0; i_push < n_pushs; ++i_push) {
        // PINFO << "Push " << i_push << std::endl;
        size_t bs = rand() % n_bs + 10;
        size_t n_ele = rand() % n_bs + 10;
        auto idx = new int[bs];
        for (size_t i = 0; i < bs; ++i) {
            idx[i] = rand() % n_ele;
        }
        std::sort(idx, idx + bs);
        bs = std::unique(idx, idx + bs) - idx;
        auto idx_tensor = Tensor(idx, bs * sizeof(int));
        auto pout = emb->pullOut(idx_tensor, bs);
        auto out = pout.first;

#ifdef DEBUG_
        PINFO << "Pull " <<  i_push << " results\n";
        for (size_t i = 0; i < bs; ++i) {
            auto data_ptr = (float*)out.getPtr() + i * embsz;
            std::cerr << idx[i] << ":\t";
            for (size_t j = 0; j < embsz; ++j) {
                std::cerr << data_ptr[j] << "\t";
            }
            std::cerr << "\n";
        }
#endif  // DEBUG_

        size_t n_checked = 0;
        for (size_t i = 0; i < bs; ++i) {
            auto data_ptr = (float*)out.getPtr() + i * embsz;
            if (embr.find(idx[i]) == embr.end()) {
                embr[idx[i]] = new float[embsz];
                memcpy(embr[idx[i]], data_ptr, sizeof(float) * embsz);
            } else {
                auto ref_ptr = embr[idx[i]];
                for (size_t j = 0; j < embsz; ++j) {
                    if (std::abs(data_ptr[j] - ref_ptr[j] < 1e-5)) {
                        continue;
                    }
                    ASSERT_EQ(data_ptr[j], ref_ptr[j])
                        << " Rank " << rank
                        << " Different result at pull " << i_push
                        << " sample " << i << " / " << idx[i];
                }
                ++n_checked;
            }
        }

        float* grads = new float[bs * embsz];
        for (size_t i = 0; i < bs * embsz; ++i) {
            grads[i] = (rand() % 201 - 100) / 100.0;
        }
        auto grad_tensor = Tensor(grads, bs * embsz * sizeof(float));

#ifdef DEBUG_
        PINFO << "Grads " <<  i_push << " \n";
        for (size_t i = 0; i < bs; ++i) {
            auto data_ptr = grads + i * embsz;
            std::cerr << idx[i] << ":\t";
            for (size_t j = 0; j < embsz; ++j) {
                std::cerr << data_ptr[j] << "\t";
            }
            std::cerr << "\n";
        }
#endif  // DEBUG_

        emb->push(pout.second, grad_tensor);

#ifdef DRESS_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif  // DRESS_USE_MPI

        emb->update();

        for (int ti = 0; ti < world_size; ++ti) {
            for (size_t i = 0; i < bs; ++i) {
                if (embr.find(idx[i]) == embr.end()) {
                    continue;
                }
                auto param_ptr = embr[idx[i]];
                for (size_t j = 0; j < embsz; ++j) {
                    param_ptr[j] -= grads[i * embsz + j] * lr;
                }
            }
#ifdef DRESS_USE_MPI
            if (ti + 1 < world_size) {
                size_t next_bs;
                MPI_Sendrecv(&bs, 1, MPI_UNSIGNED_LONG_LONG,
                        (rank + 1) % world_size, 111,
                        &next_bs, 1, MPI_UNSIGNED_LONG_LONG,
                        (rank - 1 + world_size) % world_size, 111,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int* next_idx = new int[next_bs];
                MPI_Sendrecv(idx, bs, MPI_INT,
                        (rank + 1) % world_size, 222,
                        next_idx, next_bs, MPI_INT,
                        (rank - 1 + world_size) % world_size, 222,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                float* next_grads = new float[next_bs * embsz];
                MPI_Sendrecv(grads, bs * embsz, MPI_FLOAT,
                        (rank + 1) % world_size, 333,
                        next_grads, next_bs * embsz, MPI_FLOAT,
                        (rank - 1 + world_size) % world_size, 333,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                delete [] idx;
                delete [] grads;
                bs = next_bs;
                idx = next_idx;
                grads = next_grads;
            }
        }
#endif  // DRESS_USE_MPI
        delete [] idx;

#ifdef DEBUG_
        PINFO << "Updated " <<  i_push << " locals\n";
        for (auto& i: embr) {
            std::cerr << i.first << ":\t";
            for (size_t j = 0; j < embsz; ++j) {
                std::cerr << i.second[j] << "\t";
            }
            std::cerr << "\n";
        }
#endif  // DEBUG_
       delete [] grads;
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();
#ifdef DRESS_USE_MPI
    MPI_Finalize();
#endif  // DRESS_USE_MPI
    return res;
}

