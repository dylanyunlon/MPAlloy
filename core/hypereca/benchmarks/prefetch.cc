#include "dress/dedup.h"
#include "../src/embeddings/host_embedding.h"
#include "../src/optimizers/sgd.h"
#include "common.h"

#include <omp.h>
#include <mpi.h>
#include <shared_mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <unistd.h>

#define local_gpu_count 8
#define n_tests 9


int main() {
    fprintf(stderr, "Starting\n");
    size_t batch_size;
    int* keys;
    FILE* fdump = fopen("keys.in", "r");
    fscanf(fdump, "%lu", &batch_size);
    // batch_size = 16;
    keys = new int[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        fscanf(fdump, "%d", keys + i);
    }
    fclose(fdump);

    auto chunk_size = 32768 / 8;
    int nth = 48;
    assert(nth * chunk_size <= batch_size);

    int rank, world_size;

    double tott = 0.;

    std::shared_mutex mtx;
    mtx.lock();

    Dress::HostEmbedding emb(64);
    Dress::SGD sgd(Dress::FP32, 1.0);
    emb.setOptimizer(&sgd);

    std::atomic_int cnt_ready;

    std::vector<std::thread> th;
    int test_j = -1;
    for (int i = 0; i < nth; ++i) {
        th.push_back(std::thread([&](int rank) {
            for (int j = 0; j < n_tests; ++j) {
                auto keys_tensor = Dress::Tensor(keys + rank * chunk_size
                        + j * nth * chunk_size,
                        chunk_size * sizeof(int));
                while (test_j != j) {
                    std::this_thread::yield();
                }
                auto out = dedupKeys(keys_tensor, Dress::CppTypeToDress<int>());
                auto qid = emb.prefetch(out.val, out.nnz, 0);
                ++cnt_ready;
                // fprintf(stderr, "Rank %d test %d\n", rank, j);
            }
        }, i));
    }

    double dur = 0;
    for (int j = 0; j < n_tests; ++j) {
        cnt_ready = 0;
        test_j = j;
        timestamp(tpf);
        while (cnt_ready != nth) {
            usleep(100);
        }
        timestamp(tpe);
        double t = getDuration(tpf, tpe);
        fprintf(stderr, "Test %d time %.3lf ms\n", j, t * 1e3);
        if (j) {
            tott += t;
        }
    }
    fprintf(stderr, "Mean time %.3lf ms\n", tott * 1e3 / (n_tests - 1));
    for (auto& t: th) {
        t.join();
    }
}
