#include "dress/dedup.h"
#include "../src/embeddings/host_embedding.h"
#include "../src/optimizers/sgd.h"
#include "common.h"

#include <mpi.h>

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

    auto chunk_size = batch_size / 8;

    int rank, world_size;

    {
        Dress::HostEmbedding emb(64);
        Dress::SGD sgd(Dress::FP32, 1.0);
        emb.setOptimizer(&sgd);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        fprintf(stderr, "Rank %d/%d initialized\n", rank, world_size);

        double dur = 0;
        for (int j = 0; j < n_tests; ++j) {
            auto nnz = 0;
            std::vector<timestamp_t> tss;
            double durs[10] = {0}, duro[10];
            for (int k = rank; k < 8; k += world_size) {
                tss.clear();
                MPI_Barrier(MPI_COMM_WORLD);
                auto keys_tensor = Dress::Tensor(keys + k * chunk_size, chunk_size * sizeof(int));
                SET_TS(tss);
                auto out = dedupKeys(keys_tensor, Dress::CppTypeToDress<int>());
                Dress::Tensor t = Dress::Tensor::create(out.nnz * 64, Dress::FP32);
                SET_TS(tss);
                auto qid = emb.prefetch(out.val, out.nnz, &t);
                SET_TS(tss);
                emb.pull(qid);
                SET_TS(tss);
                emb.push(qid, t);
                MPI_Barrier(MPI_COMM_WORLD);
                SET_TS(tss);

                nnz += out.nnz;
                size_t n_ts = tss.size();
                for (size_t i = 0; i + 1 < n_ts; ++i) {
                    durs[i] += getDuration(tss[i], tss[i + 1]);
                }
            }
            timestamp(tupd0);
            emb.update();
            timestamp(tupd1);

            size_t n_ts = tss.size();
            MPI_Reduce(durs, duro, n_ts, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                fprintf(stderr, "Rank %d: Run %d time ", rank, j);
                for (size_t i = 0; i + 1 < n_ts; ++i) {
                    fprintf(stderr, "%9.3lf", duro[i] * 1e3);
                }
                fprintf(stderr, " update %9.3lf ms", getDuration(tupd0, tupd1) * 1e3);
                fprintf(stderr, "\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // fprintf(stderr, "Rank %d: Mean time %.3lf ms\n", rank, dur / (n_tests - 1) * 1e3);
    }
}
