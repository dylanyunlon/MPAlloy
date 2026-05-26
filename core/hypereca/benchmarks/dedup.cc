#include "dress/dedup.h"

#include "common.h"

#define local_gpu_count 8
#define n_tests 9

int main() {
    fprintf(stderr, "Starting\n");
    size_t batch_size;
    int* keys;
    FILE* fdump = fopen("keys.in", "r");
    fscanf(fdump, "%lu", &batch_size);
    keys = new int[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        fscanf(fdump, "%d", keys + i);
    }
    fclose(fdump);

    auto chunk_size = batch_size / 8;

    double dur = 0;
    double bwa = 0;
    int bwc = 0;
    for (int j = 0; j < n_tests; ++j) {
        auto nnz = 0;
        timestamp(ts);
        vector<thread> ths;
        for (int k = 0; k < 8; ++k) {
            auto keys_tensor = Dress::Tensor(keys + k * chunk_size, chunk_size * sizeof(int));
            timestamp(tx);
            auto out = dedupKeys(keys_tensor, Dress::CppTypeToDress<int>());
            timestamp(ty);
            auto dur = getDuration(tx, ty);
            fprintf(stderr, "Dedup %d nnz %lu -> %lu (%.2f%%) time %.3lf ms %.3lf gkps\n",
                    k, chunk_size, out.nnz, (float)out.nnz / chunk_size * 1e2,
                    dur * 1e3, chunk_size / dur * 1e-9);
            /*
            ths.push_back(thread([&](int k) {
                // timmestamp(t0);
                auto ov = out.val.to(Dress::CUDA, k);
                auto ol = out.lookup.to(Dress::CUDA, k);
                // timestamp(t1);
                // fprintf(stderr, "Copy %d time %.3lf ms\n", k, getDuration(t0, t1) * 1e3);
            }, k));
            */
        }
        for (auto& th: ths) {
            th.join();
        }
        timestamp(te);
        timestamp(tq);
        auto keys_tensor = Dress::Tensor(keys, batch_size * sizeof(int));
        auto out = dedupKeys(keys_tensor, Dress::CppTypeToDress<int>());
        timestamp(tp);
        // auto ov = out.val.to(Dress::CUDA);
        nnz = out.nnz;
        auto durc = getDuration(ts, te);
        fprintf(stderr, "Dedup all nnz %lu -> %lu (%.2f%%) time %.3lf ms %.3lf gkps\n",
                batch_size, out.nnz, (float)out.nnz / batch_size * 1e2,
                durc * 1e3, batch_size / durc * 1e-9);
        bwa += batch_size / durc, bwc += 1;
        if (j) {
            dur += durc;
        }
    }
    fprintf(stderr, "Mean time %.3lf ms, bw %.3lf mkps\n",
            dur / (n_tests - 1) * 1e3,
            bwa / bwc * 1e-6);
}
