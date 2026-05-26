#include <thread>
#include <condition_variable>
#include <vector>
#include <mpi.h>
#include <unistd.h>

#include "hugectr/hugectr_utils.hh"
#include "dress/embedding.h"
#include "dress/dedup.h"
#include "common.h"

using namespace HugeCTR;

int nth = 48;

const int slot_num = 16;
const int batch_size = 131072;
const int dense_dim = 1;
const int local_gpu_count = 8;
const int n_iter = 500;

int rank, world_size;

Dress::Embedding* emb;

void init() {
    char* nth_str = getenv("NTH");
    if (nth_str) {
        nth = atoi(nth_str);
    }

    Dress::EmbeddingOption emb_opt;
    emb_opt.type = Dress::EmbeddingOption::Hypra;
    emb_opt.item_sz = 64;
    emb_opt.ktype = Dress::INT32;
    emb_opt.dtype = Dress::FP32;
    emb = Dress::Embedding::create(emb_opt);

    Dress::OptimizerOption opt_opt;
    opt_opt.lr = 0.;
    opt_opt.dtype = Dress::FP32;
    opt_opt.type = Dress::OptimizerOption::SGD;
    emb->setOptimizer(Dress::Optimizer::create(opt_opt));
}

uint64_t prefetchToDress(int* keys, int nnz) {
    auto key_tensor = Dress::Tensor(keys, nnz * sizeof(int)); 
    auto dedupd = Dress::dedupKeys(key_tensor, Dress::INT32);

    auto lookup_ptr = (uint64_t*)dedupd.lookup.getPtr();
    std::copy(lookup_ptr, lookup_ptr + nnz, keys);
    dedupd.lookup.free();
    auto qid = emb->prefetch(dedupd.val, dedupd.nnz);
    dedupd.val.free();
    return qid;
}

int main(int argc, char* args[]) {
    int flag;
    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &flag);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    init();
    if (argc != 2) {
        fprintf(stderr, "No list file specified. Exiting.\n");
        return 0;
    }

    auto listfile = args[1];
    std::vector<std::thread> ths;
    std::vector<std::condition_variable> cv_start(nth), cv_fin(nth);
    std::vector<int> rd_cnt(nth);
    std::vector<int> proc_wait(nth);
    std::vector<std::mutex> mtx(nth);
    for (int i = 0; i < nth; ++i) {
        ths.emplace_back([&](int i) {
            auto source = new HugeCTR::FileSource(i, nth, listfile);
            auto checker = new HugeCTR::CheckSum(*source);
            auto reader = new HugeCTR::DataReader(checker, slot_num,
                    batch_size, dense_dim);
            reader->local_gpu_count = local_gpu_count;
            reader->total_gpu_count = local_gpu_count * world_size;
            reader->pid = rank;
            for (int j = 0; j * nth < n_iter; ++j) {
                std::unique_lock<std::mutex> lk(mtx[i]);
                cv_start[i].wait(lk, [&]() { return proc_wait[i] >= j; });
                // fprintf(stderr, "Thread %d read %d start\n", i, j);
                timestamp(br);
                reader->read_a_batch();
                timestamp(pr);
                rd_cnt[i] = j + 1;
                lk.unlock();

                cv_fin[i].notify_all();
                // fprintf(stderr, "Thread %d read %d time %.3lf ms\n", i, j,
                //         getDuration(br, pr) * 1e3);
            }
        }, i);
    }

    double tott = 0;
    for (int i = 0; i < n_iter; ++i) {
        timestamp(t0);
        int thrank = i % nth;
        {
            std::unique_lock<std::mutex> lk(mtx[thrank]);
            cv_fin[thrank].wait(lk, [&]() {
                return rd_cnt[thrank] > i / nth;
            });
        }
        timestamp(t1);
        double dur = getDuration(t0, t1);
        if (i) {
            tott += dur;
        }
        if (rank == 0) {
            printf("Iter %d time %.3lf ms\n", i, dur * 1e3);
        }
        if (dur < 0.01) {
            usleep((0.01 - dur) * 1e6);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        {
            std::lock_guard<std::mutex> lk(mtx[thrank]);
            proc_wait[thrank] = i / nth + 1;
        }
        cv_start[thrank].notify_all();
    }

    printf("Rank %d mean time %.3lf ms\n", rank, tott / (n_iter - 1) * 1e3);

    for (auto& th: ths) {
        th.join();
    }
    MPI_Finalize();
}
