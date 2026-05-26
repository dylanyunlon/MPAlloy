#include <cstdio>
#include <cstring>

#include <vector>

#include <thread>
#include <chrono>
#include <nccl.h>
#include <omp.h>

using namespace std;

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
        std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}

#define timestamp(__var__) cudaDeviceSynchronize(); auto __var__ = std::chrono::system_clock::now();

#define local_gpu_count 4
#define n_tests 16

void cpbf(size_t batch_size, size_t *x_host, size_t **out_gpu, cudaStream_t* streams) {
#pragma omp parallel for
    for (int i = 0; i < local_gpu_count; ++i) {
        // cudaMemcpyAsync(out_gpu[i], x_host, sizeof(size_t) * batch_size,
                // cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(out_gpu[i], x_host, sizeof(size_t) * batch_size,
                cudaMemcpyHostToDevice);
    }
    /* for (int i = 0; i < local_gpu_count; ++i) {
        cudaStreamSynchronize(streams[i]);
    } */
}

void cpbc(size_t batch_size, size_t *x_host, size_t **out_gpu,
        cudaStream_t* streams, ncclComm_t* comms) {
#pragma omp parallel num_threads(local_gpu_count)
    {
        int rank = omp_get_thread_num();
        auto chunk_size = batch_size / local_gpu_count;
        // cudaSetDevice(rank);
        cudaMemcpy(out_gpu, x_host + chunk_size * rank, 
                sizeof(size_t) * chunk_size,
                cudaMemcpyHostToDevice);
#pragma omp barrier
        ncclAllGather(out_gpu + rank * chunk_size, out_gpu, 
                chunk_size, ncclUint64, comms[rank], streams[rank]);
        // cudaStreamSynchronize(streams[rank]);
        // for (int i = 0; i < local_gpu_count; ++i) {
        //     bcths.push_back(thread(cpth, 
        //             i, batch_size, x_host, out_gpu[i], streams[i], comms[i]));
        // }
    }
    for (int i = 0; i < local_gpu_count; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
}

void ncclInit(ncclComm_t* comm_ptr, int rank, ncclUniqueId uid) {
    cudaSetDevice(rank);
    ncclCommInitRank(comm_ptr, local_gpu_count, uid, rank);
}

int main() {
    size_t batch_size = 1 << 18;
    size_t *x_host = new size_t[batch_size];
    // cudaMallocManaged(&x_host, sizeof(size_t) * batch_size);
    for (int i = 0; i < batch_size; ++i) {
        x_host[i] = i * i;
    }

    size_t **out_gpu = new size_t*[local_gpu_count];
    ncclComm_t comms[local_gpu_count];
    cudaStream_t streams[local_gpu_count];
    
    vector<thread> init_ths;
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);
    for (int i = 0; i < local_gpu_count; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&out_gpu[i], sizeof(size_t) * batch_size);
        cudaStreamCreate(streams + i);
        init_ths.push_back(thread(ncclInit, comms + i, i, uid));
    }
    for (int i = 0; i < local_gpu_count; ++i) {
        init_ths[i].join();
    }

    fprintf(stderr, "Copy by brute force\n");
    double dur = 0;
    for (int j = 0; j < n_tests; ++j) {
        timestamp(ts);
        cpbf(batch_size, x_host, out_gpu, streams);
        timestamp(te);
        auto durc = getDuration(ts, te);
        fprintf(stderr, "Run %d time %.3lf ms\n", j, durc * 1e3);
        if (j > 3) {
            dur += durc;
        }
    }
    auto bw = batch_size * sizeof(size_t) * (n_tests - 4) / dur * 1e-9;
    fprintf(stderr, "Mean time %.3lf ms, bw %.3lf GBps\n", dur / (n_tests - 1) * 1e3, bw);

    fprintf(stderr, "Copy with bcast\n");
    dur = 0;
    for (int j = 0; j < n_tests; ++j) {
        timestamp(ts);
        cpbc(batch_size, x_host, out_gpu, streams, comms);
        timestamp(te);
        auto durc = getDuration(ts, te);
        fprintf(stderr, "Run %d time %.3lf ms\n", j, durc * 1e3);
        if (j) {
            dur += durc;
        }
    }
    bw = batch_size * sizeof(size_t) * (n_tests - 1) / dur * 1e-9;
    fprintf(stderr, "Mean time %.3lf ms, bw %.3lf GBps\n", dur / (n_tests - 1) * 1e3, bw);

    size_t *x_res = new size_t[100];
    cudaMemcpy(x_res, out_gpu[3], 100 * sizeof(size_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; ++i) {
        printf("%llu ", x_res[i]);
    }
    putchar(10);
}

