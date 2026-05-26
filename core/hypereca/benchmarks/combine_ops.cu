#include "dress/combine_ops.h"

#include <cstdio>
#include <cstring>

#include <vector>

#include <chrono>

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
        std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}

#define timestamp(__var__) cudaDeviceSynchronize(); auto __var__ = std::chrono::system_clock::now();

#define local_gpu_count 8
#define n_tests 8

int main() {
    size_t batch_size, nnz, embedding_size;
    size_t slot_num_per_gpu_[local_gpu_count];
    int* row_offsets_[local_gpu_count];
    FILE* fdump = fopen("meta.txt", "r");
    fscanf(fdump, "%lu%lu%lu", &batch_size, &nnz, &embedding_size);
    for (int i = 0; i < local_gpu_count; ++i) {
        fscanf(fdump, "%lu", slot_num_per_gpu_ + i);
    }
    fclose(fdump);
    fdump = fopen("data.bin", "rb");
    for (int i = 0; i < local_gpu_count; ++i) {
        auto slot_num = slot_num_per_gpu_[i];
        row_offsets_[i] = new int[batch_size * slot_num + 1];
        fread(row_offsets_[i], sizeof(int), batch_size * slot_num + 1, fdump);
    }
    fclose(fdump);

    float* fake_input = new float[nnz * embedding_size];
    for (size_t i = 0; i < nnz * embedding_size; ++i) {
        fake_input[i] = (i % 1000) * .001;
    }

    float *out_gpu[local_gpu_count];
    for (int i = 0; i < local_gpu_count; ++i) {
        auto slot_num = slot_num_per_gpu_[i];
        auto out_size = batch_size * slot_num * embedding_size;
        cudaSetDevice(i);
        cudaMalloc(&out_gpu[i], sizeof(float) * out_size);
    }

    double dur = 0;
    for (int j = 0; j < n_tests; ++j) {
        size_t nnz_offset = 0;
        timestamp(ts);
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            auto numel = row_offsets_[i][batch_size * slot_num];
            auto out_size = batch_size * slot_num * embedding_size;

            float* out_cpu = new float[out_size];
            Dress::combineByCSRForward
                (batch_size * slot_num, embedding_size, 
                 Dress::INT32, row_offsets_[i],
                 Dress::FP32, fake_input + nnz_offset * embedding_size,
                 out_cpu, 1., 0.);
            cudaMemcpy(out_gpu[i], out_cpu, 
                    sizeof(float) * out_size, cudaMemcpyHostToDevice);
            delete [] out_cpu;
            nnz_offset += numel;
        }
        timestamp(te);
        auto durc = getDuration(ts, te);
        fprintf(stderr, "Run %d time %.3lf ms\n", j, durc * 1e3);
        if (j) {
            dur += durc;
        }
    }
    fprintf(stderr, "Mean time %.3lf ms\n", dur / (n_tests - 1) * 1e3);
}
