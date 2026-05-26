#define DRESS_USE_CUDA
#include "dress/tensor.h"
#include "dress/combine_ops.h"
#include "dress/dedup.h"

#define CUDA_SAFE_CHECK { \
    cudaError err = cudaGetLastError(); \
    if (cudaSuccess != err) { \
        fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
}

#include "common.h"

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

    fdump = fopen("keys.in", "r");
    size_t nnz_ck;
    fscanf(fdump, "%lu", &nnz_ck);
    assert(nnz_ck == nnz);
    int *vals_org = new int[nnz];
    for (size_t i = 0; i < nnz; ++i) {
        fscanf(fdump, "%d", vals_org + i);
    }

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
        timestamp(ts);

        CUDA_SAFE_CHECK;

        size_t nnz_offset = 0;
        vector<thread> thrs;
        for (int i = 0; i < local_gpu_count; ++i) {
            cudaSetDevice(i);
            auto slot_num = slot_num_per_gpu_[i];

            auto key_count = row_offsets_[i][batch_size * slot_num];
            fprintf(stderr, "Key count %d: %d, slot num %lu\n", i, key_count, slot_num);

            int *ptr_cuda;
            cudaMalloc(&ptr_cuda, (batch_size * slot_num + 1) * sizeof(int));
            cudaMemcpy(ptr_cuda, row_offsets_[i],
                    (batch_size * slot_num + 1) * sizeof(int),
                    cudaMemcpyHostToDevice);
            CUDA_SAFE_CHECK;
            auto key_tensor = Dress::Tensor(
                    vals_org + nnz_offset,
                    key_count * sizeof(int));
            auto dedupd = dedupKeys(key_tensor, Dress::CppTypeToDress<int>());
            thrs.push_back(thread([=](int i) mutable {
                cudaSetDevice(i);
                auto emb_tensor = Dress::Tensor(fake_input,
                        sizeof(float) * dedupd.nnz * embedding_size);
                auto emb_cuda = emb_tensor.to(Dress::CUDA, i);
                auto lookup_cuda = dedupd.lookup.to(Dress::CUDA, i);
                Dress::combineDedupedByCSRForward(batch_size * slot_num,
                        embedding_size,
                        Dress::CppTypeToDress<int>(),
                        ptr_cuda, 0,
                        dedupd.nnz, (size_t*)lookup_cuda.getPtr(),
                        Dress::CppTypeToDress<float>(),
                        emb_cuda.getPtr(),
                        out_gpu[i], 0);
                cudaStreamSynchronize(0);
                CUDA_SAFE_CHECK;
                Dress::combineDedupedByCSRBackward(batch_size * slot_num,
                        embedding_size,
                        Dress::CppTypeToDress<int>(),
                        ptr_cuda, 0,
                        dedupd.nnz, (size_t*)lookup_cuda.getPtr(),
                        Dress::CppTypeToDress<float>(),
                        out_gpu[i],
                        emb_cuda.getPtr(),
                        0);
                cudaStreamSynchronize(0);
                CUDA_SAFE_CHECK;
                emb_cuda.free();
                lookup_cuda.free();
                dedupd.lookup.free();
                dedupd.val.free();
                cudaFree(ptr_cuda);
            }, i));
            nnz_offset += key_count;
        }
        for (auto &thr: thrs) {
            thr.join();
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
