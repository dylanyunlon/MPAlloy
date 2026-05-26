#pragma once

#include <stdint.h>
#include <dress/spec.h>
#include <dress/logging.h>
#include "HugeCTR/include/csr_chunk.hpp"

namespace HugeCTR {

int getPid();
int getLocalGPUCount();
int getLoaderNth();
int getWorkerNth();

uint64_t prefetchToDress(void*, size_t, Dress::DType);

template<typename T>
void preDressCSR(CSRChunk<T>* cnk, size_t n_row, int* ptr, T* nnzs) {
    int pid = getPid();
    int local_gpu_count = getLocalGPUCount();
    int micro_batch_size = n_row / cnk->get_num_devices();
    auto dtype = Dress::CppTypeToDress<T>();
    for (int i = pid * local_gpu_count; i < (pid + 1) * local_gpu_count; ++i) {
        int key_offset = ptr[i * micro_batch_size];
        int key_count = ptr[(i + 1) * micro_batch_size] - key_offset;
        auto qid = prefetchToDress(nnzs + key_offset, key_count, dtype);
        cnk->setQid(i - pid * local_gpu_count, qid);
        for (int j = i * micro_batch_size; j < (i + 1) * micro_batch_size; ++j) {
            cnk->get_csr_buffer(i).new_row();
            for (int k = ptr[j]; k < ptr[j + 1]; ++k) {
                cnk->get_csr_buffer(i).push_back(nnzs[k]);
            }
        }
    }
}

};  // namespace HugeCTR
