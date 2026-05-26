#pragma once
#ifndef DRESS_CONTEXT_H
#define DRESS_CONTEXT_H

#include <vector>
#ifdef DRESS_USE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#endif  // DRESS_USE_CUDA

namespace Dress {

#ifdef DRESS_USE_CUDA
void setCudaStream(int idx, cudaStream_t stream);
cudaStream_t getCudaStream(int idx);
std::vector<int> getCudaDevices();
void setNcclComm(int idx, ncclComm_t comm);
ncclComm_t getNcclComm(int idx);
#endif  // DRESS_USE_CUDA

};

#endif  // DRESS_CONTEXT_H
