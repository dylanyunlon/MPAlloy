#pragma once

#ifndef DRESS_LOGGING_H
#define DRESS_LOGGING_H

#include <cstdlib>
#include <iostream>
#include <iomanip>

#define PERROR std::cerr << "[DrESS Error] "
#define PINFO std::cerr << "[DrESS Info] "
#ifdef NDEBUG
#define PWARNING if (false) std::cerr 
#else
#define PWARNING std::cerr << "[DrESS Warning] "
#endif

#ifdef DRESS_USE_MPI

#define PERROR_ALL std::cerr << "[DrESS Error#" << this->rank << "] "
#define PWARNING_ALL std::cerr << "[DrESS Warning#" << this->rank << "] "
#define PINFO_ALL std::cerr << "[DrESS Info #" << this->rank << "] "

#else

#define PERROR_ALL PERROR
#define PWARNING_ALL PWARNING
#define PINFO_ALL PINFO

#endif  // DRESS_USE_MPI

void printTensorStat(const char* name, float* ptr, size_t size);

#ifdef DRESS_USE_CUDA

#define CUDA_SAFE_CHECK { \
    cudaError err = cudaGetLastError(); \
    if (cudaSuccess != err) { \
        PERROR << "CUDA error at " << __FILE__ << ":" << __LINE__ \
            << " code " << err << ": " << cudaGetErrorString(err) << "\n"; \
        throw "CUDA Error"; \
    } \
}

#define CUDA_SAFE_CHECK_WINFO(__more__) { \
    cudaError err = cudaGetLastError(); \
    if (cudaSuccess != err) { \
        PERROR << "CUDA error at " << __FILE__ << ":" << __LINE__ \
            << " code " << err << ": " << cudaGetErrorString(err) << __more__ << "\n"; \
        throw "CUDA Error"; \
    } \
}

#endif  // DRESS_USE_CUDA

#endif // DRESS_LOGGING_H
