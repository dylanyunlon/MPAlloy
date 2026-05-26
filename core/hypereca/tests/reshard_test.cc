#include <memory>
#include <vector>
#include <unordered_map>

#include <cstdlib>
#include <ctime>
#include "dress/embedding.h"
#include "dress/context.h"
#include "dress/logging.h"
#include "dress/profiler.h"

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

using namespace Dress;

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d: %s\n", device, deviceProp.name);
        cudaSetDevice(device);
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        Dress::setCudaStream(device, stream);
    }

    const size_t embsz = 128;
    EmbeddingOption emb_opt;
    emb_opt.type = EmbeddingOption::Hypra;
    emb_opt.item_sz = embsz;
    emb_opt.ktype = UINT64;
    std::shared_ptr<Embedding> emb(Embedding::create(emb_opt));
    PINFO << "created\n";
    const size_t n_tests = 10;

    std::unordered_map<size_t, float*> embr;

    std::vector<void*> ptrs;
    int rank = 0;
#ifdef DRESS_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif  // DRESS_USE_MPI
    srand(114514 + rank);
    PINFO << "mpi init " << rank << "\n";

    for (size_t i_test = 0; i_test < n_tests; ++i_test) {
        TIMER_START(reshard);
        emb->reshard();
        TIMER_END(reshard);
#ifdef DRESS_USE_MPI
        if (rank == 0)
#endif  // DRESS_USE_MPI
        PINFO << REPORT_TIME(reshard) << std::endl;
    }
    for (auto& ptr: ptrs) {
        delete [] ptr;
    }
}
