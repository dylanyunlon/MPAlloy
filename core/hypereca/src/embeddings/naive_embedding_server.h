#pragma once

#ifndef DRESS_NAIVE_EMBEDDING_SERVER_H
#define DRESS_NAIVE_EMBEDDING_SERVER_H

#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

#include "host_embedding_server.h"

namespace Dress {

class NaiveEmbeddingServer: public HostEmbeddingServer {
public:
    NaiveEmbeddingServer(size_t item_sz, DType ktype, DType dtype):
        HostEmbeddingServer(item_sz, ktype, dtype) {}

    template<typename KeyType>
    void localLookup(KeyType*, size_t batch_size, uint64_t* idx,
            int preferred_rank);
    void localLookup(Tensor uid, size_t batch_size, uint64_t* idx,
            int preferred_rank=-1);
};

}; // namespace Dress

#endif  // DRESS_NAIVE_EMBEDDING_SERVER_H

