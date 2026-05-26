#include "HugeCTR/include/embeddings/dress_embedding.hpp"
#include "HugeCTR/include/dress_util.hpp"
#include "HugeCTR/include/embedding.hpp"

#define DRESS_USE_CUDA
#include "dress/combine_ops.h"
#include "dress/config.h"
#include "dress/context.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif  // ENABLE_MPI

namespace HugeCTR {

std::shared_ptr<Dress::Embedding> emb_;
std::shared_ptr<GPUResourceGroup> rsr_grp_;
int pid_ = -1;

int getLoaderNth() {
    int n = Dress::cfgi()->getInt("nth_loader");
    return n;
}
int getWorkerNth() {
    return Dress::cfgi()->getInt("nth_worker");
}

void createEmbedding(Dress::DType ktype, Dress::DType dtype,
        SparseEmbeddingHashParams<float> embedding_params,
        const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    rsr_grp_ = gpu_resource_group;
    auto local_gpu_count_ = rsr_grp_->size();
    for (size_t id = 0; id < local_gpu_count_; id++) {
        auto device_id = (*rsr_grp_)[id]->get_device_id();
        auto stream = (*rsr_grp_)[id]->get_stream();
        Dress::setCudaStream(device_id, stream);
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        Dress::setCudaStream(device_id | 0x10000, stream);
        Dress::setNcclComm(device_id, *(*gpu_resource_group)[id]->get_nccl_ptr());
    }

    Dress::EmbeddingOption emb_opt;
    emb_opt.type = Dress::EmbeddingOption::Hypra;
    if (Dress::cfgi()->getString("type") == "host") {
        emb_opt.type = Dress::EmbeddingOption::Host;
    }
    emb_opt.item_sz = embedding_params.embedding_vec_size;
    emb_opt.ktype = ktype;
    emb_opt.dtype = dtype;
    emb_.reset(Dress::Embedding::create(emb_opt));
    Dress::OptimizerOption opt_opt;
    opt_opt.lr = embedding_params.opt_params.lr / embedding_params.opt_params.scaler;
    opt_opt.dtype = Dress::FP32;
    switch (embedding_params.opt_params.optimizer) {
        case Optimizer_t::SGD:
            opt_opt.type = Dress::OptimizerOption::SGD;
            break;
        default:
            throw "Unimplemented";
    }
    emb_->setOptimizer(Dress::Optimizer::create(opt_opt));
}

Embedding<EmbeddingCreator::TYPE_1, float>*
EmbeddingCreator::create_dress_embedding(
        const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
        SparseEmbeddingHashParams<float> embedding_params,
        const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    createEmbedding(Dress::CppTypeToDress<TYPE_1>(), Dress::FP32,
            embedding_params, gpu_resource_group);

    Embedding<TYPE_1, float>* sparse_embedding =
        new DressEmbedding<TYPE_1, float>(
                row_offsets_tensors,
                value_tensors,
                embedding_params,
                gpu_resource_group,
                emb_.get());
    return sparse_embedding;
}

Embedding<EmbeddingCreator::TYPE_2, float>*
EmbeddingCreator::create_dress_embedding(
        const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
        SparseEmbeddingHashParams<float> embedding_params,
        const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    createEmbedding(Dress::CppTypeToDress<TYPE_2>(), Dress::FP32,
            embedding_params, gpu_resource_group);
    Embedding<TYPE_2, float>* sparse_embedding =
        new DressEmbedding<TYPE_2, float>(
                row_offsets_tensors,
                value_tensors,
                embedding_params,
                gpu_resource_group,
                emb_.get());
    return sparse_embedding;
}

Embedding<EmbeddingCreator::TYPE_1, __half>*
EmbeddingCreator::create_dress_embedding(
    const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
    SparseEmbeddingHashParams<__half> embedding_params,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    throw "Unsupported";
}

Embedding<EmbeddingCreator::TYPE_2, __half>*
EmbeddingCreator::create_dress_embedding(
    const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
    SparseEmbeddingHashParams<__half> embedding_params,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    throw "Unsupported";
}

int getPid() {
    if (pid_ == -1) {
#ifdef ENABLE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
#else
        pid_ = 0;
#endif  // ENABLE_MPI
    }
    return pid_;
}

int getLocalGPUCount() {
    while (rsr_grp_.get() == 0) {
        std::this_thread::yield();
    }
    return rsr_grp_->get_local_gpu_count();
}

bool dedup = Dress::cfgi()->getBool("dedup");

uint64_t prefetchToDress(void* keys_ptr, size_t key_count, Dress::DType ktype) {
    auto key_tensor = Dress::Tensor(keys_ptr, key_count * Dress::typeSize(ktype));
    uint64_t qid;
    if (dedup) {
        auto dedupd = dedupKeys(key_tensor, ktype);
        using namespace Dress;
        DISPATCH_INT(ktype, KeyType, [&]() {
            auto lookup_ptr = (uint64_t*)dedupd.lookup.getPtr();
            auto kp = (KeyType*)keys_ptr;
            std::copy(lookup_ptr, lookup_ptr + key_count, kp);
            dedupd.lookup.free();
        });
        while (emb_.get() == 0) {
            std::this_thread::yield();
        }
        qid = emb_->prefetch(dedupd.val, dedupd.nnz);
        dedupd.val.free();
    } else {
        while (emb_.get() == 0) {
            std::this_thread::yield();
        }
        qid = emb_->prefetch(key_tensor, key_count);
        using namespace Dress;
        DISPATCH_INT(ktype, KeyType, [&]() {
            auto kp = (KeyType*)keys_ptr;
            for (size_t i = 0; i < key_count; ++i) {
                kp[i] = i;
            }
        });
    }
    return qid;
}

}  // namespace HugeCTR
