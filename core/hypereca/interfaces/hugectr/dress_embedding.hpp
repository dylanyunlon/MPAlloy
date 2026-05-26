#pragma once

#include <vector>
#include <thread>
#include <cuda_runtime_api.h>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"

#define DRESS_USE_CUDA
#include "dress/embedding.h"
#include "dress/optimizer.h"
#include "dress/combine_ops.h"
#include "dress/profiler.h"
#include "dress/dedup.h"


namespace HugeCTR {

struct InterRes {
    uint64_t qid;
    size_t nnz;
    void *lookup;
    Dress::Tensor emb;
};

template <typename TypeHashKey, typename TypeEmbeddingComp>
class DressEmbedding: public Embedding<TypeHashKey, TypeEmbeddingComp> {
    using Base = Embedding<TypeHashKey, TypeEmbeddingComp>;

private:
    Dress::Embedding* emb_;
    SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params_;
    std::vector<int> slot_num_per_gpu_;

    std::vector<TypeHashKey*> row_offsets_;
    TypeEmbeddingComp* buf_;
    TypeEmbeddingComp* embeddings;
    std::vector<InterRes> inter_res;
    size_t nnz;
    bool localized_compatible_mode;
    bool host_mode;
    bool eval_mode;

public:
    void initInterRes(
            SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params) {
        auto local_gpu_count_ = Base::device_resources_->size();
        auto total_gpu_count_ = Base::device_resources_->get_total_gpu_count();
        auto buf_sz = (embedding_params_.batch_size / total_gpu_count_) *
                    embedding_params_.slot_num *
                    embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp);
        this->inter_res.resize(local_gpu_count_);
        for (size_t id = 0; id < local_gpu_count_; id++) {
            int gid = Base::device_resources_->get_global_id(id);
            int slot_num_per_gpu =
                embedding_params_.slot_num / total_gpu_count_ +
                ((gid < (int)(embedding_params_.slot_num % total_gpu_count_)) ? 1 : 0);
            slot_num_per_gpu_.push_back(slot_num_per_gpu);
            auto device_id = (*Base::device_resources_)[id]->get_device_id();
            cudaSetDevice(device_id);
            char* emb_buf;
            cudaMalloc(&emb_buf, buf_sz);
            this->inter_res[id].emb = Dress::Tensor(emb_buf, -1ull, Dress::CUDA);
        }
    }

    DressEmbedding(const Tensors<TypeHashKey> &row_offsets_tensors,
            const Tensors<TypeHashKey> &hash_key_tensors,
            SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params,
            const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
            Dress::Embedding* emb):
        embedding_params_(embedding_params),
        Base(row_offsets_tensors,
            hash_key_tensors, embedding_params.batch_size,
            embedding_params.slot_num, embedding_params.embedding_vec_size,
            gpu_resource_group, embedding_params.opt_params.scaler),
        eval_mode(false),
        emb_(emb)
    {
        auto local_gpu_count_ = Base::device_resources_->size();
        this->initInterRes(embedding_params);
        this->localized_compatible_mode = (Base::row_offsets_tensors_.size() ==
                local_gpu_count_);
        assert(!this->localized_compatible_mode);
        this->host_mode = emb_->type() == Dress::EmbeddingOption::Host;
        if (this->host_mode || this->localized_compatible_mode) {
            CK_CUDA_THROW_(cudaMallocHost((void**)&buf_,
                    embedding_params_.batch_size * embedding_params_.slot_num *
                    embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp)));
        }
    }

    // Copy ctor for eval
    DressEmbedding(const Tensors<TypeHashKey> &row_offsets_tensors,
            const Tensors<TypeHashKey> &value_tensors,
            size_t batchsize, const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
            DressEmbedding& obj):
        embedding_params_(obj.embedding_params_),
        emb_(obj.emb_),
        localized_compatible_mode(obj.localized_compatible_mode),
        host_mode(obj.host_mode),
        eval_mode(true),
        Base(row_offsets_tensors, value_tensors, batchsize, obj.embedding_params_.slot_num,
             obj.embedding_params_.embedding_vec_size, gpu_resource_group,
             obj.embedding_params_.opt_params.scaler),
        slot_num_per_gpu_(obj.slot_num_per_gpu_) {
        this->embedding_params_.batch_size = batchsize;
        this->initInterRes(obj.embedding_params_);
        if (this->host_mode || this->localized_compatible_mode) {
            CK_CUDA_THROW_(cudaMallocHost((void**)&buf_,
                    embedding_params_.batch_size * embedding_params_.slot_num *
                    embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp)));
        }
    }

    ~DressEmbedding() {
        if (this->host_mode || this->localized_compatible_mode) {
            CK_CUDA_THROW_(cudaFreeHost(buf_));
        }
    }

    void syncAllGPUs() {
        auto local_gpu_count = Base::device_resources_->get_local_gpu_count();
        for (int i = 0; i < local_gpu_count; ++i) {
            auto device_id = (*Base::device_resources_)[i]->get_device_id();
            cudaSetDevice(device_id);
            cudaStreamSynchronize((*Base::device_resources_)[i]->get_stream());
        }
    }

    void forward() override {
        if (localized_compatible_mode) {
            forward_localized();
            return;
        }
        TIMER_START(fused_fwd);
        auto local_gpu_count = Base::device_resources_->get_local_gpu_count();
        auto total_gpu_count = Base::device_resources_->get_total_gpu_count();
        auto batch_size = this->embedding_params_.batch_size;
        auto batch_size_per_gpu = batch_size / total_gpu_count;
        auto slot_num = embedding_params_.slot_num;
        auto embedding_size = embedding_params_.embedding_vec_size;

        std::vector<std::thread> thrs;
        auto chunk_size = batch_size_per_gpu * slot_num;
        auto qids = (uint64_t*)Base::row_offsets_tensors_[local_gpu_count]->get_ptr();
        for (int i = 0; i < local_gpu_count; ++i) {
            Base::device_resources_->results[i] = Base::device_resources_->train_thread_pool.push(
                    [=](int _id) mutable {
                auto row_offset = Base::row_offsets_tensors_[i]->get_ptr();
                auto stream = (*Base::device_resources_)[i]->get_stream();
                auto device_id = (*Base::device_resources_)[i]->get_device_id();
                auto per_gpu_size = chunk_size * embedding_size;
                this->inter_res[i].qid = qids[i];
                this->inter_res[i].nnz = emb_->getBatchSize(this->inter_res[i].qid);

                if (this->host_mode) {
                    auto emb_sz = sizeof(TypeEmbeddingComp) * embedding_size * this->inter_res[i].nnz;
                    auto emb_tensor = Dress::Tensor(this->buf_ + i * per_gpu_size, emb_sz);
                    this->emb_->pull(this->inter_res[i].qid, &emb_tensor);
                    cudaSetDevice(device_id);
                    this->inter_res[i].emb.copy(emb_tensor, stream);
                } else {
                    cudaSetDevice(device_id);
                    this->emb_->pull(this->inter_res[i].qid,
                            &this->inter_res[i].emb);
                }
                auto lookup = (TypeHashKey*)Base::value_tensors_[i]->get_ptr();
                Dress::combineDedupedByCSRForward(chunk_size, embedding_size,
                        Dress::CppTypeToDress<TypeHashKey>(),
                        row_offset, 0,
                        this->inter_res[i].nnz, lookup,
                        Dress::CppTypeToDress<TypeEmbeddingComp>(),
                        this->inter_res[i].emb.getPtr(),
                        Base::output_tensors_[i]->get_ptr(),
                        stream);
                cudaStreamSynchronize(stream);
                CUDA_SAFE_CHECK;
            });
        }
        for (int i = 0; i < local_gpu_count; ++i) {
            Base::device_resources_->results[i].get();
        }
        if (this->eval_mode) {
            for (int i = 0; i < local_gpu_count; ++i) {
                auto& saved = this->inter_res[i];
                this->emb_->release(saved.qid);
            }
        }
        PFR_ADD_TIME("fwd", PFR_THR_MAIN, TIMER_READ(fused_fwd));
    }

    void backward() override {
        if (localized_compatible_mode) {
            backward_localized();
            return;
        }
        TIMER_START(fused_bwd);
        auto local_gpu_count = Base::device_resources_->get_local_gpu_count();
        auto total_gpu_count = Base::device_resources_->get_total_gpu_count();
        auto batch_size = this->embedding_params_.batch_size;
        auto batch_size_per_gpu = batch_size / total_gpu_count;
        auto slot_num = embedding_params_.slot_num;
        auto embedding_size = embedding_params_.embedding_vec_size;

        auto chunk_size = batch_size_per_gpu * slot_num;
        for (int i = 0; i < local_gpu_count; ++i) {
            Base::device_resources_->results[i] = Base::device_resources_->train_thread_pool.push(
                    [=](int _id) mutable {
                auto row_offset = Base::row_offsets_tensors_[i]->get_ptr();
                auto lookup = (TypeHashKey*)Base::value_tensors_[i]->get_ptr();
                auto device_id = (*Base::device_resources_)[i]->get_device_id();
                cudaSetDevice(device_id);
                auto& saved = this->inter_res[i];
                auto stream = (*Base::device_resources_)[i]->get_stream();
                Dress::combineDedupedByCSRBackward(chunk_size, embedding_size,
                        Dress::CppTypeToDress<TypeHashKey>(),
                        row_offset, 0,
                        saved.nnz, lookup,
                        Dress::CppTypeToDress<TypeEmbeddingComp>(),
                        Base::output_tensors_[i]->get_ptr(),
                        saved.emb.getPtr(),
                        stream);
                cudaStreamSynchronize(stream);
                if (this->host_mode) {
                    auto per_gpu_size = chunk_size * embedding_size;
                    auto emb_cpu = Dress::Tensor(this->buf_ + i * per_gpu_size);
                    emb_cpu.copy(saved.emb);
                    this->emb_->push(saved.qid, emb_cpu);
                } else {
                    this->emb_->push(saved.qid, saved.emb);
                }
            });
        }
        for (int i = 0; i < local_gpu_count; ++i) {
            Base::device_resources_->results[i].get();
        }
        PFR_ADD_TIME("bwd", PFR_THR_MAIN, TIMER_READ(fused_bwd));
    }

    void update_params() override {
        this->emb_->update();
    }

    void init_params() override {
        // As DrESS dynamically allocate blocks, there is no need to initialize.
    }

    void upload_params_to_device(std::ifstream& weight_stream) override {
        // FIXME: update later
    }

    void download_params_to_host(std::ofstream& weight_stream) override {
        // FIXME: update later
    }

    void set_learning_rate(float lr) override {
        this->emb_->getOptimizer()->setLR(lr / Base::scaler_);
    }

    size_t get_params_num() override {
        // FIXME: update later
        return 0;
    }

    void check_overflow() const override {
        // FIXME: update later
    }

    void get_forward_results(TypeEmbeddingComp* embedding_features) override {
        // FIXME: update later
    }

    void get_backward_results(TypeEmbeddingComp* grad, int devIndex) override {
        // FIXME: update later
    }

    void get_update_params_results(TypeHashKey* hash_table_key, float* values) {
        // FIXME: update later
    }

    Embedding<TypeHashKey, TypeEmbeddingComp> *clone_eval(
          const Tensors<TypeHashKey> &row_offsets_tensors, const Tensors<TypeHashKey> &value_tensors,
          size_t batchsize, const std::shared_ptr<GPUResourceGroup> &gpu_resource_group) {
        Embedding<TypeHashKey, TypeEmbeddingComp> *new_embedding =
            new DressEmbedding<TypeHashKey, TypeEmbeddingComp>(
                row_offsets_tensors, value_tensors, batchsize, gpu_resource_group, *this);
        return new_embedding;
    }

    void forward_localized() {
        if (row_offsets_.size() > 0) {
            for (auto& p: row_offsets_) {
                delete [] p;
            }
            row_offsets_.clear();
            delete [] embeddings;
        }

        auto local_gpu_count = Base::device_resources_->get_total_gpu_count();
        auto batch_size = this->embedding_params_.batch_size;
        auto batch_size_per_gpu = batch_size / local_gpu_count;
        nnz = 0;

        this->syncAllGPUs();
        TIMER_START(fwd_copy_to_cpu);
        // Copy all CSR pointers to CPU
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            auto row_offset_cpu = new TypeHashKey[batch_size * slot_num + 1];
            cudaMemcpyAsync(row_offset_cpu, Base::row_offsets_tensors_[i]->get_ptr(),
                    (batch_size * slot_num + 1) * sizeof(TypeHashKey),
                    cudaMemcpyDeviceToHost,
                    (*Base::device_resources_)[i]->get_stream());
            row_offsets_.push_back(row_offset_cpu);
        }
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            cudaStreamSynchronize((*Base::device_resources_)[i]->get_stream());
            nnz += row_offsets_[i][batch_size * slot_num];
        }

        assert(nnz > 0);

        // Concat all hash keys on CPU
        size_t nnz_offset = 0;
        auto hash_keys = new TypeHashKey[nnz];
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            auto numel = row_offsets_[i][batch_size * slot_num];
            if (numel == 0) {
                continue;
            }
            cudaMemcpyAsync(hash_keys + nnz_offset, Base::value_tensors_[i]->get_ptr(),
                     numel * sizeof(TypeHashKey),
                    cudaMemcpyDeviceToHost,
                    (*Base::device_resources_)[i]->get_stream());
            /*for (int j = 0; j < batch_size * slot_num; ++j) {
                auto tag = ((i + j % slot_num * local_gpu_count) << 24);
                for (auto k = row_offsets_[i][j]; k < row_offsets_[i][j + 1]; ++k) {
                    *(hash_keys + nnz_offset + k) ^= tag;
                }
            }*/
            nnz_offset += numel;
        }

        for (int i = 0; i < local_gpu_count; ++i) {
            cudaStreamSynchronize((*Base::device_resources_)[i]->get_stream());
        }

        TIMER_END(fwd_copy_to_cpu);

        // Embedding Lookup
        auto hash_key_tensor = Dress::Tensor(hash_keys, nnz * sizeof(TypeHashKey));
        auto out = this->emb_->pullOut(hash_key_tensor, nnz);
        auto embeddings_tensor = out.first;
        this->inter_res.resize(1);
        this->inter_res[0].qid = out.second;

        embeddings = (TypeEmbeddingComp*)embeddings_tensor.getPtr();
        auto embedding_size = embedding_params_.embedding_vec_size;

        TIMER_START(fwd_combine);
        // Combine embeddings
        nnz_offset = 0;
        std::vector<TypeEmbeddingComp*> outs_cpu;
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            auto numel = row_offsets_[i][batch_size * slot_num];
            // FIXME: combiner
            auto out_size = batch_size * slot_num * embedding_size;
            assert(out_size > 0);
            auto out_cpu = new TypeEmbeddingComp[out_size];
            Dress::combineByCSRForward(batch_size * slot_num, embedding_size,
                Dress::CppTypeToDress<TypeHashKey>(), row_offsets_[i],
                Dress::CppTypeToDress<TypeEmbeddingComp>(),
                embeddings + nnz_offset * embedding_size, out_cpu, 1., 0.);
            outs_cpu.push_back(out_cpu);
            nnz_offset += numel;
        }
        TIMER_END(fwd_combine);

        auto tot_slot_num = embedding_params_.slot_num;

        TIMER_START(fwd_copy_to_gpu);
        // Reorder and copy back
        for (int i = 0; i < local_gpu_count; ++i) {
            auto per_gpu_size = batch_size_per_gpu * embedding_size * tot_slot_num;
            auto out_cpu = this->buf_ + i * per_gpu_size;
            for (int j = 0; j < batch_size_per_gpu; ++j) {
                for (int k = 0; k < tot_slot_num; ++k) {
                    int gpu_id = k % local_gpu_count;
                    int idx_on_gpu = k / local_gpu_count;
                    auto src_ptr = outs_cpu[gpu_id] + ((i * batch_size_per_gpu
                            + j) * slot_num_per_gpu_[gpu_id] + idx_on_gpu) * embedding_size;
                    auto dst_ptr = out_cpu + (j * tot_slot_num + k) * embedding_size;
                    memcpy(dst_ptr, src_ptr, embedding_size * sizeof(TypeEmbeddingComp));
                }
            }
            cudaMemcpyAsync(Base::output_tensors_[i]->get_ptr(), out_cpu,
                    sizeof(TypeEmbeddingComp) * per_gpu_size,
                    cudaMemcpyHostToDevice,
                    (*Base::device_resources_)[i]->get_stream());
        }
        this->syncAllGPUs();
        TIMER_END(fwd_copy_to_gpu);

        PINFO << REPORT_TIME(fwd_copy_to_cpu) << REPORT_TIME(fwd_combine)
            << REPORT_TIME(fwd_copy_to_gpu) << std::endl;

        for (auto& p: outs_cpu) {
            delete [] p;
        }
        delete [] hash_keys;
    }

    void backward_localized() {
        auto local_gpu_count = Base::device_resources_->get_total_gpu_count();
        auto embedding_size = embedding_params_.embedding_vec_size;
        auto batch_size = this->embedding_params_.batch_size;
        auto batch_size_per_gpu = batch_size / local_gpu_count;
        auto tot_slot_num = embedding_params_.slot_num;
        size_t nnz_offset = 0;

        for (int i = 0; i < local_gpu_count; ++i) {
            auto device_id = (*Base::device_resources_)[i]->get_device_id();
            cudaSetDevice(device_id);
            cudaDeviceSynchronize();
        }
        TIMER_START(bwd_copy_to_cpu);
        // Copy grads to CPU
        std::vector<TypeEmbeddingComp*> grads_cpu;
        auto per_gpu_size = batch_size_per_gpu * tot_slot_num * embedding_size;
        this->syncAllGPUs();
        for (int i = 0; i < local_gpu_count; ++i) {
            auto grad_cpu = this->buf_ + i * per_gpu_size;
            cudaMemcpyAsync(grad_cpu, Base::output_tensors_[i]->get_ptr(),
                    per_gpu_size * sizeof(TypeEmbeddingComp),
                    cudaMemcpyDeviceToHost,
                    (*Base::device_resources_)[i]->get_stream());
            grads_cpu.push_back(grad_cpu);
        }
        this->syncAllGPUs();
        TIMER_END(bwd_copy_to_cpu);

        TIMER_START(bwd_reorder);
        // Reorder and combine backward
        for (int i = 0; i < local_gpu_count; ++i) {
            auto slot_num = slot_num_per_gpu_[i];
            auto numel = row_offsets_[i][batch_size * slot_num];
            if (numel == 0) {
                continue;
            }
            auto grad_cpu = new TypeEmbeddingComp[batch_size * slot_num * embedding_size];
            for (int j = 0; j < batch_size; ++j) {
                auto gpu_id = j / batch_size_per_gpu;
                auto idx_on_gpu = j % batch_size_per_gpu;
                for (int si = 0; si < slot_num; ++si) {
                    auto src_ptr = grads_cpu[gpu_id]
                        + (idx_on_gpu * tot_slot_num
                                + si * local_gpu_count + i) * embedding_size;
                    auto dst_ptr = grad_cpu + (j * slot_num + si) * embedding_size;
                    memcpy(dst_ptr, src_ptr, embedding_size * sizeof(TypeEmbeddingComp));
                }
            }
            Dress::combineByCSRBackward(batch_size * slot_num, embedding_size,
                    Dress::CppTypeToDress<TypeHashKey>(), row_offsets_[i],
                    Dress::CppTypeToDress<TypeEmbeddingComp>(),
                    grad_cpu, embeddings + nnz_offset * embedding_size);
            delete [] grad_cpu;
            nnz_offset += numel;
        }
        TIMER_END(bwd_reorder);

        for (auto& p: row_offsets_) {
            delete [] p;
        }
        row_offsets_.clear();

        // PINFO << REPORT_TIME(bwd_copy_to_cpu) << REPORT_TIME(bwd_reorder) << std::endl;
        // Push to DrESS
        auto t = Dress::Tensor(embeddings,
                nnz * embedding_size * sizeof(TypeEmbeddingComp));
        this->emb_->push(this->inter_res[0].qid, t);
    }
};

}  // namespace HugeCTR
