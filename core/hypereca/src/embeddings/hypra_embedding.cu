#include "hypra_embedding.h"

#include <dress/profiler.h>
#include <dress/context.h>

#include <fstream>
#include <assert.h>
#include <omp.h>
#ifdef DRESS_USE_MPI
#include <mpi.h>
#endif  // DRESS_USE_MPI

#include "index_kernels.cuh"

namespace Dress {

static const auto nth_prefetch = cfgi()->getInt("nth_prefetch");
static const auto nth_embedding = cfgi()->getInt("nth_embedding");
static const bool drop_unfreq = cfgi()->getBool("drop_unfreq");

void HypraEmbedding::init() {
    cudaGetDeviceCount(&gpu_count_);
    this->host_buf_.resize(gpu_count_ * 2);
    this->cuda_buf_.resize(gpu_count_);
    this->gpu_pool_.resize(gpu_count_);
    this->gpu_grad_.resize(gpu_count_);

    auto freq_filename = cfgi()->getString("freq_list_file");
    if (freq_filename != "") {
        std::ifstream fin(freq_filename);
        key_t x;
        while (fin >> x) {
            this->freq_keys_[x] = this->freq_keys_.size();
        }
        fin.close();
    }

    if (this->freq_keys_.size() > 0) {
        auto freq_sz = this->freq_keys_.size();
        auto uid(Tensor::create(freq_sz, this->ktype_));
        DISPATCH_INT(this->ktype_, KeyType, ([&]() {
            auto ptr = (KeyType*)uid.getPtr();
            for (auto& it: this->freq_keys_) {
                ptr[it.second] = it.first;
            }
        }));
        uint64_t qid = HostEmbedding::prefetch(uid, freq_sz);
        auto sz = this->itemBytes() * this->freq_keys_.size();
        auto emb_host(Tensor::create(sz));
        HostEmbedding::pull(qid, &emb_host);

        for (auto& device_id: getCudaDevices()) {
            this->gpu_pool_[device_id] = emb_host.to(CUDA, device_id);
            char* ptr;
            cudaMalloc(&ptr, sz);
            this->gpu_grad_[device_id] = Tensor(ptr, sz, CUDA);
        }
        emb_host.free();
        uid.free();
    }
}

void HypraEmbedding::reshard() {
    if (this->server_->getRank() == 0 && this->freq_keys_.size() > 0) {
        auto freq_sz = this->freq_keys_.size();
        auto uid(Tensor::create(freq_sz, this->ktype_));
        DISPATCH_INT(this->ktype_, KeyType, ([&]() {
            auto ptr = (KeyType*)uid.getPtr();
            for (auto& it: this->freq_keys_) {
                ptr[it.second] = it.first;
            }
        }));
        uint64_t qid = HostEmbedding::prefetch(uid, freq_sz);
        auto sz = this->itemBytes() * this->freq_keys_.size();
        auto devices = getCudaDevices();
        auto emb_host = this->gpu_pool_[getCudaDevices()[0]].to(CPU);
        HostEmbedding::push(qid, emb_host); // TODO: This is a mocked push. Should assign the values directly.
        for (auto& device_id: getCudaDevices()) {
            this->gpu_pool_[device_id].free();
            this->gpu_grad_[device_id].free();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // TODO: update the freq keys
    if (this->freq_keys_.size() > 0) {
        auto freq_sz = this->freq_keys_.size();
        auto uid(Tensor::create(freq_sz, this->ktype_));
        DISPATCH_INT(this->ktype_, KeyType, ([&]() {
            auto ptr = (KeyType*)uid.getPtr();
            for (auto& it: this->freq_keys_) {
                ptr[it.second] = it.first;
            }
        }));
        uint64_t qid = HostEmbedding::prefetch(uid, freq_sz);
        auto sz = this->itemBytes() * this->freq_keys_.size();
        auto emb_host(Tensor::create(sz));
        HostEmbedding::pull(qid, &emb_host);

        for (auto& device_id: getCudaDevices()) {
            this->gpu_pool_[device_id] = emb_host.to(CUDA, device_id);
            char* ptr;
            cudaMalloc(&ptr, sz);
            this->gpu_grad_[device_id] = Tensor(ptr, sz, CUDA);
        }
        emb_host.free();
        uid.free();
    }
}

void HypraEmbedding::destroy() {
}

void* HypraEmbedding::getBuffer(size_t sz, bool host, int idx) {
    auto& buf_ = host ? this->host_buf_ : this->cuda_buf_;
    if (buf_[idx].sz >= sz) {
        return buf_[idx].ptr;
    }
    if (buf_[idx].ptr) {
        cudaFree(buf_[idx].ptr);
    }
    buf_[idx].sz = sz * 2;
    if (host) {
        cudaMallocHost(&buf_[idx].ptr, sz * 2 * this->itemBytes());
    } else {
        cudaMalloc(&buf_[idx].ptr, sz * 2 * this->itemBytes());
    }
    return buf_[idx].ptr;
}

template<typename ValType>
using Vector2d = std::vector<std::vector<ValType>>;

template<typename KeyType>
void lookupKeys(size_t batch_size, KeyType* key_ptr,
        const std::unordered_map<key_t, size_t> freq_keys,
        Vector2d<size_t>& cp_froms, Vector2d<size_t>& cp_tos) {
    // int nf = 0;
    int nth = nth_prefetch;
// #pragma omp parallel for num_threads(nth)
    for (size_t i = 0; i < batch_size; ++i) {
        auto it = freq_keys.find(key_ptr[i]);
        if (it != freq_keys.end()) {
            // ++nf;
            key_ptr[i] = it->second | flag_skip;
            int rank = omp_get_thread_num();
            cp_froms[rank].push_back(it->second);
            cp_tos[rank].push_back(i);
        }
    }
    /*
    fprintf(stderr, "Hypra hit rate %.3lf comm reduction rate %.3lf\n",
            (double)nf / freq_keys.size(),
            (double)nf / batch_size);
            */
}

uint64_t HypraEmbedding::prefetch(Tensor uid, size_t batch_size, Tensor* out) {
    // Shall not determine prefetch target on runtime
    assert(out == 0);

    int nth = nth_prefetch;
    Vector2d<size_t> cp_froms(nth), cp_tos(nth);
    DISPATCH_INT(this->ktype_, KeyType, ([&]() {
        auto key_ptr = (KeyType*)uid.getPtr();
        lookupKeys<KeyType>(batch_size, key_ptr,
                this->freq_keys_, cp_froms, cp_tos);
    }));
    std::vector<size_t> szs(nth + 1);
    szs[0] = 0;
    for (size_t i = 0; i < nth; ++i) {
        szs[i + 1] = szs[i] + cp_froms[i].size();
    }
    size_t* ptr;
    cudaMallocHost(&ptr, szs[nth] * sizeof(size_t));
    Tensor cp_from(ptr);
    cudaMallocHost(&ptr, szs[nth] * sizeof(size_t));
    Tensor cp_to(ptr);
// #pragma omp parallel for num_threads(nth)
    for (size_t rank = 0; rank < nth; ++rank) {
        std::copy(cp_froms[rank].begin(), cp_froms[rank].end(),
                (size_t*)cp_from.getPtr() + szs[rank]);
        cp_froms[rank].clear();
        std::copy(cp_tos[rank].begin(), cp_tos[rank].end(),
                (size_t*)cp_to.getPtr() + szs[rank]);
        cp_tos[rank].clear();
    }

    auto qid = HostEmbedding::prefetch(uid, batch_size);

    QueryData q;
    q.cp_from = cp_from;
    q.cp_to = cp_to;
    q.sz = szs[nth];

    this->qry_data_mtx_.lock();
    this->qry_data_[qid] = q;
    this->qry_data_mtx_.unlock();

    return qid;
}

void HypraEmbedding::pull(uint64_t qid, Tensor* out) {
    assert(out != 0);
    assert(out->device() == CUDA);
    int device_id;
    cudaGetDevice(&device_id);
    auto stream = getCudaStream(device_id);

    this->qry_mtx_.lock();
    auto qry = this->qrys_[qid];
    this->qry_mtx_.unlock();
    if (out) {
        // DANGEROUS: assume out is no more used in push
        qry.out = out;
    }
    auto out_ptr = (char*)qry.out->getPtr();

    this->qry_data_mtx_.lock();
    auto qd = this->qry_data_[qid];
    this->qry_data_mtx_.unlock();

    // On-device pull
    DISPATCH_INT(this->ktype_, KeyType, [&]() {
        DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
            if (qd.sz) {
                auto per_block = 512 / this->item_sz_;
                dim3 block_dim(this->item_sz_, per_block);
                dim3 grid_dim((qd.sz - 1) / per_block + 1);
                indexCopyKernel<DataType><<<grid_dim, block_dim, 0, stream>>>(
                        qd.sz, (DataType*)out_ptr, (size_t*)qd.cp_to.getPtr(),
                        (DataType*)gpu_pool_[device_id].getPtr(),
                        (size_t*)qd.cp_from.getPtr());
            }
        }));
    });

    // Cross-device pull
    if (!drop_unfreq) {
#ifdef DRESS_USE_UCWO
        this->server_->createChunks();
        MPI_Barrier(MPI_COMM_WORLD);
        // temporary drawback for backward
        auto idx_ptr = this->getBuffer(qry.batch_size, false, device_id);
        int server_world_size = this->server_->getWorldSize();
        auto n = qry.rank_ptr[server_world_size];
        cudaMemcpyAsync(idx_ptr, qry.rank_ridx, sizeof(size_t) * n,
                cudaMemcpyHostToDevice, stream);

        TIMER_START(remote_pull);
        auto w = this->server_->getWorker();
        int c = 0;
        for (size_t i = 0; i < qry.batch_size; ++i) {
            if (qry.idx[i] & flag_skip) {
                continue;
            }
            this->server_->pullRemote(w, qry.idx[i],
                    out_ptr + i * this->itemBytes());
            ++c;
        }

        /*
        int rank = this->server_->getRank();
        auto host_buf = (char*)getBuffer(qry.batch_size, true, device_id);
        this->server_->pullLocal(qry.packs[rank], host_buf);

        DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
            int n = qry.rank_ptr[rank + 1] - qry.rank_ptr[rank];
            auto per_block = 512 / this->item_sz_;
            dim3 block_dim(this->item_sz_, per_block);
            dim3 grid_dim((n - 1) / per_block + 1);
            indexPutKernel<DataType><<<grid_dim, block_dim, 0, stream>>>(
                    n, (DataType*)out_ptr, (DataType*)host_buf,
                    (size_t*)idx_ptr + qry.rank_ptr[rank]);
        }));
        */
        w->flush();
        TIMER_END(remote_pull);
        auto t = TIMER_READ(remote_pull);
        auto bw = c * this->itemBytes() / t;
        PFR_ADD_TIME("pull", device_id, t);
#else
        auto host_buf = (char*)getBuffer(qry.batch_size, true, device_id);
        HostEmbedding::pullExchange(qry, host_buf);
        out->resize(qry.batch_size * this->itemBytes());
        auto idx_ptr = this->getBuffer(qry.batch_size, false, device_id);
        int server_world_size = this->server_->getWorldSize();
        auto n = qry.rank_ptr[server_world_size];
        cudaMemcpyAsync(idx_ptr, qry.rank_ridx, sizeof(size_t) * n,
                cudaMemcpyHostToDevice, stream);
        DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
            auto per_block = 512 / this->item_sz_;
            dim3 block_dim(this->item_sz_, per_block);
            dim3 grid_dim((n - 1) / per_block + 1);
            indexPutKernel<DataType><<<grid_dim, block_dim, 0, stream>>>(
                    n, (DataType*)out_ptr, (DataType*)host_buf, (size_t*)idx_ptr);
        }));
#endif  // DRESS_USE_UCWO
    }
}

void HypraEmbedding::push(uint64_t qid, const Tensor& grad) {
    TIMER_START(push);
    assert(grad.device() == CUDA);

    this->qry_mtx_.lock();
    auto qry = this->qrys_[qid];
    this->qry_mtx_.unlock();

    int device_id;
    cudaGetDevice(&device_id);
    auto stream = getCudaStream(device_id);

    cudaEvent_t evt_s, evt_e;
    cudaEventCreate(&evt_s);
    cudaEventCreate(&evt_e);

    auto host_buf = this->getBuffer(qry.batch_size, true, device_id + this->gpu_count_);
    int server_world_size = this->server_->getWorldSize();
    auto n = qry.rank_ptr[server_world_size];

    if (!drop_unfreq) {
        // DANGEROUS: we assume the buffer is the same for push and pull, which
        // may change in the future
        auto idx_ptr = this->getBuffer(qry.batch_size, false, device_id);

        DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
            auto per_block = 512 / this->item_sz_;
            dim3 block_dim(this->item_sz_, per_block);
            dim3 grid_dim((n - 1) / per_block + 1);
            indexGetKernel<DataType><<<grid_dim, block_dim, 0, stream>>>(
                    n, (DataType*)host_buf, (DataType*)grad.getPtr(),
                    (size_t*)idx_ptr);
        }));
        cudaStreamSynchronize(stream);
    }

    TIMER_START(qdl);
    this->qry_data_mtx_.lock();
    auto qd = this->qry_data_[qid];
    this->qry_data_mtx_.unlock();
    PFR_ADD_TIME("bwth", device_id, TIMER_READ(qdl));

    // On-device push
    if (this->freq_keys_.size()) {
        cudaMemsetAsync(this->gpu_grad_[device_id].getPtr(), 0,
                this->gpu_grad_[device_id].size(), stream);
    }
    DISPATCH_INT(this->ktype_, KeyType, [&]() {
        DISPATCH_FLOAT(this->dtype_, DataType, ([&]() {
            auto grad_out = (DataType*)gpu_grad_[device_id].getPtr();
            if (qd.sz) {
                auto per_block = 512 / this->item_sz_;
                dim3 block_dim(this->item_sz_, per_block);
                dim3 grid_dim((qd.sz - 1) / per_block + 1);
                indexCopyKernel<DataType><<<grid_dim, block_dim, 0, stream>>>(
                        qd.sz,
                        grad_out, (size_t*)qd.cp_from.getPtr(),
                        (DataType*)grad.getPtr(), (size_t*)qd.cp_to.getPtr());
            }
            cudaStreamSynchronize(stream);
            auto comm = getNcclComm(device_id);
            // TODO: Support for half data type
            cudaEventRecord(evt_s, stream);
            ncclAllReduce(grad_out, grad_out,
                    this->item_sz_ * this->freq_keys_.size(),
                    ncclFloat, ncclSum, comm, stream);
            cudaEventRecord(evt_e, stream);
        }));
    });

    if (!drop_unfreq) {
        HostEmbedding::pushExchange(qry, (char*)host_buf);
    } else {
        delete [] qry.idx;
        delete [] qry.rank_ridx;
        // delete [] qry.buf;
        this->qry_mtx_.lock();
        this->qrys_.erase(qry.qid);
        this->qry_mtx_.unlock();
    }
    cudaStreamSynchronize(stream);
    float t;
    cudaEventElapsedTime(&t, evt_s, evt_e);
    cudaEventDestroy(evt_s);
    cudaEventDestroy(evt_e);
    PFR_ADD_TIME("ard", device_id, t * 1e-3);
}

void HypraEmbedding::update() {
    if (!this->opt_) {
        return;
    }
    if (this->freq_keys_.size()) {
        if (!this->opt_tid_.size()) {
            this->opt_tid_.resize(this->gpu_count_);
            for (auto dev_id: getCudaDevices()) {
                cudaSetDevice(dev_id);
                this->opt_tid_[dev_id] =
                    this->opt_->addTensor(this->gpu_pool_[dev_id]);
            }
        }
        for (auto dev_id: getCudaDevices()) {
            auto stream = getCudaStream(dev_id);
            cudaSetDevice(dev_id);
            this->opt_->step(this->opt_tid_[dev_id],
                    this->item_sz_ * this->freq_keys_.size(),
                    this->gpu_grad_[dev_id], stream);
        }
    }
    if (!drop_unfreq) {
        HostEmbedding::update();
    }
}

};  // namespace Dress
