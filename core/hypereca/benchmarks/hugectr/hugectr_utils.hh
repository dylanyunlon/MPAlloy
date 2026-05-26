#pragma once

#include <set>
#include <vector>
#include <map>
#include <algorithm>

#include "file_source.hpp"
#include "check_sum.hpp"


uint64_t prefetchToDress(int* keys, int nnz);

namespace HugeCTR {

DataSetHeader read_new_file(HugeCTR::Checker* checker_) {
    DataSetHeader data_set_header_;  
    for (int i = 0; i < 10; i++) {
        checker_->next_source();

        Error_t err =
            checker_->read(reinterpret_cast<char*>(&data_set_header_),
                    sizeof(DataSetHeader));
        if (data_set_header_.error_check != 1) {
            ERROR_MESSAGE_("DataHeaderError");
            continue;
        }
        if (err == Error_t::Success) {
            return data_set_header_;
        }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
    return data_set_header_;
}

struct csr_t {
    std::vector<int> ptr, val;
};

class DataReader {
protected:
    CheckSum* checker;
    int num_slots, batch_size, samples_left, dense_dim;
    float* y;

public:
    int local_gpu_count, total_gpu_count, pid;
public:
    DataReader(CheckSum* checker,
            int num_slots, int batch_size, int dense_dim) {
        this->checker = checker;
        this->num_slots = num_slots;
        this->batch_size = batch_size;
        this->samples_left = 0;
        this->dense_dim = dense_dim;
        this->y = new float[dense_dim];
    }

    void read_a_batch() {
        thread_local static int *idx_tmp = 0;
        thread_local static int *nnz_tmp = 0;
        if (nnz_tmp == 0) {
            nnz_tmp = new int[batch_size * num_slots + 1];
            nnz_tmp[0] = 0;
            idx_tmp = new int[batch_size * num_slots * 40];
        }
        size_t row_i = 0;
        for (int i = 0; i < batch_size; ++i) {
            if (!samples_left) {
                auto header = read_new_file(checker);
                samples_left = header.number_of_records;
            }
            --samples_left;
            checker->read(reinterpret_cast<char*>(&y), sizeof(float) * dense_dim);
            for (int k = 0; k < num_slots; ++k) {
                int nnz;
                checker->read(reinterpret_cast<char*>(&nnz), sizeof(int));
                nnz_tmp[row_i + 1] = nnz_tmp[row_i] + nnz;
                ++row_i;
                checker->read(reinterpret_cast<char*>(idx_tmp + nnz_tmp[row_i]),
                        sizeof(int) * nnz);
            }
        }

        auto n_row = batch_size * num_slots;
        int micro_batch_size = n_row / total_gpu_count;
        for (int i = pid * local_gpu_count; i < (pid + 1) * local_gpu_count; ++i) {
            int key_offset = nnz_tmp[i * micro_batch_size];
            int key_count = nnz_tmp[(i + 1) * micro_batch_size] - key_offset;
            auto qid = prefetchToDress(idx_tmp + key_offset, key_count);
        }
    }
};

};


template<class T>
std::vector<std::pair<int, int>> countAndSort(std::map<int, T>& ents) {
    std::vector<std::pair<int, T>> svc;
    for (auto& p: ents) {
        svc.push_back(std::pair<int, int>(-p.second, p.first));
    }
    std::sort(svc.begin(), svc.end());
    return svc;
}

template<class T>
void outputCnt(std::vector<std::pair<int, T>> svc, std::string filename) {
    std::ofstream fou(filename);
    for (auto& p: svc) {
        fou << -p.first << " ";
    }
}

template<class T>
void outputList(std::vector<std::pair<int, T>> svc, std::string filename) {
    std::ofstream fou(filename);
    for (auto& p: svc) {
        fou << p.second << " ";
    }
}

std::set<int> loadCache(int size, std::string filename) {
    std::ifstream fin(filename);
    int x;
    std::set<int> s;
    for (int i = 0; i < size; ++i) {
        fin >> x;
        s.insert(x);
    }
    return s;
}

