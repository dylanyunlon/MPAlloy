#include <assert.h>
#include <cstring>


template<typename KeyType, typename DataType>
void combineByCSRForwardImpl(size_t n_rows, size_t item_size,
        const KeyType* ptr, const DataType* val, DataType* out,
        const float alpha, const float beta) {
    assert(alpha == 1. && beta == 0.);
    if (beta == 0.) {
        // memset(out, 0, sizeof(DataType) * n_rows * item_size);
    } else {
        for (size_t i = 0; i < n_rows * item_size; ++i) {
            out[i] *= beta;
        }
    }
#pragma omp parallel for
    for (size_t i = 0; i < n_rows; ++i) {
        memset(out + i * item_size, 0, sizeof(DataType) * item_size);
        for (KeyType j = ptr[i]; j < ptr[i + 1]; ++j) {
#pragma GCC ivdep
            for (size_t k = 0; k < item_size; ++k) {
                out[i * item_size + k] += val[j * item_size + k];
            }
        }
    }
}

template<typename KeyType, typename DataType>
void combineByCSRBackwardImpl(size_t n_rows, size_t item_size,
        const KeyType* ptr, const DataType* grad, DataType* out,
        const float alpha, const float beta) {
    // memset(out, 0, sizeof(out));
    assert(alpha == 1. && beta == 0.);
#pragma omp parallel for
    for (size_t i = 0; i < n_rows; ++i) {
        for (KeyType j = ptr[i]; j < ptr[i + 1]; ++j) {
            // memcpy(out + j * item_size, grad + i * item_size,
                    // item_size * sizeof(DataType));
#pragma GCC ivdep
            for (size_t k = 0; k < item_size; ++k) {
                out[j * item_size + k] = grad[i * item_size + k];
            }
        }
    }
}

