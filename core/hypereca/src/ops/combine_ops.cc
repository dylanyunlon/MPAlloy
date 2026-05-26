#include <dress/combine_ops.h>
#include "combine_ops_impl.hh"

namespace Dress {

void combineByCSRForward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr,
        DType val_t, const void* val, void* out,
        const float alpha, const float beta) {
    DISPATCH_KNV(([&] {
            combineByCSRForwardImpl<KeyType, ValType>(n_rows,
                    item_sz, (KeyType*)ptr, (ValType*)val, (ValType*)out,
                    alpha, beta);
    }));
}

void combineByCSRBackward(size_t n_rows, size_t item_sz,
        DType key_t, const void* ptr,
        DType val_t, const void* grad, void* out,
        const float alpha, const float beta) {
    DISPATCH_KNV(([&] {
        combineByCSRBackwardImpl<KeyType, ValType>(n_rows,
                item_sz, (KeyType*)ptr, (ValType*)grad, (ValType*)out,
                alpha, beta);
    }));
}

};  // namespace Dress
