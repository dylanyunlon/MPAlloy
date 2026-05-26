#pragma once

#ifndef DRESS_SPEC_H
#define DRESS_SPEC_H

#include <cstddef>
#include <cstdint>

#define DRESS_TYPE_EXPAND(__fn__) \
    __fn__(FP16, short) \
    __fn__(FP32, float) \
    __fn__(FP64, double) \
    __fn__(INT32, int32_t) \
    __fn__(INT64, int64_t) \
    __fn__(UINT32, uint32_t) \
    __fn__(UINT64, uint64_t)

#define _DISPATCH_CASE(__tvar__, __realt__, __tname__, __fn__) \
    case (__tvar__): { \
        using __tname__ = __realt__; \
        (__fn__)(); \
        break; \
    }

#define DISPATCH_INT(__dt__, __tn__, __fn__) \
    switch (__dt__) { \
        _DISPATCH_CASE(INT32, int32_t, __tn__, __fn__) \
        _DISPATCH_CASE(UINT32, uint32_t, __tn__, __fn__) \
        _DISPATCH_CASE(INT64, int64_t, __tn__, __fn__) \
        _DISPATCH_CASE(UINT64, uint64_t, __tn__, __fn__) \
        default: throw "Unsupported int type"; \
    }

#define DISPATCH_FLOAT(__dt__, __tn__, __fn__) \
    switch (__dt__) { \
        _DISPATCH_CASE(FP32, float, __tn__, __fn__) \
        default: throw "Unsupported float type"; \
    }

namespace Dress {

enum Device { CPU=0, CUDA=1, Lazy=17 };

enum DType { FP16=1, FP32=2, FP64=3, INT32=20, INT64=21, UINT32=52, UINT64=53 };

template<typename T>
inline DType CppTypeToDress();
#define _CPP_TYPE_TO_DRESS(__dresst__, __cppt__) \
    template <> \
    inline DType CppTypeToDress<__cppt__>() { return __dresst__; }
DRESS_TYPE_EXPAND(_CPP_TYPE_TO_DRESS);
_CPP_TYPE_TO_DRESS(INT64, long long);
#undef _CPP_TYPE_TO_DRESS

#define _CASE_FN(__dresst__, __cppt__) \
    case (__dresst__): return sizeof(__cppt__);
static inline size_t typeSize(DType d) {
    switch (d) {
        DRESS_TYPE_EXPAND(_CASE_FN);
        default:
            throw "Unknown data type";
    }
}
#undef _CASE_FN

};

#endif  // DRESS_SPEC_H
