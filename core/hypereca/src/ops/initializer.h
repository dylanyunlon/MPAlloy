#pragma once
#ifndef DRESS_INITIALIZER_H
#define DRESS_INITIALIZER_H

#include <random>

namespace Dress {

// TODO: Customize initializer

template<typename T>
void normFill(void* ptr, size_t n, double mu=0., double sigma=1.) {
    T* ptr_ = (T*)ptr;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mu, sigma);
    for (size_t i = 0; i < n; ++i) {
        ptr_[i] = distribution(generator);
    }
}

template<typename T>
void uniformFill(void* ptr, size_t n, double lo=0., double hi=1.) {
    static const auto nth = cfgi()->getInt("nth_init");
    T* ptr_ = (T*)ptr;
    std::uniform_real_distribution<T> distribution(lo, hi);
#pragma omp parallel for num_threads(nth)
    for (size_t i = 0; i < n; ++i) {
        static thread_local std::default_random_engine gen_;
        ptr_[i] = distribution(gen_);
    }
}
};

#endif  // DRESS_INITIALIZER_H
