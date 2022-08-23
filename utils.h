#ifndef ACA_CUDA_UTILS_H
#define ACA_CUDA_UTILS_H

#include <cmath>
#include <cstdio>
#include <driver_types.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

/**
 * Assumes that n is a power of four
 * @return base 4 integer logarithm of n
 */
CUDA_HOSTDEV int log4(int n) {
    // same as log(n) / log(4)
    return static_cast<int>(std::log2(n) / 2);
}

CUDA_HOSTDEV int pow4(int n) {
    // same as pow(4, n)
    return int(std::pow(4, n));
}

bool is_power_of_four(int n) {
    return n == std::pow(4, log4(n));
}

void CHECK(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

#endif//ACA_CUDA_UTILS_H
