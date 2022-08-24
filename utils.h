#ifndef ACA_CUDA_UTILS_H
#define ACA_CUDA_UTILS_H

#include <driver_types.h>
#include <cstdio>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

/**
 * Assumes that n is a power of four
 * @return base 4 integer logarithm of n
 */
CUDA_HOSTDEV int log4(int n);

CUDA_HOSTDEV int pow4(int n);

bool is_power_of_four(int n);

#define CHECK(call) {\
    auto error = call;\
    if (error != cudaSuccess) {\
        fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}
#endif//ACA_CUDA_UTILS_H
