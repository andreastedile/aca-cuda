#ifndef ACA_CUDA_UTILS_H
#define ACA_CUDA_UTILS_H

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
CUDA_HOSTDEV int log4(int n);

CUDA_HOSTDEV int pow4(int n);

bool is_power_of_four(int n);

void CHECK(cudaError_t error);

#endif//ACA_CUDA_UTILS_H
