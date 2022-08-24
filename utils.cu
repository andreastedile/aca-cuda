#include "utils.h"

#include <cmath>
#include <cstdio>

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
