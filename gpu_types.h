#ifndef ACA_CUDA_GPU_TYPES_H
#define ACA_CUDA_GPU_TYPES_H

#include "cpu_types.h"
#include "rgb.h"
#include "utils.h"
#include <cstdint>
#include <cuda_runtime.h>

using U8Array = uint8_t *;
class U8ArraySoa : public RGB<U8Array> {
public:
    explicit U8ArraySoa(const U8VectorSoa &soa) {// NOLINT(cppcoreguidelines-pro-type-member-init)
        size_t n_pixels = soa.r.size();
        CHECK(cudaMalloc(&r, n_pixels * sizeof(uint8_t)));
        CHECK(cudaMalloc(&g, n_pixels * sizeof(uint8_t)));
        CHECK(cudaMalloc(&b, n_pixels * sizeof(uint8_t)));

        CHECK(cudaMemcpy(r, soa.r.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(g, soa.g.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b, soa.b.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    ~U8ArraySoa() {
        CHECK(cudaFree(r));
        CHECK(cudaFree(g));
        CHECK(cudaFree(b));
    }
};

#endif//ACA_CUDA_GPU_TYPES_H
