#ifndef ACA_CUDA_FLATTEN_H
#define ACA_CUDA_FLATTEN_H

#include "cpu_types.h"

U8VectorSoa flatten(const uint8_t *pixels, int n_pixels);

#endif//ACA_CUDA_FLATTEN_H
