#ifndef ACA_CUDA_GPU_TYPES_H
#define ACA_CUDA_GPU_TYPES_H

#include "rgb.h"
#include <cstdint>
#include <vector>

using U8Array = uint8_t *;
using U8ArraySoa = RGB<U8Array>;

#endif//ACA_CUDA_GPU_TYPES_H
