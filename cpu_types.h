#ifndef ACA_CUDA_CPU_TYPES_H
#define ACA_CUDA_CPU_TYPES_H

#include "rgb.h"
#include <cstdint>
#include <vector>

using U8Vector = std::vector<uint8_t>;
using U8VectorSoa = RGB<U8Vector>;

#endif//ACA_CUDA_CPU_TYPES_H
