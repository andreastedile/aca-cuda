#ifndef ACA_CUDA_FLATTEN_H
#define ACA_CUDA_FLATTEN_H

#include "soa.h"

#include <vector>

using ColorArray = std::vector<color_t>;
using ColorSoa = PixelSoa<ColorArray>;

ColorSoa flatten(const uint8_t *pixels, int n_pixels);

#endif//ACA_CUDA_FLATTEN_H
