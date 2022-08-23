#ifndef ACA_CUDA_COLORIZATION_H
#define ACA_CUDA_COLORIZATION_H

#include "node.h"

#include <cstdint>

void colorize(uint8_t *pixels, int n_rows, int n_cols, const Node *quadtree_nodes, int min_depth);

#endif// ACA_CUDA_COLORIZATION_H
