#ifndef ACA_CUDA_CONSTRUCTION_CUH
#define ACA_CUDA_CONSTRUCTION_CUH

#include "gpu_types.h"
#include "node.h"

#include <cstdio>

__global__ void build_quadtree_device(U8ArraySoa soa, Node *g_nodes, int tree_height, int n_rows, int n_cols, double detail_threshold);

__host__ void build_quadtree_host(Node *quadtree_nodes, int depth, int height, double detail_threshold);

#endif//ACA_CUDA_CONSTRUCTION_CUH
