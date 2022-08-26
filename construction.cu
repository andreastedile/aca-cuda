#include "construction.cuh"
#include "utils.h"

#include <omp.h>
#include <spdlog/spdlog.h>
#ifdef DEBUG_DEVICE_CONSTRUCTION
#include <cstdio>
#endif


__device__ __host__ bool should_merge(double detail_threshold, const RGB<double> &std) {
    return std.r <= detail_threshold ||
           std.g <= detail_threshold ||
           std.b <= detail_threshold;
}

// source
// https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation/442050#442050
__device__ __host__ RGB<double> combine_means(const Node &nw, const Node &ne, const Node &se, const Node &sw, int n_pixels_subquadrant) {
    int n_pixels = 4 * n_pixels_subquadrant;
    RGB<double> nw_mean = nw.m_mean,
                ne_mean = ne.m_mean,
                se_mean = se.m_mean,
                sw_mean = sw.m_mean;
    return {
            (nw_mean.r * n_pixels_subquadrant + ne_mean.r * n_pixels_subquadrant + se_mean.r * n_pixels_subquadrant + sw_mean.r * n_pixels_subquadrant) / n_pixels,
            (nw_mean.g * n_pixels_subquadrant + ne_mean.g * n_pixels_subquadrant + se_mean.g * n_pixels_subquadrant + sw_mean.g * n_pixels_subquadrant) / n_pixels,
            (nw_mean.b * n_pixels_subquadrant + ne_mean.b * n_pixels_subquadrant + se_mean.b * n_pixels_subquadrant + sw_mean.b * n_pixels_subquadrant) / n_pixels,
    };
}

__device__ __host__ RGB<double> combine_stds(const Node &nw, const Node &ne, const Node &se, const Node &sw, const RGB<double> &mean, int n_pixels_subquadrant) {
    int n_pixels = 4 * n_pixels_subquadrant;
    RGB<double> nw_mean = nw.m_mean,
                ne_mean = ne.m_mean,
                se_mean = se.m_mean,
                sw_mean = sw.m_mean;
    RGB<double> nw_std = nw.m_std,
                ne_std = ne.m_std,
                se_std = se.m_std,
                sw_std = sw.m_std;

    auto combine = [&](double nw_std, double ne_std, double se_std, double sw_std,
                       double nw_mean, double ne_mean, double se_mean, double sw_mean,
                       double mean) {
        return std::sqrt(
                (std::pow(nw_std, 2) * n_pixels_subquadrant + n_pixels_subquadrant * std::pow(mean - nw_mean, 2) +
                 std::pow(ne_std, 2) * n_pixels_subquadrant + n_pixels_subquadrant * std::pow(mean - ne_mean, 2) +
                 std::pow(se_std, 2) * n_pixels_subquadrant + n_pixels_subquadrant * std::pow(mean - se_mean, 2) +
                 std::pow(sw_std, 2) * n_pixels_subquadrant + n_pixels_subquadrant * std::pow(mean - sw_mean, 2)) /
                n_pixels);
    };

    return {
            combine(nw_std.r, ne_std.r, se_std.r, sw_std.r, nw_mean.r, ne_mean.r, se_mean.r, sw_mean.r, mean.r),
            combine(nw_std.g, ne_std.g, se_std.g, sw_std.g, nw_mean.g, ne_mean.g, se_mean.g, sw_mean.g, mean.g),
            combine(nw_std.b, ne_std.b, se_std.b, sw_std.b, nw_mean.b, ne_mean.b, se_mean.b, sw_mean.b, mean.b),
    };
}


__device__ __host__ Node make_internal_node(Node &nw, Node &ne, Node &se, Node &sw, int n_pixels_subquadrant, double detail_threshold) {
    auto mean = combine_means(nw, ne, se, sw, n_pixels_subquadrant);
    auto std = combine_stds(nw, ne, se, sw, mean, n_pixels_subquadrant);
    if (should_merge(detail_threshold, std)) {
        return {mean, std, Node::Type::LEAF};
    } else {
        return {mean, std, Node::Type::FORK};
    }
}


__device__ void init_quadtree_leaves(U8ArraySoa soa, Node *quadtree_nodes, int tree_height) {
    int n_higher_nodes = (pow4(tree_height) - 1) / 3;
    int block_offset = blockIdx.x * blockDim.x;

    Node *write_ptr = quadtree_nodes + n_higher_nodes + block_offset + threadIdx.x;

    auto r_read_ptr = soa.r + block_offset + threadIdx.x;
    auto g_read_ptr = soa.g + block_offset + threadIdx.x;
    auto b_read_ptr = soa.b + block_offset + threadIdx.x;

#ifdef DEBUG_DEVICE_CONSTRUCTION
    printf("[%3d/%3d] n_higher_nodes: %3d, block_offset: %3d → read @ %3d, write @ %3d\n",
           blockIdx.x, threadIdx.x,
           n_higher_nodes, block_offset,
           block_offset + threadIdx.x,
           n_higher_nodes + block_offset + threadIdx.x);
#endif

    RGB<double> mean = {double(*r_read_ptr), double(*g_read_ptr), double(*b_read_ptr)};
    RGB<double> std = {0.0, 0.0, 0.0};
    *write_ptr = Node(mean, std, Node::Type::LEAF);
}

__host__ void build_quadtree_host(Node *quadtree_nodes, int depth, int height, double detail_threshold) {
#pragma omp parallel num_threads(8)
    {
#ifdef DEBUG_HOST_CONSTRUCTION
        spdlog::debug("Thread {}", omp_get_thread_num());
#endif
        for (int stride = pow4(depth), subquadrant_n_pixels = pow4(height - 1);
             stride > 0;
             stride /= 4, subquadrant_n_pixels *= 4) {
            int write_offset = (stride - 1) / 3;
            int read_offset = write_offset + stride;
            Node *write_position = quadtree_nodes + write_offset;
            Node *read_position = quadtree_nodes + read_offset;
#pragma omp for
            for (int write_idx = 0; write_idx < stride; write_idx++) {
                int read_idx = write_idx * 4;
                write_position[write_idx] = make_internal_node(
                        read_position[read_idx + 0],
                        read_position[read_idx + 1],
                        read_position[read_idx + 2],
                        read_position[read_idx + 3],
                        subquadrant_n_pixels,
                        detail_threshold);
            }
        }
    }
}


/**
*
* @param soa
* @param g_nodes Pointer to the whole array of nodes implementing the quadtree
* @param tree_height
* @param n_rows
* @param n_cols
*/
__global__ void build_quadtree_device(U8ArraySoa soa, Node *g_nodes, int tree_height, double detail_threshold) {
    init_quadtree_leaves(soa, g_nodes, tree_height);
    __syncthreads();

    unsigned int tid = threadIdx.x;

    // We want to build the quadtree with an iterative reduction that
    // starts from the leaf nodes and proceeds towards the root node,
    // progressively building the intermediate levels.
    // The iteration implementing the reduction stops when
    // the number of nodes at some level is equal to the number of blocks;
    // this happens when depth reaches the following value:
    const int min_depth = tree_height - log4(blockDim.x);
#ifdef DEBUG_DEVICE_CONSTRUCTION
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("min depth: %d\n", min_depth);
    }
#endif

    for (int depth = tree_height - 1,// We have already init the leaves of the tree, so we start at the level above
         n_active_threads_per_block = blockDim.x / 4,
             subquadrant_n_pixels = 1;
         depth >= min_depth;// the stopping condition could also be n_active_threads_per_block > 0
         depth--, n_active_threads_per_block /= 4,
             subquadrant_n_pixels *= 4) {

        int block_offset = n_active_threads_per_block * blockIdx.x;

        // il thread corrente ha 4 nodi da processare livello corrente.
        // Questo si verifica quando l'id del thread è < al numero di thread che operano in un blocco a questo livello
        if (tid < n_active_threads_per_block) {

            int levels_offset = (pow4(depth) - 1) / 3;
            int write_offset = levels_offset + block_offset + tid;
            // Pointer in g_nodes where to write the result of the reduction
            Node *write_position = g_nodes + write_offset;

            int read_offset = 4 * write_offset + 1;
            // Pointer in g_nodes where to read the four nodes to reduce
            Node *read_position = g_nodes + read_offset;

#ifdef DEBUG_DEVICE_CONSTRUCTION
            printf("[block %d / thread %d]: "
                   "depth: %d, "
                   "block offset: %d, "
                   "depth >= min depth (%d): %d, "
                   "block offset: %d, "
                   "write_pos: %d, "
                   "read_pos: %d\n",
                   blockIdx.x,
                   tid,
                   depth,
                   block_offset,
                   min_depth, depth >= min_depth,
                   block_offset,
                   write_offset,
                   read_offset);
#endif

            *write_position = make_internal_node(
                    read_position[0],
                    read_position[1],
                    read_position[2],
                    read_position[3],
                    subquadrant_n_pixels,
                    detail_threshold);
        }
        __syncthreads();
    }
}