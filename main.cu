#include "flatten.h"
#include "gpu_types.h"
#include "node.h"

#include <argparse/argparse.hpp>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>

void CHECK(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

/**
 * Compute base 4 integer logarithm of n
 * @param n
 * @return
 */
__device__ __host__ int log4(int n) {
    return static_cast<int>(std::log2(n) / 2);
}

__device__ __host__ bool should_merge(double detail_threshold, const RGB<double> &std) {
    return std.r <= detail_threshold ||
           std.g <= detail_threshold ||
           std.b <= detail_threshold;
}

// source
// https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation/442050#442050
__device__ __host__ RGB<double> combine_means(const Node &nw, const Node &ne, const Node &se, const Node &sw) {
    int pixels = nw.n_rows * nw.n_cols + ne.n_rows * ne.n_cols + se.n_rows * se.n_cols + sw.n_rows * sw.n_cols;
    auto nw_mean = nw.m_mean,
         ne_mean = ne.m_mean,
         se_mean = se.m_mean,
         sw_mean = sw.m_mean;
    auto nw_pixels = nw.n_rows * nw.n_cols,
         ne_pixels = ne.n_rows * ne.n_cols,
         se_pixels = se.n_rows * se.n_cols,
         sw_pixels = sw.n_rows * nw.n_cols;
    return {
            (nw_mean.r * nw_pixels + ne_mean.r * ne_pixels + se_mean.r * se_pixels + sw_mean.r * sw_pixels) / pixels,
            (nw_mean.g * nw_pixels + ne_mean.g * ne_pixels + se_mean.g * se_pixels + sw_mean.g * sw_pixels) / pixels,
            (nw_mean.b * nw_pixels + ne_mean.b * ne_pixels + se_mean.b * se_pixels + sw_mean.b * sw_pixels) / pixels,
    };
}

__device__ __host__ RGB<double> combine_stds(const Node &nw, const Node &ne, const Node &se, const Node &sw, const RGB<double> &mean) {
    int pixels = nw.n_rows * nw.n_cols + ne.n_rows * ne.n_cols + se.n_rows * se.n_cols + sw.n_rows * sw.n_cols;
    auto nw_mean = nw.m_mean,
         ne_mean = ne.m_mean,
         se_mean = se.m_mean,
         sw_mean = sw.m_mean;
    auto nw_std = nw.m_std,
         ne_std = ne.m_std,
         se_std = se.m_std,
         sw_std = sw.m_std;
    auto nw_pixels = nw.n_rows * nw.n_cols,
         ne_pixels = ne.n_rows * ne.n_cols,
         se_pixels = se.n_rows * se.n_cols,
         sw_pixels = sw.n_rows * nw.n_cols;

    return {
            std::sqrt(
                    (std::pow(nw_std.r, 2) * nw_pixels + nw_pixels * std::pow(mean.r - nw_mean.r, 2) +
                     std::pow(ne_std.r, 2) * ne_pixels + ne_pixels * std::pow(mean.r - ne_mean.r, 2) +
                     std::pow(se_std.r, 2) * se_pixels + se_pixels * std::pow(mean.r - se_mean.r, 2) +
                     std::pow(sw_std.r, 2) * sw_pixels + sw_pixels * std::pow(mean.r - sw_mean.r, 2)) /
                    pixels),

            std::sqrt(
                    (std::pow(nw_std.g, 2) * nw_pixels + nw_pixels * std::pow(mean.g - nw_mean.g, 2) +
                     std::pow(ne_std.g, 2) * ne_pixels + ne_pixels * std::pow(mean.g - ne_mean.g, 2) +
                     std::pow(se_std.g, 2) * se_pixels + se_pixels * std::pow(mean.g - se_mean.g, 2) +
                     std::pow(sw_std.g, 2) * sw_pixels + sw_pixels * std::pow(mean.g - sw_mean.g, 2)) /
                    pixels),

            std::sqrt(
                    (std::pow(nw_std.b, 2) * nw_pixels + nw_pixels * std::pow(mean.b - nw_mean.b, 2) +
                     std::pow(ne_std.b, 2) * ne_pixels + ne_pixels * std::pow(mean.b - ne_mean.b, 2) +
                     std::pow(se_std.b, 2) * se_pixels + se_pixels * std::pow(mean.b - se_mean.b, 2) +
                     std::pow(sw_std.b, 2) * sw_pixels + sw_pixels * std::pow(mean.b - sw_mean.b, 2)) /
                    pixels),
    };
}

__device__ __host__ Node make_internal_node(Node &nw, Node &ne, Node &se, Node &sw, int detail_threshold) {
    int height = nw.height + 1;
    int depth = nw.depth - 1;
    int i = nw.i;
    int j = nw.j;
    int n_rows = nw.n_rows + sw.n_rows;
    int n_cols = nw.n_cols + ne.n_cols;
    auto mean = combine_means(nw, ne, se, sw);
    auto std = combine_stds(nw, ne, se, sw, mean);
    auto color = Pixel{static_cast<uint8_t>(mean.r), static_cast<uint8_t>(mean.g), static_cast<uint8_t>(mean.b)};
    bool is_leaf = should_merge(detail_threshold, std);

    return {height, depth, i, j, n_rows, n_cols, color, mean, std, is_leaf};
}


__device__ void init_quadtree_leaves(U8ArraySoa soa, Node *quadtree_nodes, int tree_height, int n_rows, int n_cols) {
    unsigned int tid = threadIdx.x;
    auto r = soa.r + blockIdx.x * blockDim.x;
    auto g = soa.g + blockIdx.x * blockDim.x;
    auto b = soa.b + blockIdx.x * blockDim.x;

    int offset = static_cast<int>(pow(4, tree_height) / 3);

    // Skip internal nodes
    Node *leaves = quadtree_nodes + offset;
    // Points to the position in the leaves subarray where this block will start to write
    leaves = leaves + blockIdx.x * blockDim.x;

    int pos = tid + blockIdx.x * blockDim.x;
    int i = pos / n_cols;
    int j = pos % n_cols;

    Pixel color = {r[tid], g[tid], b[tid]};
    RGB<double> mean = {double(r[tid]), double(g[tid]), double(b[tid])};
    RGB<double> std = {0.0, 0.0, 0.0};

    leaves[tid] = Node(0, tree_height, i, j, 1, 1, color, mean, std, true);
}

/**
 *
 * @param soa
 * @param g_nodes Pointer to the whole array of nodes implementing the quadtree
 * @param tree_height
 * @param n_rows
 * @param n_cols
 */
__global__ void build_quadtree(U8ArraySoa soa, Node *g_nodes, int tree_height, int n_rows, int n_cols, int detail_threshold) {
#ifndef NDEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Build quadtree called. Block id: %d, thread id: %d\n", blockIdx.x, threadIdx.x);
    }
#endif

    init_quadtree_leaves(soa, g_nodes, tree_height, n_rows, n_cols);
    __syncthreads();
#ifndef NDEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Leaves have been init\n");
    }
#endif


    unsigned int tid = threadIdx.x;

    // We want to build the quadtree with an iterative reduction that
    // starts from the leaf nodes and proceeds towards the root node,
    // progressively building the intermediate levels.
    // The iteration implementing the reduction stops when
    // the number of nodes at some level is equal to the number of blocks;
    // this happens when depth reaches the following value:
    const int min_depth = tree_height - log4(blockDim.x);
#ifndef NDEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("min depth: %d\n", min_depth);
    }
#endif

    for (int depth = tree_height - 1,// We have already init the leaves of the tree, so we start at the level above
         n_active_threads_per_block = blockDim.x / 4;
         depth >= min_depth;// the stopping condition could also be n_active_threads_per_block > 0
         depth--, n_active_threads_per_block /= 4) {

        int block_offset = n_active_threads_per_block * blockIdx.x;
        // il thread corrente ha 4 nodi da processare livello corrente.
        // Questo si verifica quando l'id del thread è < al numero di thread che operano in un blocco a questo livello
        if (tid < n_active_threads_per_block) {

            int levels_offset = (static_cast<int>(pow(4, depth)) - 1) / 3;
            int write_offset = levels_offset + block_offset + tid;
            // Pointer in g_nodes where to write the result of the reduction
            Node *write_position = g_nodes + write_offset;

            int read_offset = 4 * write_offset + 1;
            // Pointer in g_nodes where to read the four nodes to reduce
            Node *read_position = g_nodes + read_offset;

#ifndef NDEBUG
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
                    detail_threshold);
        }
        __syncthreads();
    }
}

__host__ void finish_build_quadtree(Node *quadtree_nodes, int depth, int detail_threshold) {
    for (int stride = pow(4, depth); stride > 0; stride /= 4) {
        int write_offset = (stride - 1) / 3;
        int read_offset = write_offset + stride;
        Node *write_position = quadtree_nodes + write_offset;
        Node *read_position = quadtree_nodes + read_offset;
        for (int write_idx = 0, read_idx = 0; write_idx < stride; write_idx++, read_idx += 4) {
            write_position[write_idx] = make_internal_node(
                    read_position[read_idx + 0],
                    read_position[read_idx + 1],
                    read_position[read_idx + 2],
                    read_position[read_idx + 3],
                    detail_threshold);
        }
    }
}

bool is_power_of_four(int n) {
    return n == pow(4, log4(n));
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser app("jqc");
    app.add_description("A JPEG compressor based on the quadtree algorithm");

    app.add_argument("input")
            .required()
            .help("specify the input file");
    app.add_argument("--max-depth")
            .scan<'d', int>()
            .default_value(8)
            .help("specify the max depth");
    app.add_argument("--detail-threshold")
            .scan<'d', int>()
            .default_value(13)
            .help("specify the detail threshold");
    app.add_argument("--reduction")
            .scan<'d', int>()
            .default_value(0)
            .help("specify the reduction");

    try {
        app.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << app;
        std::exit(EXIT_FAILURE);
    }

    auto input = app.get("input");
    auto max_depth = app.get<int>("--max-depth");
    auto detail_threshold = app.get<int>("--detail-threshold");

    // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    // %o Elapsed time in milliseconds since previous message
    // %^ start color range (can be used only once)
    // %l The log level of the message
    // %$ end color range (for example %^[+++]%$ %v) (can be used only once)
    // %v The actual text to log
    spdlog::set_pattern("[elapsed: %o ms] [%^%l%$] %v");

#ifdef NDEBUG
    spdlog::warn("jqc compiled in RELEASE mode");
#else
    spdlog::warn("jqc compiled in DEBUG mode");
#endif

    spdlog::info("Reading {}...", input);

    int n_cols, n_rows, n;
    uint8_t *pixels = stbi_load(input.c_str(), &n_cols, &n_rows, &n, 3);
    if (!pixels) {
        spdlog::error("Could not open file " + input);
        std::exit(EXIT_FAILURE);
    }

    spdlog::info("Image is {}x{} px", n_rows, n_cols);

    if (n_cols != n_rows) {
        spdlog::error("Image is not square");
        std::exit(EXIT_FAILURE);
    }
    if (!is_power_of_four(n_rows) || !is_power_of_four(n_cols)) {
        spdlog::error("The number of pixels on the sides of the image is not a power of four");
        std::exit(EXIT_FAILURE);
    }

    int n_pixels = n_rows * n_cols;

    spdlog::info("Flattening image...");
    auto h_color_soa = flatten(pixels, n_pixels);

    spdlog::info("Copying pixels onto device...");
    U8ArraySoa d_color_soa;

    CHECK(cudaMalloc(&d_color_soa.r, n_pixels * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_color_soa.g, n_pixels * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_color_soa.b, n_pixels * sizeof(uint8_t)));

    CHECK(cudaMemcpy(d_color_soa.r, h_color_soa.r.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_color_soa.g, h_color_soa.g.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_color_soa.b, h_color_soa.b.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    spdlog::info("Building quadtree leaves...");

    int tree_height = log4(n_pixels);
    int n_nodes = static_cast<int>(std::pow(4, tree_height + 1) / 3);
    Node *d_quadtree_nodes = NULL;
    CHECK(cudaMalloc(&d_quadtree_nodes, n_nodes * sizeof(Node)));

    dim3 block(16);               // number of threads per block
    dim3 grid(n_pixels / block.x);// number of blocks
    build_quadtree<<<grid, block>>>(d_color_soa, d_quadtree_nodes, tree_height, n_rows, n_cols, detail_threshold);
    CHECK(cudaDeviceSynchronize());

    spdlog::info("Copying pixels back to host...");

    Node *h_quadtree_nodes = static_cast<Node *>(malloc(n_nodes * sizeof(uint8_t)));
    CHECK(cudaMemcpy(h_quadtree_nodes, d_quadtree_nodes, n_nodes * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    int from_depth = tree_height - log4(block.x) - 1;
    finish_build_quadtree(h_quadtree_nodes, from_depth, detail_threshold);

    CHECK(cudaFree(d_color_soa.r));
    CHECK(cudaFree(d_color_soa.g));
    CHECK(cudaFree(d_color_soa.b));

    CHECK(cudaFree(d_quadtree_nodes));
    free(h_quadtree_nodes);

    spdlog::info("Writing output file...", input);
    stbi_write_jpg("result.jpg", n_cols, n_rows, 3, pixels, 100);

    stbi_image_free(pixels);

    spdlog::info("Done.");
}
