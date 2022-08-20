#include "flatten.h"
#include "gpu_types.h"
#include "node.h"

#include <argparse/argparse.hpp>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>
#include <cuda_runtime.h>

__device__ __host__ int log4(int n) {
    return static_cast<int>(log(n) / log(4));
}

__device__ Node make_internal_node(Node &nw, Node &ne, Node &se, Node &sw) {
    return Node(0, 0, 0, 0, 0, 0, {}, {}, {}, false);
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
__global__ void build_quadtree(U8ArraySoa soa, Node *g_nodes, int tree_height, int n_rows, int n_cols) {
    init_quadtree_leaves(soa, g_nodes, tree_height, n_rows, n_cols);

    unsigned int tid = threadIdx.x;

    // We want to build the quadtree with an iterative reduction that
    // starts from the leaf nodes and proceeds towards the root node,
    // progressively building the intermediate levels.
    // The iteration implementing the reduction stops when
    // the number of nodes at some level is equal to the number of blocks;
    // this happens when depth reaches the following value:
    const int min_depth = tree_height - log4(blockDim.x);

    for (int depth = tree_height - 1,// We have already init the leaves of the tree, so we start at the level above
         block_offset = blockDim.x * blockIdx.x / 4;
         depth >= min_depth;
         depth--, block_offset /= 4) {


        // il thread corrente ha 4 nodi da processare livello corrente.
        // Questo si verifica quando l'id del thread è < al numero di thread che operano in un blocco a questo livello
        if (tid < block_offset / blockIdx.x) {

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

            // fa il calcolo e lo salva
            *write_position = make_internal_node(
                    read_position[0],
                    read_position[1],
                    read_position[2],
                    read_position[3]);
        }
        __syncthreads();
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
    auto color_soa = flatten(pixels, n_pixels);

    spdlog::info("Copying pixels onto device...");
    U8ArraySoa device_soa;

    cudaMalloc(&device_soa.r, n_pixels * sizeof(uint8_t));
    cudaMalloc(&device_soa.g, n_pixels * sizeof(uint8_t));
    cudaMalloc(&device_soa.b, n_pixels * sizeof(uint8_t));

    cudaMemcpy(device_soa.r, color_soa.r.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_soa.g, color_soa.g.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_soa.b, color_soa.b.data(), n_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    spdlog::info("Init quadtree leaves...");

    int tree_height = log4(n_pixels);
    int n_nodes = static_cast<int>(std::pow(4, tree_height + 1) / 3);
    Node *quadtree_nodes = NULL;
    cudaMalloc(&quadtree_nodes, n_nodes * sizeof(Node));

    dim3 block(1024);
    dim3 grid((n_pixels + block.x - 1) / block.x);
    build_quadtree<<<grid, block>>>(device_soa, quadtree_nodes, tree_height, n_rows, n_cols);
    cudaDeviceSynchronize();

    spdlog::info("Copying pixels back to host...");

    cudaMemcpy(color_soa.r.data(), device_soa.r, n_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(color_soa.g.data(), device_soa.g, n_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(color_soa.b.data(), device_soa.b, n_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(device_soa.r);
    cudaFree(device_soa.g);
    cudaFree(device_soa.b);

    spdlog::info("Writing output file...", input);
    stbi_write_jpg("result.jpg", n_cols, n_rows, 3, pixels, 100);

    stbi_image_free(pixels);

    spdlog::info("Done.");
}