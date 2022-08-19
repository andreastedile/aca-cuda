#include "flatten.h"
#include "gpu_types.h"
#include "node.h"

#include <argparse/argparse.hpp>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>

__global__ void init_quadtree_leaves(U8ArraySoa soa, Node *quadtree_nodes, int tree_height, int n_rows, int n_cols) {
    unsigned int tid = threadIdx.x;
    auto r = soa.r + blockIdx.x * blockDim.x;
    auto g = soa.g + blockIdx.x * blockDim.x;
    auto b = soa.b + blockIdx.x * blockDim.x;

    int offset = static_cast<int>(std::pow(4, tree_height) / 3);

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

bool is_power_of_four(int n) {
    return n == pow(4, (log(n) / log(4)));
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

    int tree_height = static_cast<int>(std::log(n_pixels) / std::log(4));
    int n_nodes = static_cast<int>(std::pow(4, tree_height + 1) / 3);
    Node *quadtree_nodes = NULL;
    cudaMalloc(&quadtree_nodes, n_nodes * sizeof(Node));

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