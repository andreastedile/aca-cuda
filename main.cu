#include "colorization.h"
#include "construction.cuh"
#include "flatten.h"
#include "gpu_types.h"
#include "node.h"
#include "utils.h"

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser app("jqc");
    app.add_description("A JPEG compressor based on the quadtree algorithm");

    app.add_argument("input")
            .required()
            .help("specify the input file");
    app.add_argument("-b", "--blocks")
            .scan<'d', int>()
            .required()
            .help("specify the number of blocks in the kernel launch");
    app.add_argument("-t", "--threads")
            .scan<'d', int>()
            .required()
            .help("specify the number of thread per blocks in the kernel launch");
    app.add_argument("-d", "--detail-threshold")
            .scan<'g', double>()
            .default_value(13.0)
            .help("specify the detail threshold");
    app.add_argument("--save-intermediate-levels")
            .default_value(false)
            .implicit_value(true)
            .help("save the intermediate levels to files");

    try {
        app.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << app;
        std::exit(EXIT_FAILURE);
    }

    auto input = app.get("input");
    int n_blocks = app.get<int>("--blocks");
    int n_threads_per_block = app.get<int>("--threads");
    auto detail_threshold = app.get<double>("--detail-threshold");
    auto save_intermediate_levels = app.get<bool>("--save-intermediate-levels");

    spdlog::set_pattern("[elapsed: %o ms] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);

#ifndef NDEBUG
    spdlog::warn("jqc compiled in DEBUG mode");
#endif

    spdlog::info("Reading {}...", input);

    int n_cols, n_rows, n, n_pixels;
    uint8_t *pixels = stbi_load(input.c_str(), &n_cols, &n_rows, &n, 3);
    if (!pixels) {
        spdlog::error("Could not open file " + input);
        std::exit(EXIT_FAILURE);
    } else {
        n_pixels = n_rows * n_cols;
        spdlog::info("Image is {}x{} px ({})", n_rows, n_cols, n_pixels);

        if (n_cols != n_rows) {
            spdlog::error("Image is not square");
            std::exit(EXIT_FAILURE);
        }
        if (!is_power_of_four(n_rows) || !is_power_of_four(n_cols)) {
            spdlog::error("The number of pixels on the sides of the image is not a power of four");
            std::exit(EXIT_FAILURE);
        }
    }

    if (n_blocks * n_threads_per_block != n_pixels) {
        spdlog::error("The number of blocks times the number of threads per block ({}) is not equal to the number of pixels of the image ({})", n_blocks * n_threads_per_block, n_pixels);
        std::exit(EXIT_FAILURE);
    }

    spdlog::info("Flattening the image...");
    U8VectorSoa h_color_soa = flatten(pixels, n_pixels);

    spdlog::info("Copying the pixels on the device...");
    U8ArraySoa d_color_soa(h_color_soa);

    // The first part of the quadtree is built on the device, and the remaining on the host.
    Node *d_quadtree_nodes;
    Node *h_quadtree_nodes;

    int tree_height = log4(n_pixels);
    int n_nodes = (pow4(tree_height + 1) - 1) / 3;

    spdlog::info("Allocating memory for the quadtree (height: {}, nodes: {})...", tree_height, n_nodes);
    CHECK(cudaMalloc(&d_quadtree_nodes, n_nodes * sizeof(Node)));

    dim3 threads(n_threads_per_block);
    dim3 blocks(n_pixels / threads.x);
    spdlog::info("Building the quadtreee on the device (blocks: {}, threads per block: {})...", blocks.x, threads.x);
    build_quadtree_device<<<blocks, threads>>>(d_color_soa, d_quadtree_nodes, tree_height, n_rows, n_cols, detail_threshold);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());

    spdlog::info("Copying the quadtree back to the host...");
    h_quadtree_nodes = static_cast<Node *>(malloc(n_nodes * sizeof(Node)));
    CHECK(cudaMemcpy(h_quadtree_nodes, d_quadtree_nodes, n_nodes * sizeof(Node), cudaMemcpyDeviceToHost));

    int from_depth = tree_height - log4(n_threads_per_block) - 1;
    spdlog::info("Building the remaining quadtree on the host (from depth: {})...", from_depth);
    build_quadtree_host(h_quadtree_nodes, from_depth, tree_height - from_depth, detail_threshold);

    if (save_intermediate_levels) {
        for (int i = 0; i < tree_height; i++) {
            spdlog::info("Coloring the image for level {}...", i);
            colorize(pixels, n_rows, n_cols, h_quadtree_nodes, i);

            spdlog::info("Writing the output file for level {}...", i);
            std::string filename = "level" + std::to_string(i) + ".jpg";
            stbi_write_jpg(filename.c_str(), n_cols, n_rows, 3, pixels, 100);
        }
    }

    spdlog::info("Coloring the resulting image...");
    colorize(pixels, n_rows, n_cols, h_quadtree_nodes, 0);

    spdlog::info("Writing the resulting output file...");
    stbi_write_jpg("result.jpg", n_cols, n_rows, 3, pixels, 100);

    d_color_soa.dispose();
    CHECK(cudaFree(d_quadtree_nodes));
    free(h_quadtree_nodes);

    stbi_image_free(pixels);

    spdlog::info("Done.");
}
