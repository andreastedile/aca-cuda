#include "flatten.h"

#include <argparse/argparse.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>

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
        std::exit(1);
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
        throw std::runtime_error("Could not open file " + input);
    }

    spdlog::info("Image is {}x{} px", n_rows, n_cols);

    spdlog::info("Flattening image...");
    auto color_soa = flatten(pixels, n_rows * n_cols);

    spdlog::info("Writing output file...", input);
    stbi_write_jpg("result.jpg", n_cols, n_rows, 3, pixels, 100);

    stbi_image_free(pixels);

    spdlog::info("Done.");
}