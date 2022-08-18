#include "flatten.h"

#include <cmath>
#include <cstdint>

void flatten_data_impl(const uint8_t *pixels,
                       ColorSoa &soa,
                       int i, int j,
                       int n_rows, int n_cols,
                       int idx,
                       int N_COLS) {
    if (n_cols == 1) {
        soa.r[idx] = pixels[(i * N_COLS + j) * 3 + 0];
        soa.g[idx] = pixels[(i * N_COLS + j) * 3 + 1];
        soa.b[idx] = pixels[(i * N_COLS + j) * 3 + 2];
    } else {
        auto q_rows = n_rows / 2,
             q_cols = n_cols / 2;
        auto q_size = q_rows * q_cols;
        // clang-format off
        flatten_data_impl(pixels, soa, i + 0,      j + 0,      q_rows, q_cols, idx + q_size * 0, N_COLS); // nw
        flatten_data_impl(pixels, soa, i + 0,      j + q_cols, q_rows, q_cols, idx + q_size * 1, N_COLS); // ne
        flatten_data_impl(pixels, soa, i + q_rows, j + 0,      q_rows, q_cols, idx + q_size * 2, N_COLS); // sw
        flatten_data_impl(pixels, soa, i + q_rows, j + q_cols, q_rows, q_cols, idx + q_size * 3, N_COLS); // se
        // clang-format on
    }
}

ColorSoa flatten(const uint8_t *pixels, int n_pixels) {
    auto soa = ColorSoa{ColorArray(n_pixels), ColorArray(n_pixels), ColorArray(n_pixels)};
    // Fixme: fix the order of the parameters in the function call
    flatten_data_impl(pixels, soa, 0, 0,
                      static_cast<int>(std::sqrt(n_pixels)),
                      static_cast<int>(std::sqrt(n_pixels)),
                      0,
                      static_cast<int>(std::sqrt(n_pixels)));
    return soa;
}