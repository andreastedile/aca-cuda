#include "colorization.h"

#include <spdlog/spdlog.h>

void colorize_impl(uint8_t *pixels, int N_COLS, const Node *quadtree, int q_idx, int node_i, int node_j, int n_rows, int n_cols, int residual_levels) {
    spdlog::debug("q_idx: {}", q_idx);

    if (residual_levels <= 0 && quadtree[q_idx].is_leaf) {
        const auto i_from = node_i;
        const auto j_from = node_j;

        const auto i_to = i_from + n_rows;
        const auto j_to = j_from + n_cols;

        spdlog::debug("is leaf., "
                      "i from = {}, "
                      "i to = {}, "
                      "j from = {}, "
                      "j to = {}",
                      i_from, i_to, j_from, j_to);

        for (int i = i_from; i < i_to; i++) {
            for (auto j = j_from; j < j_to; j++) {
                spdlog::debug("i = {}, j = {}, "
                              "(i * N_COLS + j) * 3 + 0 = {}, ",
                              "(i * N_COLS + j) * 3 + 1 = {}, "
                              "(i * N_COLS + j) * 3 + 2 = {}",
                              i, j,
                              (i * N_COLS + j) * 3 + 0,
                              (i * N_COLS + j) * 3 + 1,
                              (i * N_COLS + j) * 3 + 2);
                pixels[(i * N_COLS + j) * 3 + 0] = quadtree[q_idx].color().r;
                pixels[(i * N_COLS + j) * 3 + 1] = quadtree[q_idx].color().g;
                pixels[(i * N_COLS + j) * 3 + 2] = quadtree[q_idx].color().b;
            }
        }
    } else {
        spdlog::debug("is fork.\n"
                      "4 * q_idx + 1 = {}, "
                      "4 * q_idx + 2 = {}, "
                      "4 * q_idx + 3 = {}, "
                      "4 * q_idx + 4 = {}",
                      4 * q_idx + 1,
                      4 * q_idx + 2,
                      4 * q_idx + 3,
                      4 * q_idx + 4);
        // clang-format off
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 1, node_i,              node_j,              n_rows / 2, n_cols / 2, residual_levels-1);// NW
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 2, node_i,              node_j + n_cols / 2, n_rows / 2, n_cols / 2, residual_levels-1);// NE
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 3, node_i + n_rows / 2, node_j + n_cols / 2, n_rows / 2, n_cols / 2, residual_levels-1);// SE
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 4, node_i + n_rows / 2, node_j,              n_rows / 2, n_cols / 2, residual_levels-1);// SW
        // clang-format on
    }
}

void colorize(uint8_t *pixels, int n_rows, int n_cols, const Node *quadtree_nodes, int min_depth) {
    colorize_impl(pixels, n_cols, quadtree_nodes, 0, 0, 0, n_rows, n_cols, min_depth);
}