#include "colorization.h"

#include <spdlog/spdlog.h>

void colorize_impl(uint8_t *pixels, int N_COLS, const Node *quadtree, int q_idx, int node_i, int node_j, int n_rows, int n_cols) {
    spdlog::debug("q_idx: {}", q_idx);

    if (quadtree[q_idx].is_leaf) {
        const auto i_from = node_i;
        const auto j_from = node_j;

        const auto i_to = i_from + n_rows;
        const auto j_to = j_from + n_cols;

        spdlog::debug("is leaf.\n"
                      "i from = {}\n"
                      "i to = {}\n"
                      "j from = {}\n"
                      "j to = {}\n",
                      i_from, i_to, j_from, j_to);

        for (int i = i_from; i < i_to; i++) {
            for (auto j = j_from; j < j_to; j++) {
                spdlog::debug("i = {}, j = {},\n"
                              "(i * N_COLS + j) * 3 + 0 = {}\n",
                              "(i * N_COLS + j) * 3 + 1 = {}\n"
                              "(i * N_COLS + j) * 3 + 2 = {}\n",
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
                      "4 * q_idx + 1 = {}\n"
                      "4 * q_idx + 2 = {}\n"
                      "4 * q_idx + 3 = {}\n"
                      "4 * q_idx + 4 = {}\n",
                      4 * q_idx + 1,
                      4 * q_idx + 2,
                      4 * q_idx + 3,
                      4 * q_idx + 4);
        // clang-format off
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 1, node_i,              node_j,              n_rows / 2, n_cols / 2);// NW
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 2, node_i,              node_j + n_cols / 2, n_rows / 2, n_cols / 2);// NE
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 3, node_i + n_rows / 2, node_j + n_cols / 2, n_rows / 2, n_cols / 2);// SE
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 4, node_i + n_rows / 2, node_j,              n_rows / 2, n_cols / 2);// SW
        // clang-format on
    }
}

void colorize(uint8_t *pixels, int n_rows, int n_cols, const Node *quadtree_nodes) {
    colorize_impl(pixels, n_cols, quadtree_nodes, 0, 0, 0, n_rows, n_cols);
}