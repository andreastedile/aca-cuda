#include "colorization.h"

void colorize_impl(uint8_t *pixels, int N_COLS, const Node *quadtree, int q_idx) {
    if (quadtree[q_idx].is_leaf) {
        const auto i_from = quadtree[q_idx].i;
        const auto j_from = quadtree[q_idx].j;

        const auto i_to = i_from + quadtree[q_idx].n_rows;
        const auto j_to = j_from + quadtree[q_idx].n_cols;

        for (int i = i_from; i < i_to; i++) {
            for (auto j = j_from; j < j_to; j++) {
                pixels[(i * N_COLS + j) * 3 + 0] = quadtree[q_idx].m_color.r;
                pixels[(i * N_COLS + j) * 3 + 1] = quadtree[q_idx].m_color.g;
                pixels[(i * N_COLS + j) * 3 + 2] = quadtree[q_idx].m_color.b;
            }
        }
    } else {
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 1);
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 2);
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 3);
        colorize_impl(pixels, N_COLS, quadtree, 4 * q_idx + 4);
    }
}

void colorize(uint8_t *pixels, const Node *quadtree_nodes) {
    colorize_impl(pixels, quadtree_nodes[0].n_cols, quadtree_nodes, 0);
}