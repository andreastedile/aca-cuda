#ifndef ACA_CUDA_NODE_H
#define ACA_CUDA_NODE_H

#include "pixel.h"
class Node {
    const int height;
    const int depth;

    const int i;
    const int j;
    const int n_rows;
    const int n_cols;

    Pixel m_color;
    RGB<double> m_mean;
    RGB<double> m_std;

    bool is_leaf;
};

#endif//ACA_CUDA_NODE_H
