#ifndef ACA_CUDA_NODE_H
#define ACA_CUDA_NODE_H

#include "pixel.h"
class Node {
public:
    __host__ __device__ Node(const int height, const int depth, const int i, const int j, const int nRows, const int nCols, const Pixel &mColor, const RGB<double> &mMean, const RGB<double> &mStd, bool isLeaf)
        : height(height), depth(depth), i(i), j(j), n_rows(nRows), n_cols(nCols), m_color(mColor), m_mean(mMean), m_std(mStd), is_leaf(isLeaf) {}
    int height;
    int depth;

    int i;
    int j;
    int n_rows;
    int n_cols;

    Pixel m_color;
    RGB<double> m_mean;
    RGB<double> m_std;

    bool is_leaf;
};

#endif//ACA_CUDA_NODE_H
