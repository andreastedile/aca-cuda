#ifndef ACA_CUDA_NODE_H
#define ACA_CUDA_NODE_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


#include "pixel.h"

class Node {
public:
    CUDA_HOSTDEV Node(const RGB<double> &mMean, const RGB<double> &mStd, bool isLeaf)
        : m_mean(mMean), m_std(mStd), is_leaf(isLeaf) {}

    RGB<double> m_mean;
    RGB<double> m_std;

    bool is_leaf;

    [[nodiscard]] Pixel color() const {
        return {(uint8_t) m_mean.r, (uint8_t) m_mean.g, (uint8_t) m_mean.b};
    }
};

#endif//ACA_CUDA_NODE_H
