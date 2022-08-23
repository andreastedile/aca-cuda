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
    // https://github.com/NVIDIA/libcudacxx/issues/162
    enum class Type {
        LEAF,
        FORK
    };

    CUDA_HOSTDEV Node(const RGB<double> &mMean, const RGB<double> &mStd, Type type)
        : m_mean(mMean), m_std(mStd), m_type(type) {}

    RGB<double> m_mean;
    RGB<double> m_std;

    Type m_type;

    [[nodiscard]] Pixel color() const {
        return {(uint8_t) m_mean.r, (uint8_t) m_mean.g, (uint8_t) m_mean.b};
    }
};

#endif//ACA_CUDA_NODE_H
