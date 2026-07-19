#include "sys_types.h"
#include <cmath>
#include <execution>
#include <numeric>
#include <algorithm>
#include <vector>

void cpu_rms_norm(const float* X, const float* weight, float* out, uint32_t outer, uint32_t inner, float eps) {
    std::vector<uint32_t> indices(outer);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](uint32_t ob) {
        uint64_t bI = ob * inner;
        double ss = 0.0;
        for (uint32_t i = 0; i < inner; ++i) ss += X[bI + i] * X[bI + i];
        float inv = static_cast<float>(1.0 / std::sqrt(ss / inner + eps));
        for (uint32_t i = 0; i < inner; ++i) out[bI + i] = static_cast<float>(X[bI + i] * inv * weight[i % inner]);
    });
}
