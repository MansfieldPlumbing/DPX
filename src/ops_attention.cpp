#include "sys_types.h"
#include <cmath>
#include <execution>
#include <numeric>
#include <algorithm>
#include <vector>

#include <immintrin.h>

void cpu_rms_norm(const float* X, const float* weight, float* out, uint32_t outer, uint32_t inner, float eps, uint32_t weight_len) {
    if (!X || !out) return;
    if (weight_len == 0) weight_len = inner;

    std::vector<uint32_t> indices(outer);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](uint32_t ob) {
        uint64_t bI = ob * inner;
        double ss = 0.0;
        for (uint32_t i = 0; i < inner; ++i) ss += X[bI + i] * X[bI + i];
        float inv = static_cast<float>(1.0 / std::sqrt(ss / inner + eps));
        if (weight) {
            for (uint32_t i = 0; i < inner; ++i) {
                float w = weight[i % weight_len];
                out[bI + i] = static_cast<float>(X[bI + i] * inv * w);
            }
        } else {
            for (uint32_t i = 0; i < inner; ++i) {
                out[bI + i] = static_cast<float>(X[bI + i] * inv);
            }
        }
    });
}

void cpu_attention_gqa(const float* Q, const float* K, const float* V, float* out, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t seq_len) {
    uint32_t group_size = num_q_heads / (num_kv_heads > 0 ? num_kv_heads : 1);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (uint32_t qh = 0; qh < num_q_heads; ++qh) {
        uint32_t kvh = qh / (group_size > 0 ? group_size : 1);
        const float* q_ptr = Q + qh * head_dim;

        std::vector<float> scores(seq_len, 0.0f);
        float max_score = -1e9f;

        for (uint32_t s = 0; s < seq_len; ++s) {
            const float* k_ptr = K + (s * num_kv_heads + kvh) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q_ptr[d] * k_ptr[d];
            }
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }

        float sum_exp = 0.0f;
        for (uint32_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (uint32_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        float* out_ptr = out + qh * head_dim;
        std::fill_n(out_ptr, head_dim, 0.0f);

        for (uint32_t s = 0; s < seq_len; ++s) {
            const float* v_ptr = V + (s * num_kv_heads + kvh) * head_dim;
            float w = scores[s];
            for (uint32_t d = 0; d < head_dim; ++d) {
                out_ptr[d] += w * v_ptr[d];
            }
        }
    }
}
