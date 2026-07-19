#include "sys_types.h"
#include <algorithm>

void cpu_conv_transpose_1d(
    const float* X, const float* W, const float* B, float* Y,
        uint32_t batch, uint32_t in_C, uint32_t out_C, uint32_t in_S,
            uint32_t k_size, uint32_t stride, uint32_t pad
            ) {
                uint32_t out_S = (in_S - 1) * stride + k_size - 2 * pad;
                    std::fill_n(Y, batch * out_C * out_S, 0.0f);
                    
    for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t oc = 0; oc < out_C; ++oc) {
                        for (uint32_t ic = 0; ic < in_C; ++ic) {
                                        const float* x_slice = X + b * in_C * in_S + ic * in_S;
                                                        const float* w_slice = W + ic * out_C * k_size + oc * k_size;
                                                                        float* y_slice = Y + b * out_C * out_S + oc * out_S;
                                                                        
                for (uint32_t is = 0; is < in_S; ++is) {
                                    float x_val = x_slice[is];
                                                        if (x_val == 0.0f) continue;
                                                                            int32_t base_o = is * stride - pad;
                                                                                                for (uint32_t k = 0; k < k_size; ++k) {
                                                                                                                        int32_t o_pos = base_o + k;
                                                                                                                                                if (o_pos >= 0 && o_pos < (int32_t)out_S) {
                                                                                                                                                                            y_slice[o_pos] += x_val * w_slice[k];
                                                                                                                                                                                                    }
                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                if (B) {
                                                                                                                                                                                                                                                                                float bias = B[oc];
                                                                                                                                                                                                                                                                                                float* y_slice = Y + b * out_C * out_S + oc * out_S;
                                                                                                                                                                                                                                                                                                                for (uint32_t s = 0; s < out_S; ++s) y_slice[s] += bias;
                                                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                                                        