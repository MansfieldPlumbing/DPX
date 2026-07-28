#include "sys_types.h"
#include <execution>
#include <numeric>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void cpu_stft_exact(
    const float* sig, const float* win, float* out_ri, 
        uint32_t batch, uint32_t sig_len, uint32_t frame_step, uint32_t frame_length, bool onesided
        ) {
            uint32_t num_frames = (sig_len - frame_length) / frame_step + 1;
                uint32_t bins = onesided ? frame_length / 2 + 1 : frame_length;
                
    std::vector<double> cs(bins * frame_length), sn(bins * frame_length);
        for (uint32_t k = 0; k < bins; ++k) {
                for (uint32_t m = 0; m < frame_length; ++m) {
                            double ang = -2.0 * M_PI * k * m / frame_length;
                                        cs[k * frame_length + m] = std::cos(ang);
                                                    sn[k * frame_length + m] = std::sin(ang);
                                                            }
                                                                }
                                                                
    std::vector<uint32_t> frames_idx(batch * num_frames);
        std::iota(frames_idx.begin(), frames_idx.end(), 0);
        
    std::for_each(std::execution::par_unseq, frames_idx.begin(), frames_idx.end(), [&](uint32_t bfi) {
            uint32_t b = bfi / num_frames; uint32_t f = bfi % num_frames;
                    uint64_t s_base = b * sig_len + f * frame_step;
                            for (uint32_t k = 0; k < bins; ++k) {
                                        double re = 0.0, im = 0.0;
                                                    uint32_t kb = k * frame_length;
                                                                for (uint32_t m = 0; m < frame_length; ++m) {
                                                                                double xv = sig[s_base + m] * (win ? win[m] : 1.0f);
                                                                                                re += xv * cs[kb + m]; im += xv * sn[kb + m];
                                                                                                            }
                                                                                                                        uint64_t ob = (((uint64_t)b * num_frames + f) * bins + k) * 2;
                                                                                                                                    out_ri[ob] = static_cast<float>(re);
                                                                                                                                                out_ri[ob + 1] = static_cast<float>(im);
                                                                                                                                                        }
                                                                                                                                                            });
                                                                                                                                                            }
                                                                                                                                                            