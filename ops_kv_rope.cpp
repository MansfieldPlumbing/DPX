#include "sys_types.h"
#include <execution>
#include <numeric>
#include <vector>

void cpu_rotary_embedding(
    const float* in_ptr, const int64_t* pos_ids, const float* cos_cache, const float* sin_cache,
        float* out_ptr, uint32_t B, uint32_t S, uint32_t Nh, uint32_t Hd, bool interleaved
        ) {
            uint32_t half_d = Hd / 2;
                std::vector<uint32_t> tasks(B * S);
                    std::iota(tasks.begin(), tasks.end(), 0);
                    
    std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), [&](uint32_t bs) {
            uint32_t b = bs / S; uint32_t s = bs % S;
                    int64_t p = pos_ids[bs];
                            uint64_t cb = p * half_d; 
                                    for (uint32_t h = 0; h < Nh; ++h) {
                                                uint64_t bI = (((uint64_t)b * S + s) * Nh + h) * Hd;
                                                            for (uint32_t i = 0; i < half_d; ++i) {
                                                                            float c = cos_cache[cb + i]; float sn = sin_cache[cb + i];
                                                                                            if (interleaved) {
                                                                                                                float a0 = in_ptr[bI + 2 * i], a1 = in_ptr[bI + 2 * i + 1];
                                                                                                                                    out_ptr[bI + 2 * i] = a0 * c - a1 * sn;
                                                                                                                                                        out_ptr[bI + 2 * i + 1] = a1 * c + a0 * sn;
                                                                                                                                                                        } else {
                                                                                                                                                                                            float a0 = in_ptr[bI + i], a1 = in_ptr[bI + i + half_d];
                                                                                                                                                                                                                out_ptr[bI + i] = a0 * c - a1 * sn;
                                                                                                                                                                                                                                    out_ptr[bI + i + half_d] = a1 * c + a0 * sn;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                            });
                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                            