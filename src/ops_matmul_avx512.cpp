#include "sys_kernel_dispatch.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

void cpu_matmul_avx512_bw(
    const float* a, 
    const float* scales, 
    const uint8_t* b, 
    const uint8_t* zp, 
    float* c_out, 
    uint32_t M, 
    uint32_t N, 
    uint32_t K, 
    uint32_t block_size
) {
    if (!a || !b || !c_out) return;

    uint32_t num_blocks_per_row = K / block_size;
    uint32_t bytes_per_row = K / 2;

#pragma omp parallel for schedule(static)
    for (int n = 0; n < (int)N; ++n) {
        const uint8_t* b_row = b + n * bytes_per_row;
        // Native FP32 read (loader already expanded it)
        const float* scale_row_fp32 = scales ? (scales + n * num_blocks_per_row) : nullptr;

        float sum = 0.0f;
        for (uint32_t blk = 0; blk < num_blocks_per_row; ++blk) {
            float sc = scale_row_fp32 ? scale_row_fp32[blk] : 1.0f;
            if (std::isnan(sc) || std::isinf(sc) || sc == 0.0f) sc = 1.0f;

            uint32_t k_start = blk * block_size;
            const uint8_t* b_blk = b_row + blk * (block_size / 2);
            const float* a_ptr = a + k_start;

            if (block_size == 32) {
                // INT8 + INT32 ACCUMULATION KERNEL (vpmaddubsw / vpaddd)
                alignas(64) uint8_t u8_act[32];
                alignas(64) int8_t  i8_wt[32];

                float max_abs_a = 1e-5f;
                for (int i = 0; i < 32; ++i) {
                    float abs_v = std::abs(a_ptr[i]);
                    if (abs_v > max_abs_a) max_abs_a = abs_v;
                }
                float act_scale = (max_abs_a > 1e-8f) ? (max_abs_a / 127.0f) : 1.0f;
                float inv_act_scale = 1.0f / act_scale;

                for (int i = 0; i < 32; ++i) {
                    u8_act[i] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(a_ptr[i] * inv_act_scale + 128.0f), 0, 255));
                }

                __m128i raw_16 = _mm_loadu_si128((const __m128i*)b_blk);
                __m128i low_nib = _mm_and_si128(raw_16, _mm_set1_epi8(0x0F));
                __m128i high_nib = _mm_and_si128(_mm_srli_epi16(raw_16, 4), _mm_set1_epi8(0x0F));

                __m128i nib_0_15 = _mm_unpacklo_epi8(low_nib, high_nib);
                __m128i nib_16_31 = _mm_unpackhi_epi8(low_nib, high_nib);

                __m128i sub_8 = _mm_set1_epi8(8);
                __m128i wt_0_15 = _mm_sub_epi8(nib_0_15, sub_8);
                __m128i wt_16_31 = _mm_sub_epi8(nib_16_31, sub_8);

                _mm_storeu_si128((__m128i*)i8_wt, wt_0_15);
                _mm_storeu_si128((__m128i*)(i8_wt + 16), wt_16_31);

                __m256i a_vec = _mm256_loadu_si256((const __m256i*)u8_act);
                __m256i w_vec = _mm256_loadu_si256((const __m256i*)i8_wt);

                __m256i dot16 = _mm256_maddubs_epi16(a_vec, w_vec);
                __m256i dot32 = _mm256_madd_epi16(dot16, _mm256_set1_epi16(1));

                int32_t int_sum = 0;
                alignas(32) int32_t d32[8];
                _mm256_storeu_si256((__m256i*)d32, dot32);
                for (int i = 0; i < 8; ++i) int_sum += d32[i];

                int32_t wt_sum = 0;
                for (int i = 0; i < 32; ++i) wt_sum += i8_wt[i];
                int32_t net_int_sum = int_sum - (128 * wt_sum);

                sum += static_cast<float>(net_int_sum) * (act_scale * sc);
            } else {
                float blk_sum = 0.0f;
                for (uint32_t i = 0; i < block_size; ++i) {
                    uint8_t byte_val = b_blk[i / 2];
                    uint8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
                    float w = (static_cast<float>(nibble) - 8.0f) * sc;
                    blk_sum += a_ptr[i] * w;
                }
                sum += blk_sum;
            }
        }
        c_out[n] = sum;
    }
}