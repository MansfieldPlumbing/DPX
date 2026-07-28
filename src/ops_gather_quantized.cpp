#include "sys_kernel_dispatch.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

void cpu_gather_block_quantized(
    const uint8_t* data, 
    const int32_t* indices, 
    uint32_t num_indices, 
    const float* scales, 
    const uint8_t* zp, 
    float* out, 
    uint32_t num_embeds, 
    uint32_t embed_dim, 
    uint32_t block_size
) {
    if (!data || !indices || !out) return;
    
    uint32_t num_blocks = embed_dim / block_size;
    uint32_t row_bytes = embed_dim / 2;

    for (uint32_t t = 0; t < num_indices; t++) {
        // Safe token index resolution (handles both float and int32_t representations)
        int32_t raw_val = indices[t];
        float float_val = reinterpret_cast<const float*>(indices)[t];
        int32_t token_id = 0;

        if (raw_val >= 0 && raw_val < (int32_t)num_embeds) {
            token_id = raw_val;
        } else if (!std::isnan(float_val) && float_val >= 0.0f && float_val < static_cast<float>(num_embeds)) {
            token_id = static_cast<int32_t>(float_val);
        }

        if (token_id < 0 || token_id >= (int32_t)num_embeds) token_id = 0;

        const uint8_t* row_data = data + (uint64_t)token_id * row_bytes;
        
        // Native FP32 read (loader already expanded it)
        const float* row_scales_fp32 = scales ? (scales + (uint64_t)token_id * num_blocks) : nullptr;
        float* out_row = out + (uint64_t)t * embed_dim;

        for (uint32_t i = 0; i < embed_dim; i++) {
            uint8_t packed = row_data[i / 2];
            uint8_t nibble = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
            
            uint32_t blk_idx = i / block_size;
            float scale = row_scales_fp32 ? row_scales_fp32[blk_idx] : 1.0f;
            if (std::isnan(scale) || std::isinf(scale) || scale == 0.0f) scale = 1.0f;

            float dequant = (static_cast<float>(nibble) - 8.0f) * scale;
            out_row[i] = dequant;
        }
    }
}