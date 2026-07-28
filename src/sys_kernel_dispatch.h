#pragma once
#include <cstdint>

typedef void (*MatMulKernelFunc)(const float* a, const float* scales, const uint8_t* b, const uint8_t* zp, float* c_out, uint32_t M, uint32_t N, uint32_t K, uint32_t block_size);
typedef void (*GatherQuantizedFunc)(const uint8_t* data, const int32_t* indices, uint32_t num_indices, const float* scales, const uint8_t* zp, float* out, uint32_t num_embeds, uint32_t embed_dim, uint32_t block_size);

extern MatMulKernelFunc g_matmul_kernel;
extern GatherQuantizedFunc g_gather_kernel;

void dpx_init_cpu_dispatch();
