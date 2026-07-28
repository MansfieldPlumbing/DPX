#include "sys_kernel_dispatch.h"
#include "sys_precision_config.h"
#include "sys_cpu_detect.h"
#include <iostream>

DpxPrecisionMode g_dpx_precision = DpxPrecisionMode::FP32;

extern void cpu_matmul_nbits_simd(const float* a, const float* scales, const uint8_t* b, const uint8_t* zp, float* c_out, uint32_t M, uint32_t N, uint32_t K, uint32_t block_size);
extern void cpu_matmul_avx512_bw(const float* a, const float* scales, const uint8_t* b, const uint8_t* zp, float* c_out, uint32_t M, uint32_t N, uint32_t K, uint32_t block_size);
extern void cpu_gather_block_quantized(const uint8_t* data, const int32_t* indices, uint32_t num_indices, const float* scales, const uint8_t* zp, float* out, uint32_t num_embeds, uint32_t embed_dim, uint32_t block_size);

MatMulKernelFunc g_matmul_kernel = cpu_matmul_nbits_simd;
GatherQuantizedFunc g_gather_kernel = cpu_gather_block_quantized;

void dpx_init_cpu_dispatch() {
    auto cpu = dpx_detect_cpu();
    if (cpu.avx512bw) {
        g_matmul_kernel = cpu_matmul_avx512_bw;
        std::cout << "\033[1;32m[+] Dispatched AVX-512 BW 512-bit SIMD Kernel (Dual FMA / vpshufb)\033[0m\n";
    } else if (cpu.avx2) {
        g_matmul_kernel = cpu_matmul_nbits_simd;
        std::cout << "\033[1;33m[+] Dispatched AVX2 256-bit SIMD Kernel\033[0m\n";
    } else {
        g_matmul_kernel = cpu_matmul_nbits_simd;
        std::cout << "\033[1;33m[+] Dispatched Standard SIMD Fallback Kernel\033[0m\n";
    }
    g_gather_kernel = cpu_gather_block_quantized;
}
