#include "sys_types.h"
#include <execution>
#include <numeric>
#include <algorithm>
#include <vector>

inline int nib4(const uint8_t* buf, uint32_t rowOff, uint32_t idx) {
    return (buf[rowOff + (idx >> 1)] >> ((idx & 1) << 2)) & 0xF;
}

void cpu_matmul_nbits_simd(
    const float* a, const float* scsp, const uint8_t* bSpan, const uint8_t* zpSpan, float* c_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t block_size
) {
    uint32_t nBlk = K / block_size;
    uint32_t rowBytes = nBlk * (block_size * 4 / 8);
    uint32_t zpRowBytes = (nBlk * 4 + 7) / 8;

    std::vector<uint32_t> n_indices(N);
    std::iota(n_indices.begin(), n_indices.end(), 0);

    std::for_each(std::execution::par_unseq, n_indices.begin(), n_indices.end(), [&](uint32_t nn) {
        uint32_t rb = nn * rowBytes, zb = nn * zpRowBytes;
        for (uint32_t m = 0; m < M; ++m) {
            uint32_t ao = m * K;
            float acc = 0.0f;
            for (uint32_t b = 0; b < nBlk; ++b) {
                float s = scsp[nn * nBlk + b];
                int zp = zpSpan ? nib4(zpSpan, zb, b) : 8;
                float aq = 0.0f, asum = 0.0f;
                uint32_t k0 = b * block_size;
                for (uint32_t i = 0; i < block_size; ++i) {
                    uint32_t k = k0 + i;
                    int q = (bSpan[rb + (k >> 1)] >> ((k & 1) << 2)) & 0xF;
                    float av = a[ao + k];
                    aq += av * q;
                    asum += av;
                }
                acc += s * (aq - zp * asum);
            }
            c_out[m * N + nn] = acc;
        }
    });
}

void cpu_matmul_fp32(
    const float* A, const float* B, float* C,
    uint32_t M, uint32_t N, uint32_t K
) {
    std::vector<uint32_t> m_indices(M);
    std::iota(m_indices.begin(), m_indices.end(), 0);

    std::for_each(std::execution::par_unseq, m_indices.begin(), m_indices.end(), [&](uint32_t m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    });
}
