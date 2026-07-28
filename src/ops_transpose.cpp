#include "sys_types.h"
#include <vector>

void cpu_transpose(DpxGpuTensor& in, DpxGpuTensor& out, const std::vector<int>& perm) {
    if (!in.cpu_data || !out.cpu_data) return;

    int rank = in.shape.size();
    if (rank == 0) return;

    // C++ Shape Inference
    if (out.shape.empty()) {
        out.shape.resize(rank);
        for (int k = 0; k < rank; k++) {
            if (perm.size() > k && perm[k] < rank) {
                out.shape[k] = in.shape[perm[k]];
            } else {
                out.shape[k] = in.shape[k];
            }
        }
    }

    float* src = (float*)in.cpu_data;
    float* dst = (float*)out.cpu_data;

    std::vector<long> in_strides(rank);
    long acc = 1;
    for (int k = rank - 1; k >= 0; k--) {
        in_strides[k] = acc;
        acc *= in.shape[k];
    }

    long count = 1;
    for(int d : out.shape) count *= d;

    std::vector<int> idx(rank, 0);
    for (long lin = 0; lin < count; lin++) {
        long src_idx = 0;
        for (int k = 0; k < rank; k++) {
            if (perm.size() > k && perm[k] < rank) {
                src_idx += idx[k] * in_strides[perm[k]];
            }
        }
        dst[lin] = src[src_idx];
        
        for (int k = rank - 1; k >= 0; k--) {
            if (++idx[k] < out.shape[k]) break;
            idx[k] = 0;
        }
    }
}
