#pragma once
#include <vector>
#include <cstdint>
#include <d3d12.h>
#include <wrl/client.h>

// MECHANISTIC FLAT TENSOR STRUCT
struct DpxGpuTensor {
    std::vector<int> shape;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    uint64_t gpu_va;
    
    // Q4 Meta
    bool is_q4 = false;
    int block_size = 32;
    Microsoft::WRL::ComPtr<ID3D12Resource> scales_resource;
    Microsoft::WRL::ComPtr<ID3D12Resource> zp_resource;
};
