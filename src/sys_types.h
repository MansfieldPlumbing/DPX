#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <d3d12.h>
#include <wrl/client.h>

// MECHANISTIC FLAT TENSOR STRUCT
struct DpxGpuTensor {
    std::string name;
    std::vector<int> shape;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    uint64_t gpu_va;
    
    // Q4 Meta
    bool is_q4 = false;
    int block_size = 32;
    Microsoft::WRL::ComPtr<ID3D12Resource> scales_resource;
    Microsoft::WRL::ComPtr<ID3D12Resource> zp_resource;

    // Lifespan & Heap Tracking
    int first_use = -1;
    int last_use = -1;
    uint64_t required_size = 0;
    uint64_t heap_offset = 0;
    void* cpu_data = nullptr;
    void* cpu_scales = nullptr;
    void* cpu_zp = nullptr;
    // Dual-Buffer Producer/Consumer Blit Fields
    void* cpu_blit_buffer = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> blit_resource;
};

enum class DpxOp {
    MatMulQ4,
    Add,
    RMSNorm,
    AttentionGQA,
    Gather,
    ElementWise,
    Mul,
    Div,
    Sub,
    Silu,
    RoPE,
    Passthrough,
    Unknown
};

