#pragma once
#include <unordered_map>
#include "sys_types.h"

// Key on shape AND the physical GPU virtual addresses to prevent stale bindings
struct DispatchCacheKey {
    uint64_t input_va;
    uint64_t weight_va;
    uint64_t output_va;
    uint32_t M, N, K, block_size;
    int mode; // 0: GEMM, 1: Q4_GEMV, 2: Q4_TILED
    
    bool operator==(const DispatchCacheKey& o) const {
        return input_va == o.input_va &&
               weight_va == o.weight_va &&
               output_va == o.output_va &&
               M == o.M && N == o.N && K == o.K && 
               block_size == o.block_size && mode == o.mode;
    }
};

struct KeyHash {
    size_t operator()(const DispatchCacheKey& k) const { 
        return std::hash<uint64_t>{}(k.input_va) ^ 
               std::hash<uint64_t>{}(k.weight_va) ^ 
               std::hash<uint64_t>{}(k.output_va) ^ 
               k.M ^ k.N ^ k.K ^ k.mode; 
    }
};

struct CachedCommandList {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd;
};

extern std::unordered_map<DispatchCacheKey, CachedCommandList, KeyHash> g_gpu_command_cache;

uint64_t gpu_dispatch_async(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val, ID3D12GraphicsCommandList* cl);