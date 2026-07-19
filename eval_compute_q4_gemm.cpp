#include "sys_types.h"
#include "gpu_command_cache.h"
#include <stdexcept>

std::unordered_map<DispatchCacheKey, CachedCommandList, KeyHash> g_gpu_command_cache;


extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;
extern Microsoft::WRL::ComPtr<ID3D12RootSignature> g_root;
extern Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pso_gemv;
extern Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pso_tiled;

uint64_t gpu_dispatch_async(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val, ID3D12GraphicsCommandList* cl);

void eval_dispatch_gemm_q4_async(
    DpxGpuTensor& a_input,
    DpxGpuTensor& q4_weight,
    DpxGpuTensor& c_output,
    uint32_t M, uint32_t N, uint32_t K
) {
    int mode = (M == 1) ? 1 : 2; // GEMV vs Tiled
    DispatchCacheKey key = { M, N, K, static_cast<uint32_t>(q4_weight.block_size), mode };
    
    // O(1) Cache fetch: no CPU allocations, no strings
    auto it = g_gpu_command_cache.find(key);
    ID3D12GraphicsCommandList* cl = nullptr;

    if (it == g_gpu_command_cache.end()) {
        CachedCommandList cached;
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&cached.alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cached.alloc.Get(), nullptr, IID_PPV_ARGS(&cached.cmd));

        cl = cached.cmd.Get();
        cl->SetPipelineState(mode == 1 ? g_pso_gemv.Get() : g_pso_tiled.Get());
        cl->SetComputeRootSignature(g_root.Get());

        // 1. Pass Root Constants
        UINT consts[5] = { M, N, K, static_cast<UINT>(q4_weight.block_size), 1u };
        cl->SetComputeRoot32BitConstants(0, 5, consts, 0);

        // 2. Bind SRVs (Input A, Packed Bq, Block Scales, Zero Points)
        cl->SetComputeRootShaderResourceView(1, a_input.gpu_va);
        cl->SetComputeRootShaderResourceView(2, q4_weight.gpu_va);
        cl->SetComputeRootShaderResourceView(3, q4_weight.scales_resource->GetGPUVirtualAddress());
        cl->SetComputeRootShaderResourceView(4, q4_weight.zp_resource->GetGPUVirtualAddress());

        // 3. Bind UAV (Output C)
        cl->SetComputeRootUnorderedAccessView(5, c_output.gpu_va);

        // 4. Threadgroup Dispatch Geometry matching shader thread signatures
        if (mode == 1) {
            cl->Dispatch((N + 7) / 8, 1, 1); // gemm_q4_gemv numthreads(16, 8, 1) -> 8 outputs
        } else {
            cl->Dispatch((N + 15) / 16, (M + 15) / 16, 1); // gemm_q4_tiled numthreads(16, 16, 1)
        }

        // 5. Native UAV Barrier to sync in-place activations
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = c_output.resource.Get();
        cl->ResourceBarrier(1, &barrier);

        // Track compiled graph sequence
        g_gpu_command_cache[key] = cached;
    } else {
        cl = it->second.cmd.Get();
    }

    // Submit async without locking the CPU thread
    static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    static uint64_t fence_val = 0;
    if (!fence) g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    gpu_dispatch_async(g_compute_queue.Get(), fence.Get(), fence_val, cl);
}
