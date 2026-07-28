#include "sys_types.h"
#include "gpu_command_cache.h"
#include <stdexcept>
#include <iostream>

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
    if (!g_device) {
        std::cerr << ">>> [ERROR] g_device is NULL in eval_dispatch_gemm_q4_async!" << std::endl;
        return;
    }

    ID3D12PipelineState* pso = (M == 1) ? g_pso_gemv.Get() : g_pso_tiled.Get();
    if (!pso) {
        static bool warned = false;
        if (!warned) {
            std::cerr << ">>> [WARNING] PSO is NULL for mode " << (M == 1 ? "GEMV" : "Tiled") << "! Bypassing." << std::endl;
            warned = true;
        }
        return;
    }

    int mode = (M == 1) ? 1 : 2; // GEMV vs Tiled
    
    static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> s_cmd_alloc;
    static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> s_cmd_list;

    if (!s_cmd_alloc) {
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&s_cmd_alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, s_cmd_alloc.Get(), pso, IID_PPV_ARGS(&s_cmd_list));
        s_cmd_list->Close();
    }

    if (!s_cmd_alloc || !s_cmd_list) return;

    s_cmd_alloc->Reset();
    s_cmd_list->Reset(s_cmd_alloc.Get(), pso);
    ID3D12GraphicsCommandList* cl = s_cmd_list.Get();

    cl->SetComputeRootSignature(g_root.Get());

    // 1. Pass Root Constants
    UINT consts[5] = { M, N, K, static_cast<UINT>(q4_weight.block_size), q4_weight.zp_resource ? 1u : 0u };
    cl->SetComputeRoot32BitConstants(0, 5, consts, 0);

    // 2. Bind SRVs safely
    D3D12_GPU_VIRTUAL_ADDRESS scales_va = q4_weight.scales_resource ? q4_weight.scales_resource->GetGPUVirtualAddress() : 0;
    D3D12_GPU_VIRTUAL_ADDRESS zp_va = q4_weight.zp_resource ? q4_weight.zp_resource->GetGPUVirtualAddress() : 0;

    cl->SetComputeRootShaderResourceView(1, a_input.gpu_va);
    cl->SetComputeRootShaderResourceView(2, q4_weight.gpu_va);
    cl->SetComputeRootShaderResourceView(3, scales_va);
    cl->SetComputeRootShaderResourceView(4, zp_va);

    // 3. Bind UAV (Output C)
    cl->SetComputeRootUnorderedAccessView(5, c_output.gpu_va);

    // 4. Threadgroup Dispatch Geometry
    if (mode == 1) {
        cl->Dispatch((N + 7) / 8, 1, 1);
    } else {
        cl->Dispatch((N + 15) / 16, (M + 15) / 16, 1);
    }

    // 5. UAV Barrier to sync in-place activations safely
    if (c_output.resource) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = c_output.resource.Get();
        cl->ResourceBarrier(1, &barrier);
    }

    static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    static uint64_t fence_val = 0;
    if (!fence) g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    gpu_dispatch_async(g_compute_queue.Get(), fence.Get(), fence_val, cl);
}