#include "sys_types.h"
#include "sys_graph_orchestrator.h"
#include "sys_spsc_ring_buffer.h"
#include "policy_latency_dropper.h"
#include "gpu_command_cache.h"
#include <iostream>
#include <cmath>

#define NOMINMAX
#include <windows.h>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;

void eval_dispatch_gemm_q4_async(DpxGpuTensor& a, DpxGpuTensor& w, DpxGpuTensor& c, uint32_t M, uint32_t N, uint32_t K);
void gpu_issue_residual_bypass_copy(ID3D12GraphicsCommandList* cl, DpxGpuTensor& in_tensor, DpxGpuTensor& out_tensor);

// Safe lazy allocator to prevent segfaults on uninitialized intermediate tensors
void dpx_ensure_resource(DpxGpuTensor& t) {
    if (!t.resource && g_device) {
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 32 * 1024 * 1024; // 32MB dummy alloc for safety during dev
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&t.resource));
        t.gpu_va = t.resource->GetGPUVirtualAddress();
    }
}

void* MapTensor(DpxGpuTensor& t) {
    if(!t.resource) return nullptr;
    void* p; D3D12_RANGE r = {0, 0};
    t.resource->Map(0, &r, &p);
    return p;
}

void UnmapTensor(DpxGpuTensor& t) {
    if(!t.resource) return;
    D3D12_RANGE w = {0, 0};
    t.resource->Unmap(0, &w);
}

ID3D12GraphicsCommandList* get_main_compute_command_list() { 
    static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cl;
    if (!cl && g_device) {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
        cl->Close();
    }
    return cl.Get(); 
} 

void SysGraphOrchestrator::process_token_frame(uint64_t sequence_id, PolicyLatencyDropper& current_dropper) {
    memory_arena.reset_frame_allocations();
        
    for (const auto& node : compute_sequence) {
        // Prevent Segfault: Give intermediate outputs memory before execution
        for (int out_idx : node.output_registry_indices) {
            dpx_ensure_resource(active_tensors[out_idx]);
        }

        if (current_dropper.evaluate_drop_exceeds_budget()) {
            if (node.op_code == DpxOp::Add) {
                auto* cmd_list = get_main_compute_command_list(); 
                if (cmd_list) {
                    static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
                    if(!alloc) g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
                    alloc->Reset();
                    cmd_list->Reset(alloc.Get(), nullptr);

                    gpu_issue_residual_bypass_copy(cmd_list, 
                        active_tensors[node.input_registry_indices[0]], 
                        active_tensors[node.output_registry_indices[0]]
                    );

                    static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
                    static uint64_t fence_val = 0;
                    static HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                    if (!fence) g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
                    
                    gpu_dispatch_async(g_compute_queue.Get(), fence.Get(), fence_val, cmd_list);
                    if (fence->GetCompletedValue() < fence_val) {
                        fence->SetEventOnCompletion(fence_val, evt);
                        WaitForSingleObject(evt, INFINITE);
                    }
                }
                continue; 
            }
        }
                                                                                                                        
        switch (node.op_code) {
            case DpxOp::MatMulQ4:
                if (active_tensors[node.input_registry_indices[0]].resource && active_tensors[node.input_registry_indices[1]].resource) {
                    eval_dispatch_gemm_q4_async(
                        active_tensors[node.input_registry_indices[0]], 
                        active_tensors[node.input_registry_indices[1]], 
                        active_tensors[node.output_registry_indices[0]], 
                        1, 4096, 4096
                    );
                }
                break;
            default:
                break;
        }
    }
}
