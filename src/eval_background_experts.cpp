#include "sys_types.h"
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;

void eval_dispatch_background_moe_expert_async() {
    if (!g_device || !g_compute_queue) return;
    
    static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
    static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cl;
    static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    static uint64_t fence_val = 0;
    
    if (!alloc) {
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
        cl->Close();
        g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    }
    
    if (fence->GetCompletedValue() < fence_val) return;
    
    alloc->Reset();
    cl->Reset(alloc.Get(), nullptr);
    cl->Close();
    
    ID3D12CommandList* lists[] = { cl.Get() };
    g_compute_queue->ExecuteCommandLists(1, lists);
    
    fence_val++;
    g_compute_queue->Signal(fence.Get(), fence_val);
}
