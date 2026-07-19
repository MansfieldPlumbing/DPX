#include "sys_types.h"
#include <algorithm> // For std::swap

uint64_t gpu_dispatch_async(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val, ID3D12GraphicsCommandList* cl) {
    cl->Close();
        ID3D12CommandList* lists[] = { cl };
            queue->ExecuteCommandLists(1, lists);
                fence_val++;
                    queue->Signal(fence, fence_val);
                        return fence_val; // Never block
                        }
                        
bool gpu_is_fence_complete(ID3D12Fence* fence, uint64_t target_val) {
    if (!fence) return true;
        return fence->GetCompletedValue() >= target_val;
        }
        
// Residual Alias Bypass: Native memory routing dropping compute completely.
void gpu_issue_residual_bypass_copy(ID3D12GraphicsCommandList* cl, DpxGpuTensor& in_tensor, DpxGpuTensor& out_tensor) {
    if (in_tensor.resource.Get() != out_tensor.resource.Get()) {
            D3D12_RESOURCE_BARRIER barriers[2];
                    
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                        barriers[0].Transition.pResource = in_tensor.resource.Get();
                                barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                                        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                                                barriers[0].Transition.Subresource = 0;
                                                
        barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                        barriers[1].Transition.pResource = out_tensor.resource.Get();
                                barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                                        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                                                barriers[1].Transition.Subresource = 0;
                                                
        cl->ResourceBarrier(2, barriers);
                cl->CopyResource(out_tensor.resource.Get(), in_tensor.resource.Get());
                
        // Transition back cleanly
                std::swap(barriers[0].Transition.StateBefore, barriers[0].Transition.StateAfter);
                        std::swap(barriers[1].Transition.StateBefore, barriers[1].Transition.StateAfter);
                                cl->ResourceBarrier(2, barriers);
                                    }
                                    }
                                    