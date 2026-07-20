#include "sys_spsc_ring_buffer.h"
#include "policy_latency_dropper.h"
#include "sys_graph_orchestrator.h"
#include "eval_sentencepiece.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <sddl.h>

#pragma comment(lib, "advapi32.lib")

struct SharedBufferManifest {
    UINT64 frameValue;
    UINT64 bufferSize;
    LUID adapterLuid;
    UINT textureWidth;
    UINT textureHeight;
    DXGI_FORMAT textureFormat;
    WCHAR resourceName[256];
    WCHAR fenceName[256];
};

typedef void (*DpxTokenCallback)(const char*);
extern DpxTokenCallback g_ui_callback; // Reference the storage defined in sys_engine_core.cpp

extern "C" __declspec(dllexport) void dpx_set_ui_callback(DpxTokenCallback cb) {
    g_ui_callback = cb;
}

extern SentencePieceFastUnigram g_tokenizer;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;
extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern bool g_engine_running;
extern bool g_dpx_cpu_only;

extern SysGraphOrchestrator g_embed_orchestrator;
extern SysGraphOrchestrator g_decoder_orchestrator;

// --- DirectPort Shared Buffer State ---
static Microsoft::WRL::ComPtr<ID3D12Resource> g_shared_buffer;
static Microsoft::WRL::ComPtr<ID3D12Fence>    g_shared_buffer_fence;
static HANDLE                                  g_shared_buffer_handle = nullptr;
static HANDLE                                  g_shared_buffer_fence_handle = nullptr;
static HANDLE                                  g_shared_buffer_manifest_handle = nullptr;
static SharedBufferManifest*                   g_shared_buffer_manifest = nullptr;
static UINT64                                  g_shared_buffer_frame_val = 0;

void dpx_init_shared_buffer() {
    if (!g_device || g_dpx_cpu_only) return;

    UINT64 bufferSize = 32000 * sizeof(float); // Sharing the full 32K logits

    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Width = bufferSize;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_shared_buffer));
    g_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_shared_buffer_fence));

    PSECURITY_DESCRIPTOR sd = nullptr;
    ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;AU)", SDDL_REVISION_1, &sd, NULL);
    SECURITY_ATTRIBUTES sa = { sizeof(sa), sd, FALSE };

    DWORD pid = GetCurrentProcessId();
    std::wstring resourceName = L"Global\\DirectPort_Buffer_" + std::to_wstring(pid);
    std::wstring fenceName = L"Global\\DirectPort_BufferFence_" + std::to_wstring(pid);
    std::wstring manifestName = L"Global\\DirectPort_T2B_Producer_Manifest_" + std::to_wstring(pid);

    g_device->CreateSharedHandle(g_shared_buffer.Get(), &sa, GENERIC_ALL, resourceName.c_str(), &g_shared_buffer_handle);
    g_device->CreateSharedHandle(g_shared_buffer_fence.Get(), &sa, GENERIC_ALL, fenceName.c_str(), &g_shared_buffer_fence_handle);

    g_shared_buffer_manifest_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, sizeof(SharedBufferManifest), manifestName.c_str());
    LocalFree(sd);

    if (g_shared_buffer_manifest_handle) {
        g_shared_buffer_manifest = (SharedBufferManifest*)MapViewOfFile(g_shared_buffer_manifest_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedBufferManifest));
        if (g_shared_buffer_manifest) {
            ZeroMemory(g_shared_buffer_manifest, sizeof(SharedBufferManifest));
            g_shared_buffer_manifest->bufferSize = bufferSize;
            g_shared_buffer_manifest->adapterLuid = g_device->GetAdapterLuid();
            g_shared_buffer_manifest->textureWidth = 32000;
            g_shared_buffer_manifest->textureHeight = 1;
            g_shared_buffer_manifest->textureFormat = DXGI_FORMAT_R32_FLOAT;
            wcscpy_s(g_shared_buffer_manifest->resourceName, resourceName.c_str());
            wcscpy_s(g_shared_buffer_manifest->fenceName, fenceName.c_str());
        }
    }
}

void dpx_shutdown_shared_buffer() {
    if (g_shared_buffer_manifest) UnmapViewOfFile(g_shared_buffer_manifest);
    if (g_shared_buffer_manifest_handle) CloseHandle(g_shared_buffer_manifest_handle);
    if (g_shared_buffer_handle) CloseHandle(g_shared_buffer_handle);
    if (g_shared_buffer_fence_handle) CloseHandle(g_shared_buffer_fence_handle);
    g_shared_buffer.Reset();
    g_shared_buffer_fence.Reset();
}

void sys_run_consumer_loop(SysSPSCRingBuffer& ring, int prompt_size) {
    uint64_t freshest = ring.consumer_get_freshest();
    uint64_t last_consumed_id = freshest >= (uint64_t)prompt_size ? freshest - prompt_size : 0; 
    
    PolicyLatencyDropper policy(30); 
    int tokens_generated = 0;
    
    dpx_init_shared_buffer();

    Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cl;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE fence_evt = nullptr;
    uint64_t fence_val = 0;

    if (!g_dpx_cpu_only && g_device) {
        D3D12_HEAP_PROPERTIES rbProps = {}; rbProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rbDesc = {}; rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rbDesc.Width = 32000 * sizeof(float);
        rbDesc.Height = 1; rbDesc.DepthOrArraySize = 1; rbDesc.MipLevels = 1;
        rbDesc.SampleDesc.Count = 1; rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        g_device->CreateCommittedResource(&rbProps, D3D12_HEAP_FLAG_NONE, &rbDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buffer));
            
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
        cl->Close();

        g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        fence_evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

    while (g_engine_running && tokens_generated < 256) {
        uint64_t freshest_id = ring.consumer_get_freshest();
        if (freshest_id <= last_consumed_id) {
            std::this_thread::sleep_for(std::chrono::microseconds(100)); 
            continue;
        }
                                                            
        last_consumed_id = freshest_id;
        ActiveFrame& frame = ring.get_frame_memory(freshest_id);
                                        
        policy.begin_frame_clock();
        
        // Dynamic Input Binding directly from the push buffer!
        if (g_decoder_orchestrator.input_tensor_index != -1 && g_decoder_orchestrator.input_tensor_index < (int)g_decoder_orchestrator.active_tensors.size()) {
            g_decoder_orchestrator.active_tensors[g_decoder_orchestrator.input_tensor_index].resource = frame.input_activations.resource;
            g_decoder_orchestrator.active_tensors[g_decoder_orchestrator.input_tensor_index].gpu_va = frame.input_activations.gpu_va;
        }

        g_embed_orchestrator.process_token_frame(frame.monotonic_id, policy);
        g_decoder_orchestrator.process_token_frame(frame.monotonic_id, policy);
        
        // Find the designated output logit tensor safely
        int out_idx = g_decoder_orchestrator.output_tensor_index != -1 ? 
                      g_decoder_orchestrator.output_tensor_index : 
                      (int)g_decoder_orchestrator.active_tensors.size() - 1;

        if (out_idx < 0 || out_idx >= (int)g_decoder_orchestrator.active_tensors.size()) continue;
        auto& logits_tensor = g_decoder_orchestrator.active_tensors[out_idx];
        
        int best_token = 0;
        
        if (!g_dpx_cpu_only && g_device && logits_tensor.resource) {
            alloc->Reset();
            cl->Reset(alloc.Get(), nullptr);
            
            D3D12_RESOURCE_BARRIER barriers[2] = {};
            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Transition.pResource = logits_tensor.resource.Get();
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barriers[0].Transition.Subresource = 0;
            
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Transition.pResource = g_shared_buffer.Get();
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barriers[1].Transition.Subresource = 0;
            
            cl->ResourceBarrier(2, barriers);
            
            cl->CopyBufferRegion(g_shared_buffer.Get(), 0, logits_tensor.resource.Get(), 0, 32000 * sizeof(float));
            cl->CopyBufferRegion(readback_buffer.Get(), 0, logits_tensor.resource.Get(), 0, 32000 * sizeof(float));
            
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
            cl->ResourceBarrier(2, barriers);
            cl->Close();
            
            ID3D12CommandList* lists[] = { cl.Get() };
            g_compute_queue->ExecuteCommandLists(1, lists);
            
            fence_val++;
            g_compute_queue->Signal(fence.Get(), fence_val);
            
            g_shared_buffer_frame_val++;
            g_compute_queue->Signal(g_shared_buffer_fence.Get(), g_shared_buffer_frame_val);
            
            if (g_shared_buffer_manifest) {
                InterlockedExchange64(reinterpret_cast<volatile LONGLONG*>(&g_shared_buffer_manifest->frameValue), g_shared_buffer_frame_val);
                WakeByAddressAll(&g_shared_buffer_manifest->frameValue);
            }

            if (fence->GetCompletedValue() < fence_val) {
                fence->SetEventOnCompletion(fence_val, fence_evt);
                WaitForSingleObject(fence_evt, INFINITE);
            }
            
            void* pMappedData;
            readback_buffer->Map(0, nullptr, &pMappedData);
            float* logits = reinterpret_cast<float*>(pMappedData);
            
            float best_score = -9999.0f;
            for(int i = 0; i < 32000; i++) {
                if(logits[i] > best_score) {
                    best_score = logits[i];
                    best_token = i;
                }
            }
            readback_buffer->Unmap(0, nullptr);
        } else {
            std::cerr << "\n[!] CPU inference not fully wired to Orchestrator. Halting loop.\n"; break; 
        }

        std::string output_text = g_tokenizer.detokenize({best_token});
        std::cout << output_text << std::flush;
        
        if (g_ui_callback) g_ui_callback(output_text.c_str());
        
        if (best_token == 1 || best_token == 2) break;

        uint64_t next_id = frame.monotonic_id + 1;
        if (next_id > ring.consumer_get_freshest()) {
            float next_token_val = static_cast<float>(best_token);
            ring.producer_push(next_id, &next_token_val, sizeof(float));
            tokens_generated++;
        }
    }

    dpx_shutdown_shared_buffer();
}

