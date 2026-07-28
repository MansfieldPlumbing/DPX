#include "sys_spsc_ring_buffer.h"
#include "policy_latency_dropper.h"
#include "sys_graph_orchestrator.h"
#include "eval_sentencepiece.h"
#include "sys_token_sampler.h"
#include "sys_token_streamer.h"
#include "sys_kv_cache_manager.h"
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
extern DpxTokenCallback g_ui_callback;

extern "C" __declspec(dllexport) void dpx_set_ui_callback(DpxTokenCallback cb) {
    g_ui_callback = cb;
}

extern SentencePieceFastUnigram g_tokenizer;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;
extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern bool g_engine_running;
extern bool g_dpx_cpu_only;
extern int g_dpx_ctx_size;

extern SysGraphOrchestrator g_embed_orchestrator;
extern SysGraphOrchestrator g_decoder_orchestrator;

static Microsoft::WRL::ComPtr<ID3D12Resource> g_shared_buffer;
static Microsoft::WRL::ComPtr<ID3D12Fence>    g_shared_buffer_fence;
static HANDLE                                  g_shared_buffer_handle = nullptr;
static HANDLE                                  g_shared_buffer_fence_handle = nullptr;
static HANDLE                                  g_shared_buffer_manifest_handle = nullptr;
static SharedBufferManifest*                   g_shared_buffer_manifest = nullptr;
static UINT64                                  g_shared_buffer_frame_val = 0;

void dpx_init_shared_buffer() {
    if (!g_device || g_dpx_cpu_only) return;

    UINT64 bufferSize = 256000 * sizeof(float);

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
    // Start playback right before the first token in this prompt set (frame 1)
    uint64_t last_consumed_id = freshest > (uint64_t)prompt_size ? (freshest - prompt_size - 1) : 0; 
    
    PolicyLatencyDropper policy(30); 
    int tokens_generated = 0;
    
    dpx_init_shared_buffer();

    Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cl;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE fence_evt = nullptr;
    uint64_t fence_val = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> pos_buffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> mask_buffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> num_logits_buffer;
    void* pos_mapped = nullptr;
    void* mask_mapped = nullptr;
    void* num_logits_mapped = nullptr;

    if (!g_dpx_cpu_only && g_device) {
        D3D12_HEAP_PROPERTIES rbProps = {}; rbProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rbDesc = {}; rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rbDesc.Width = 256000 * sizeof(float);
        rbDesc.Height = 1; rbDesc.DepthOrArraySize = 1; rbDesc.MipLevels = 1;
        rbDesc.SampleDesc.Count = 1; rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        g_device->CreateCommittedResource(&rbProps, D3D12_HEAP_FLAG_NONE, &rbDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buffer));
            
        g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
        cl->Close();

        g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        fence_evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        D3D12_HEAP_PROPERTIES upProps = { D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = 4096 * sizeof(int64_t);
        desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        g_device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&pos_buffer));
        pos_buffer->Map(0, nullptr, &pos_mapped);

        g_device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mask_buffer));
        mask_buffer->Map(0, nullptr, &mask_mapped);

        desc.Width = sizeof(int64_t);
        g_device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&num_logits_buffer));
        num_logits_buffer->Map(0, nullptr, &num_logits_mapped);

        int64_t one = 1;
        memcpy(num_logits_mapped, &one, sizeof(int64_t));
    }

    while (g_engine_running && tokens_generated < 256) {
        uint64_t freshest_id = ring.consumer_get_freshest();
        if (last_consumed_id >= freshest_id) {
            std::this_thread::sleep_for(std::chrono::microseconds(100)); 
            continue;
        }
                                                            
        // Process sequentially so the engine naturally builds state
        last_consumed_id++; 
        ActiveFrame& frame = ring.get_frame_memory(last_consumed_id);
                                         
        policy.begin_frame_clock();
        
        // 1. Dynamic Input Binding for the Embedder
        if (g_embed_orchestrator.input_tensor_index != -1 && g_embed_orchestrator.input_tensor_index < (int)g_embed_orchestrator.active_tensors.size()) {
            g_embed_orchestrator.active_tensors[g_embed_orchestrator.input_tensor_index].resource = frame.input_activations.resource;
            g_embed_orchestrator.active_tensors[g_embed_orchestrator.input_tensor_index].gpu_va = frame.input_activations.gpu_va;
            g_embed_orchestrator.active_tensors[g_embed_orchestrator.input_tensor_index].cpu_data = frame.mapped_data;
        }

        // 2. Evaluate Embedder
        g_embed_orchestrator.process_token_frame(frame.monotonic_id, policy);

        // 3. Bridge Embedder -> Decoder
        auto& embed_tensors = g_embed_orchestrator.active_tensors;
        auto& dec_tensors = g_decoder_orchestrator.active_tensors;
        auto& embed_map = g_embed_orchestrator.tensor_name_to_index;
        auto& dec_map = g_decoder_orchestrator.tensor_name_to_index;

        int embed_out = g_embed_orchestrator.output_tensor_index;
        int dec_in = g_decoder_orchestrator.input_tensor_index;
        if (embed_out != -1 && dec_in != -1 && embed_out < (int)embed_tensors.size() && dec_in < (int)dec_tensors.size()) {
            dec_tensors[dec_in].resource = embed_tensors[embed_out].resource;
            dec_tensors[dec_in].gpu_va = embed_tensors[embed_out].gpu_va;
            dec_tensors[dec_in].cpu_data = embed_tensors[embed_out].cpu_data;
        }

        if (embed_map.count("inputs_embeds") && dec_map.count("inputs_embeds")) {
            int src = embed_map["inputs_embeds"];
            int dst = dec_map["inputs_embeds"];
            dec_tensors[dst].resource = embed_tensors[src].resource;
            dec_tensors[dst].gpu_va = embed_tensors[src].gpu_va;
            dec_tensors[dst].cpu_data = embed_tensors[src].cpu_data;
        }
        if (embed_map.count("per_layer_inputs") && dec_map.count("per_layer_inputs")) {
            int src = embed_map["per_layer_inputs"];
            int dst = dec_map["per_layer_inputs"];
            dec_tensors[dst].resource = embed_tensors[src].resource;
            dec_tensors[dst].gpu_va = embed_tensors[src].gpu_va;
            dec_tensors[dst].cpu_data = embed_tensors[src].cpu_data;
        }

        // 4. Update and Bind Context
        int64_t current_pos = frame.monotonic_id - 1;
        std::vector<int64_t> mask(g_dpx_ctx_size, 0);
        for (int i = 0; i <= current_pos && i < g_dpx_ctx_size; ++i) {
            mask[i] = 1;
        }
        
        int64_t num_logits = 1;

        if (pos_mapped && mask_mapped) {
            memcpy(pos_mapped, &current_pos, sizeof(int64_t));
            memcpy(mask_mapped, mask.data(), mask.size() * sizeof(int64_t));
        }

        if (dec_map.count("position_ids")) {
            int idx = dec_map["position_ids"];
            if (pos_buffer) {
                dec_tensors[idx].resource = pos_buffer;
                dec_tensors[idx].gpu_va = pos_buffer->GetGPUVirtualAddress();
            }
            dec_tensors[idx].cpu_data = &current_pos;
        }
        if (dec_map.count("attention_mask")) {
            int idx = dec_map["attention_mask"];
            if (mask_buffer) {
                dec_tensors[idx].resource = mask_buffer;
                dec_tensors[idx].gpu_va = mask_buffer->GetGPUVirtualAddress();
            }
            dec_tensors[idx].cpu_data = mask.data();
        }
        if (dec_map.count("num_logits_to_keep")) {
            int idx = dec_map["num_logits_to_keep"];
            if (num_logits_buffer) {
                dec_tensors[idx].resource = num_logits_buffer;
                dec_tensors[idx].gpu_va = num_logits_buffer->GetGPUVirtualAddress();
            }
            dec_tensors[idx].cpu_data = &num_logits;
        }

        // --- INJECT KV CACHE BEFORE NODE EXECUTION ---
        for (auto& t : dec_tensors) {
            if (t.name.find("past") != std::string::npos) {
                auto it = g_kv_cache_manager.cache.find(t.name);
                if (it != g_kv_cache_manager.cache.end() && !it->second.data.empty() && t.cpu_data) {
                    size_t bytes = it->second.data.size() * sizeof(float);
                    memcpy(t.cpu_data, it->second.data.data(), bytes);
                }
            }
        }

        // 5. Evaluate Decoder
        auto t0 = std::chrono::high_resolution_clock::now();
        if (last_consumed_id >= freshest_id) std::cout << "[DEBUG DPX] Evaluating " << g_decoder_orchestrator.compute_sequence.size() << " decoder nodes on CPU..." << std::endl;
        g_decoder_orchestrator.process_token_frame(frame.monotonic_id, policy);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (last_consumed_id >= freshest_id) std::cout << "[DEBUG DPX] Decoder evaluation complete in " << ms << " ms!" << std::endl;
        
        // --- EXTRACT KV CACHE AFTER NODE EXECUTION ---
        for (auto& t : dec_tensors) {
            if (t.name.find("present") != std::string::npos) {
                if (t.cpu_data) {
                    std::vector<int> act_shape = t.shape;
                    for (auto& d : act_shape) {
                        if (d == -1) d = 4096; // Expand dynamic seq_len natively for cache store
                    }
                    g_kv_cache_manager.update_cache(t.name, (float*)t.cpu_data, act_shape);
                }
            }
        }

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
            
            cl->CopyBufferRegion(g_shared_buffer.Get(), 0, logits_tensor.resource.Get(), 0, 256000 * sizeof(float));
            cl->CopyBufferRegion(readback_buffer.Get(), 0, logits_tensor.resource.Get(), 0, 256000 * sizeof(float));
            
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
            
            void* pMappedData = nullptr;
            if (readback_buffer && SUCCEEDED(readback_buffer->Map(0, nullptr, &pMappedData)) && pMappedData) {
                float* logits = reinterpret_cast<float*>(pMappedData);
                uint32_t vocab_size = 32000;
                if (!logits_tensor.shape.empty() && logits_tensor.shape.back() > 0) {
                    vocab_size = static_cast<uint32_t>(logits_tensor.shape.back());
                }
                uint64_t total_elements = vocab_size;
                if (!logits_tensor.shape.empty()) {
                    total_elements = 1;
                    for (int d : logits_tensor.shape) { if(d>0) total_elements *= d; else if(d==-1) total_elements *= 1; }
                }
                uint64_t last_token_offset = (total_elements >= vocab_size) ? (total_elements - vocab_size) : 0;
                float* last_logits = logits + last_token_offset;

                float best_score = -9999.0f;
                for(uint32_t i = 0; i < vocab_size; i++) {
                    if(last_logits[i] > best_score) {
                        best_score = last_logits[i];
                        best_token = (int)i;
                    }
                }
                readback_buffer->Unmap(0, nullptr);
            }
        } else if (logits_tensor.cpu_data) {
            best_token = sys_sample_argmax(reinterpret_cast<float*>(logits_tensor.cpu_data), logits_tensor.shape);
        }

        // Only stream token and start autoregressing if the entire prompt buffer has been exhausted 
        // to establish KV cache (simulating one-shot prefill)
        if (last_consumed_id >= ring.consumer_get_freshest()) {
            sys_stream_token(best_token);
            
            // Stop if end of generation
            if (best_token == 1 || best_token == 107) break;

            uint64_t next_id = frame.monotonic_id + 1;
            float next_token_val = static_cast<float>(best_token);
            ring.producer_push(next_id, &next_token_val, sizeof(float));
            tokens_generated++;
        }
    }

    if (pos_buffer && pos_mapped) pos_buffer->Unmap(0, nullptr);
    if (mask_buffer && mask_mapped) mask_buffer->Unmap(0, nullptr);
    if (num_logits_buffer && num_logits_mapped) num_logits_buffer->Unmap(0, nullptr);

    dpx_shutdown_shared_buffer();
}