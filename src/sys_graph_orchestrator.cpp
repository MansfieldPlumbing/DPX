#include "sys_types.h"
#include "sys_graph_orchestrator.h"
#include "sys_kernel_dispatch.h"
#include "sys_spsc_ring_buffer.h"
#include "policy_latency_dropper.h"
#include "gpu_command_cache.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#define NOMINMAX
#include <windows.h>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;

void eval_dispatch_gemm_q4_async(DpxGpuTensor& a, DpxGpuTensor& w, DpxGpuTensor& c, uint32_t M, uint32_t N, uint32_t K);
void gpu_issue_residual_bypass_copy(ID3D12GraphicsCommandList* cl, DpxGpuTensor& in_tensor, DpxGpuTensor& out_tensor);

// Safe lazy allocator to prevent segfaults on uninitialized intermediate tensors
void dpx_ensure_resource(DpxGpuTensor& t, SysMemoryArena& arena) {
    if (!t.cpu_data) {
        size_t alloc_bytes = 256000 * 36 * sizeof(float);
        if (!t.shape.empty()) {
            size_t elems = 1;
            for (int d : t.shape) elems *= d;
            alloc_bytes = elems * sizeof(float);
        }
        if (alloc_bytes < 1024 * 1024) alloc_bytes = 1024 * 1024;
        t.cpu_data = _aligned_malloc(alloc_bytes, 64);
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
    return cl.Get(); 
}

static bool execute_single_node_seh(
    const NodeTopology& node, 
    std::vector<DpxGpuTensor>& active_tensors, 
    int in0, 
    int out0, 
    size_t ni, 
    size_t total_nodes
) {
    auto& in_a_tensor = active_tensors[in0];
    auto& out = active_tensors[out0];

    EXCEPTION_POINTERS* s_seh_ex_ptrs = nullptr;
    __try {
        if (in_a_tensor.cpu_blit_buffer && in_a_tensor.cpu_data && in_a_tensor.cpu_blit_buffer != in_a_tensor.cpu_data) {
            uint64_t blit_size = 1;
            if (!in_a_tensor.shape.empty()) { for (int d : in_a_tensor.shape) blit_size *= d; } else { blit_size = 2048; }
            memcpy(in_a_tensor.cpu_data, in_a_tensor.cpu_blit_buffer, blit_size * sizeof(float));
        }

        extern void cpu_matmul_nbits_simd(const float* a, const float* scsp, const uint8_t* bSpan, const uint8_t* zpSpan, float* c_out, uint32_t M, uint32_t N, uint32_t K, uint32_t block_size);
        extern void cpu_rms_norm(const float* X, const float* weight, float* out, uint32_t outer, uint32_t inner, float eps, uint32_t weight_len);
        extern void cpu_attention_gqa(const float* Q, const float* K, const float* V, float* out, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t seq_len);

        switch (node.op_code) {
            case DpxOp::MatMulQ4: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& in_a = active_tensors[in0];
                        auto& in_w = active_tensors[in1];

                        void* scales_ptr = in_w.cpu_scales;
                        void* zp_ptr = in_w.cpu_zp;

                        // Input[2] is the scale tensor (exact name registered in index map)
                        // It may be in cpu_scales (decoded FP16) OR cpu_data (FP32 scales)
                        if (!scales_ptr && node.input_registry_indices.size() >= 3) {
                            int in2 = node.input_registry_indices[2];
                            if (in2 >= 0 && in2 < (int)active_tensors.size()) {
                                scales_ptr = active_tensors[in2].cpu_scales ? active_tensors[in2].cpu_scales
                                                                            : active_tensors[in2].cpu_data;
                            }
                        }
                        if (!zp_ptr && node.input_registry_indices.size() >= 4) {
                            int in3 = node.input_registry_indices[3];
                            if (in3 >= 0 && in3 < (int)active_tensors.size()) {
                                zp_ptr = active_tensors[in3].cpu_zp ? active_tensors[in3].cpu_zp
                                                                    : active_tensors[in3].cpu_data;
                            }
                        }

                        if(in_a.cpu_data && in_w.cpu_data && out.cpu_data && scales_ptr) {
                            uint32_t N = in_w.shape.size() > 0 ? in_w.shape[0] : 32000;
                            uint32_t K = in_w.shape.size() > 1 ? in_w.shape[1] * 32 : 2048;
                            g_matmul_kernel((float*)in_a.cpu_data, (float*)scales_ptr, (uint8_t*)in_w.cpu_data, (uint8_t*)zp_ptr, (float*)out.cpu_data, 1, N, K, 32);
                        } else if (in_a.cpu_data && in_w.cpu_data && out.cpu_data && !in_w.is_q4) {
                            // Unquantized FP32 MatMul (e.g. Q * K^T or Softmax * V in attention)
                            extern void cpu_matmul_fp32(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K);
                            uint32_t M = 1;
                            uint32_t N = in_w.shape.size() > 1 ? in_w.shape[1] : (in_w.shape.size() > 0 ? in_w.shape[0] : 2048);
                            uint32_t K = in_w.shape.size() > 0 ? in_w.shape[0] : 2048;
                            cpu_matmul_fp32((float*)in_a.cpu_data, (float*)in_w.cpu_data, (float*)out.cpu_data, M, N, K);
                        } else if (in_a.cpu_data && out.cpu_data) {
                            static int num_warns = 0;
                            if (num_warns++ < 5) {
                                printf("[MM DIAG] Node #%d MatMul missing: inputs_size=%d in0.data=%p in1.data=%p in1.scales=%p out.data=%p\n",
                                    (int)ni, (int)node.input_registry_indices.size(), in_a.cpu_data, in_w.cpu_data, in_w.cpu_scales, out.cpu_data);
                                for (size_t k = 0; k < node.input_registry_indices.size(); k++) {
                                    int idx = node.input_registry_indices[k];
                                    if (idx >= 0 && idx < (int)active_tensors.size()) {
                                        printf("   input[%d] = %d (%s) cpu_data=%p cpu_scales=%p cpu_zp=%p\n",
                                            (int)k, idx, active_tensors[idx].name.c_str(),
                                            active_tensors[idx].cpu_data, active_tensors[idx].cpu_scales, active_tensors[idx].cpu_zp);
                                    }
                                }
                            }
                            // No scales available — passthrough to avoid NaN propagation
                            uint64_t sz = 1;
                            if (!out.shape.empty()) { for (int d : out.shape) sz *= d; } else { sz = 2048; }
                            memset(out.cpu_data, 0, sz * sizeof(float));
                        }
                    }
                }
                break;
            }
            case DpxOp::RMSNorm: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& in_x = active_tensors[in0];
                        auto& in_w = active_tensors[in1];
                        if(in_x.cpu_data && out.cpu_data) {
                            uint32_t inner = in_x.shape.size() > 0 ? (in_x.shape.back() > 0 ? in_x.shape.back() : 2048) : 2048;
                            uint32_t weight_len = 1;
                            if (!in_w.shape.empty()) {
                                for (int d : in_w.shape) { if (d > 0) weight_len *= d; }
                            } else {
                                weight_len = inner;
                            }
                            if (in_w.cpu_data) {
                                cpu_rms_norm((float*)in_x.cpu_data, (float*)in_w.cpu_data, (float*)out.cpu_data, 1, inner, 1e-5f, weight_len);
                            } else {
                                // Weight not loaded — identity copy to keep activations flowing
                                memcpy(out.cpu_data, in_x.cpu_data, inner * sizeof(float));
                            }
                        }
                    }
                }
                break;
            }
            case DpxOp::Add: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& a = active_tensors[in0];
                        auto& b = active_tensors[in1];
                        if (a.cpu_data && b.cpu_data && out.cpu_data) {
                            float* af = (float*)a.cpu_data; float* bf = (float*)b.cpu_data; float* cf = (float*)out.cpu_data;
                            uint32_t size = a.shape.size() > 0 ? a.shape.back() : 2048;
                            for(uint32_t i=0; i<size; i++) cf[i] = af[i] + bf[i];
                        }
                    }
                }
                break;
            }
            case DpxOp::Gather: {
                if (node.input_registry_indices.size() >= 2) {
                    int in0_idx = node.input_registry_indices[0];
                    int in1_idx = node.input_registry_indices[1];
                    if (in0_idx >= 0 && in0_idx < (int)active_tensors.size() && in1_idx >= 0 && in1_idx < (int)active_tensors.size()) {
                        auto& in_weights = active_tensors[in0_idx];
                        auto& in_indices = active_tensors[in1_idx];
                        if (in_weights.cpu_data && out.cpu_data) {
                            int token_id = 0;
                            if (in_indices.cpu_data) {
                                int64_t int64_val = *(int64_t*)in_indices.cpu_data;
                                int32_t raw_int = *(int32_t*)in_indices.cpu_data;
                                float float_val = *(float*)in_indices.cpu_data;
                                if (int64_val >= 0 && int64_val < 256000) token_id = static_cast<int>(int64_val);
                                else if (raw_int >= 0 && raw_int < 256000) token_id = raw_int;
                                else if (!std::isnan(float_val) && float_val >= 0.0f && float_val < 256000.0f) token_id = static_cast<int>(float_val);
                            }
                            uint32_t embed_dim = in_weights.shape.size() > 1 ? in_weights.shape[1] : 2048;
                            uint32_t num_embeds = in_weights.shape.size() > 0 ? in_weights.shape[0] : 256000;
                            if (in_weights.is_q4) {
                                int32_t idx_val = token_id;
                                g_gather_kernel((uint8_t*)in_weights.cpu_data, &idx_val, 1, (float*)in_weights.cpu_scales, (uint8_t*)in_weights.cpu_zp, (float*)out.cpu_data, num_embeds, embed_dim, 32);
                            } else {
                                if (token_id < 0 || token_id >= (int)num_embeds) token_id = 0;
                                float* weight_ptr = (float*)in_weights.cpu_data + (uint64_t)token_id * embed_dim;
                                memcpy(out.cpu_data, weight_ptr, embed_dim * sizeof(float));
                            }
                        }
                    }
                }
                break;
            }
            case DpxOp::Silu: {
                auto& in_x = active_tensors[in0];
                if (in_x.cpu_data && out.cpu_data) {
                    float* xf = (float*)in_x.cpu_data;
                    float* of = (float*)out.cpu_data;
                    uint32_t size = in_x.shape.size() > 0 ? in_x.shape.back() : 2048;
                    for (uint32_t i = 0; i < size; ++i) {
                        float v = xf[i];
                        of[i] = v / (1.0f + std::exp(-v));
                    }
                }
                break;
            }
            case DpxOp::Mul: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& a = active_tensors[in0];
                        auto& b = active_tensors[in1];
                        if (a.cpu_data && b.cpu_data && out.cpu_data) {
                            float* af = (float*)a.cpu_data; float* bf = (float*)b.cpu_data; float* cf = (float*)out.cpu_data;
                            uint32_t size = a.shape.size() > 0 ? a.shape.back() : 2048;
                            for (uint32_t i = 0; i < size; i++) cf[i] = af[i] * bf[i];
                        }
                    }
                }
                break;
            }
            case DpxOp::Div: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& a = active_tensors[in0];
                        auto& b = active_tensors[in1];
                        if (a.cpu_data && b.cpu_data && out.cpu_data) {
                            float* af = (float*)a.cpu_data; float* bf = (float*)b.cpu_data; float* cf = (float*)out.cpu_data;
                            uint32_t size = a.shape.size() > 0 ? a.shape.back() : 2048;
                            for (uint32_t i = 0; i < size; i++) cf[i] = (bf[i] != 0.0f) ? (af[i] / bf[i]) : 0.0f;
                        }
                    }
                }
                break;
            }
            case DpxOp::Sub: {
                if (node.input_registry_indices.size() >= 2) {
                    int in1 = node.input_registry_indices[1];
                    if (in1 >= 0 && in1 < (int)active_tensors.size()) {
                        auto& a = active_tensors[in0];
                        auto& b = active_tensors[in1];
                        if (a.cpu_data && b.cpu_data && out.cpu_data) {
                            float* af = (float*)a.cpu_data; float* bf = (float*)b.cpu_data; float* cf = (float*)out.cpu_data;
                            uint32_t size = a.shape.size() > 0 ? a.shape.back() : 2048;
                            for (uint32_t i = 0; i < size; i++) cf[i] = af[i] - bf[i];
                        }
                    }
                }
                break;
            }
            case DpxOp::RoPE: {
                extern void cpu_rotary_embedding(const float* in_ptr, const int64_t* pos_ids, const float* cos_cache, const float* sin_cache, float* out_ptr, uint32_t B, uint32_t S, uint32_t Nh, uint32_t Hd, bool interleaved, int rank);
                auto& in_x = active_tensors[in0];
                if (in_x.cpu_data && out.cpu_data) {
                    int64_t default_pos = 0;
                    const int64_t* pos_ptr = &default_pos;
                    const float* cos_ptr = nullptr;
                    const float* sin_ptr = nullptr;

                    if (node.input_registry_indices.size() >= 4) {
                        int in1 = node.input_registry_indices[1];
                        int in2 = node.input_registry_indices[2];
                        int in3 = node.input_registry_indices[3];
                        if (in1 >= 0 && in1 < (int)active_tensors.size() && active_tensors[in1].cpu_data) pos_ptr = (const int64_t*)active_tensors[in1].cpu_data;
                        if (in2 >= 0 && in2 < (int)active_tensors.size()) cos_ptr = (const float*)active_tensors[in2].cpu_data;
                        if (in3 >= 0 && in3 < (int)active_tensors.size()) sin_ptr = (const float*)active_tensors[in3].cpu_data;
                    }

                    uint32_t head_dim = in_x.shape.size() > 0 ? in_x.shape.back() : 256;
                    uint32_t num_heads = in_x.shape.size() > 1 ? in_x.shape[in_x.shape.size() - 2] : 8;
                    int rank = (int)in_x.shape.size();

                    if (cos_ptr && sin_ptr) {
                        cpu_rotary_embedding((float*)in_x.cpu_data, pos_ptr, cos_ptr, sin_ptr, (float*)out.cpu_data, 1, 1, num_heads, head_dim, false, rank);
                    } else {
                        uint64_t total_elems = 1;
                        if (!in_x.shape.empty()) { for (int d : in_x.shape) total_elems *= d; } else { total_elems = num_heads * head_dim; }
                        memcpy(out.cpu_data, in_x.cpu_data, total_elems * sizeof(float));
                    }
                }
                break;
            }
            case DpxOp::ElementWise:
            case DpxOp::Passthrough:
            case DpxOp::Unknown: {
                auto& in_x = active_tensors[in0];
                if (in_x.cpu_data && out.cpu_data) {
                    uint64_t in_size = 1;
                    uint64_t out_size = 1;
                    if (!in_x.shape.empty()) {
                        for (int d : in_x.shape) { if (d > 0) in_size *= d; }
                    } else {
                        in_size = 1;
                    }
                    if (!out.shape.empty()) {
                        for (int d : out.shape) { if (d > 0) out_size *= d; }
                    } else {
                        out_size = in_size;
                    }
                    uint64_t copy_elems = (in_size < out_size) ? in_size : out_size;
                    if (copy_elems > 262144) copy_elems = 262144;
                    memcpy(out.cpu_data, in_x.cpu_data, copy_elems * sizeof(float));
                }
                break;
            }
            case DpxOp::AttentionGQA: {
                auto& in_x = active_tensors[in0];
                if(in_x.cpu_data && out.cpu_data) {
                    if (node.input_registry_indices.size() >= 3) {
                        int in1 = node.input_registry_indices[1];
                        int in2 = node.input_registry_indices[2];
                        if (in1 >= 0 && in1 < (int)active_tensors.size() && in2 >= 0 && in2 < (int)active_tensors.size()) {
                            auto& in_k = active_tensors[in1];
                            auto& in_v = active_tensors[in2];
                            if (in_k.cpu_data && in_v.cpu_data) {
                                cpu_attention_gqa((float*)in_x.cpu_data, (float*)in_k.cpu_data, (float*)in_v.cpu_data, (float*)out.cpu_data, 32, 8, 64, 1);
                            } else {
                                memcpy(out.cpu_data, in_x.cpu_data, 2048 * sizeof(float));
                            }
                        } else {
                            memcpy(out.cpu_data, in_x.cpu_data, 2048 * sizeof(float));
                        }
                    } else {
                        memcpy(out.cpu_data, in_x.cpu_data, 2048 * sizeof(float));
                    }
                }
                break;
            }
            default: break;
        }

        if (out.cpu_data && out.cpu_blit_buffer) {
            uint32_t size = out.shape.size() > 0 ? out.shape.back() : 2048;
            memcpy(out.cpu_blit_buffer, out.cpu_data, size * sizeof(float));
        }
        return true;
    } __except ((s_seh_ex_ptrs = GetExceptionInformation()), EXCEPTION_EXECUTE_HANDLER) {
        auto& faulty_out = active_tensors[out0];
        auto& faulty_in  = active_tensors[in0];
        std::cerr << "[SEH DIAG] Node #" << ni << " OpCode=" << (int)node.op_code
                  << " in0.cpu_data=" << (void*)faulty_in.cpu_data
                  << " out0.cpu_data=" << (void*)faulty_out.cpu_data;
        if (node.input_registry_indices.size() >= 2) {
            int i1 = node.input_registry_indices[1];
            if (i1 >= 0 && i1 < (int)active_tensors.size())
                std::cerr << " in1.cpu_data=" << (void*)active_tensors[i1].cpu_data
                          << " in1.cpu_scales=" << (void*)active_tensors[i1].cpu_scales;
        }
        if (s_seh_ex_ptrs && s_seh_ex_ptrs->ExceptionRecord) {
            std::cerr << " ExCode=0x" << std::hex << s_seh_ex_ptrs->ExceptionRecord->ExceptionCode << std::dec;
            if (s_seh_ex_ptrs->ExceptionRecord->NumberParameters >= 2)
                std::cerr << " fault_addr=" << (void*)s_seh_ex_ptrs->ExceptionRecord->ExceptionInformation[1];
        }
        std::cerr << std::endl;
        return false;
    }
}

void SysGraphOrchestrator::process_token_frame(uint64_t sequence_id, PolicyLatencyDropper& current_dropper) {
    memory_arena.reset_frame_allocations();
        
    for (size_t ni = 0; ni < compute_sequence.size(); ni++) {
        const auto& node = compute_sequence[ni];
        if (node.output_registry_indices.empty() || node.input_registry_indices.empty()) continue;
        
        for (int in_idx : node.input_registry_indices) {
            if (in_idx >= 0 && in_idx < (int)active_tensors.size()) {
                dpx_ensure_resource(active_tensors[in_idx], memory_arena);
            }
        }
        for (int out_idx : node.output_registry_indices) {
            if (out_idx >= 0 && out_idx < (int)active_tensors.size()) {
                dpx_ensure_resource(active_tensors[out_idx], memory_arena);
            }
        }

        int in0 = node.input_registry_indices[0];
        int out0 = node.output_registry_indices[0];
        if (in0 < 0 || in0 >= (int)active_tensors.size() || out0 < 0 || out0 >= (int)active_tensors.size()) continue;

        bool ok = execute_single_node_seh(node, active_tensors, in0, out0, ni, compute_sequence.size());
        if (!ok) {
            std::cerr << "[CRITICAL ENGINE FAULT] Node #" << ni << " (OpCode=" << (int)node.op_code << ") threw SEH memory fault!" << std::endl;
        }

        if (!g_dpx_cpu_only && node.op_code == DpxOp::MatMulQ4) {
            if (active_tensors[node.input_registry_indices[0]].resource && active_tensors[node.input_registry_indices[1]].resource) {
                uint32_t N = active_tensors[node.input_registry_indices[1]].shape.size() > 0 ? active_tensors[node.input_registry_indices[1]].shape[0] : 32000;
                uint32_t K = active_tensors[node.input_registry_indices[1]].shape.size() > 1 ? active_tensors[node.input_registry_indices[1]].shape[1] * 32 : 2048;
                eval_dispatch_gemm_q4_async(
                    active_tensors[node.input_registry_indices[0]], 
                    active_tensors[node.input_registry_indices[1]], 
                    active_tensors[node.output_registry_indices[0]], 
                    1, N, K
                );
            }
        }
    }
}

void SysGraphOrchestrator::allocate_placed_resources() {
    if (active_tensors.empty() || !g_device) return;

    int num_nodes = (int)compute_sequence.size();
    int num_tensors = (int)active_tensors.size();

    // 1. Initialize lifespans
    for (int i = 0; i < num_tensors; ++i) {
        active_tensors[i].first_use = -1;
        active_tensors[i].last_use = -1;
        
        // Static weights already have memory bound; they live forever
        if (active_tensors[i].resource != nullptr) {
            active_tensors[i].first_use = 0;
            active_tensors[i].last_use = num_nodes;
        }
    }

    // 2. Trace execution intervals
    for (int n = 0; n < num_nodes; ++n) {
        const auto& node = compute_sequence[n];
        for (int in_idx : node.input_registry_indices) {
            if (in_idx >= 0 && in_idx < num_tensors) {
                if (active_tensors[in_idx].first_use == -1) {
                    active_tensors[in_idx].first_use = n;
                }
                active_tensors[in_idx].last_use = n;
            }
        }
        for (int out_idx : node.output_registry_indices) {
            if (out_idx >= 0 && out_idx < num_tensors) {
                if (active_tensors[out_idx].first_use == -1) {
                    active_tensors[out_idx].first_use = n;
                }
                active_tensors[out_idx].last_use = n;
            }
        }
    }

    // 3. Define sizes (64KB aligned for D3D12 placed buffers)
    const uint64_t TENSOR_SIZE = 1 * 1024 * 1024; // 1 MB per activation
    const uint64_t ALIGNMENT = 64 * 1024;         // 64 KB D3D12 alignment
    
    std::vector<int> intermediates;
    for (int i = 0; i < num_tensors; ++i) {
        // Find true intermediates that do not contain weight data
        if (active_tensors[i].resource == nullptr && active_tensors[i].first_use != -1) {
            active_tensors[i].required_size = TENSOR_SIZE;
            intermediates.push_back(i);
        }
    }

    // 4. Sort intermediates by creation time
    std::sort(intermediates.begin(), intermediates.end(), [&](int a, int b) {
        return active_tensors[a].first_use < active_tensors[b].first_use;
    });

    // 5. Greedy interval packing
    uint64_t heap_size = 0;
    struct ActiveInterval {
        int tensor_idx;
        uint64_t offset;
        uint64_t size;
        int last_use;
    };
    std::vector<ActiveInterval> active;

    for (int idx : intermediates) {
        int first = active_tensors[idx].first_use;
        
        // Evict intervals whose last usage is completed
        active.erase(std::remove_if(active.begin(), active.end(), [&](const ActiveInterval& act) {
            return act.last_use < first;
        }), active.end());

        // Find the lowest free aligned offset that does not cause an overlap hazard
        uint64_t proposed_offset = 0;
        bool conflict = true;
        while (conflict) {
            conflict = false;
            for (const auto& act : active) {
                uint64_t act_end = act.offset + act.size;
                uint64_t prop_end = proposed_offset + TENSOR_SIZE;
                
                // Overlap collision check with defensive std::max and std::min to avoid MSVC macro expansion
                if ((std::max)(proposed_offset, act.offset) < (std::min)(prop_end, act_end)) {
                    proposed_offset = (act_end + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
                    conflict = true;
                    break;
                }
            }
        }

        active_tensors[idx].heap_offset = proposed_offset;
        heap_size = (std::max)(heap_size, proposed_offset + TENSOR_SIZE);

        active.push_back({idx, proposed_offset, TENSOR_SIZE, active_tensors[idx].last_use});
    }

    if (heap_size == 0) return;

    // 6. Allocate the Single Physical D3D12 Heap
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = heap_size;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 64KB
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

    HRESULT hr = g_device->CreateHeap(&hd, IID_PPV_ARGS(&d3d12_placed_heap));
    if (FAILED(hr)) {
        std::cerr << ">>> [CRITICAL ERROR] CreateHeap FAILED. Requested Size: " 
                  << heap_size / (1024 * 1024) << " MB. HRESULT: " << hr << std::endl;
        return;
    }

    std::cout << ">>> [INTERMEDIATE ACTIVATIONS HEAP] Physical placed heap size: " << heap_size / (1024 * 1024) 
              << " MB (Interval-packed down from " 
              << (intermediates.size() * TENSOR_SIZE) / (1024 * 1024) << " MB activation virtual footprint)." << std::endl;

    // 7. Bind the Placed Resources
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = ALIGNMENT;
    rd.Width = TENSOR_SIZE;
    rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    for (int idx : intermediates) {
        hr = g_device->CreatePlacedResource(
            d3d12_placed_heap.Get(),
            active_tensors[idx].heap_offset,
            &rd,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&active_tensors[idx].resource)
        );
        if (SUCCEEDED(hr) && active_tensors[idx].resource) {
            active_tensors[idx].gpu_va = active_tensors[idx].resource->GetGPUVirtualAddress();
        } else {
            std::cerr << ">>> [ERROR] CreatePlacedResource FAILED at offset: " 
                      << active_tensors[idx].heap_offset << " HRESULT: " << hr << std::endl;
        }
    }
}
