#include "sys_graph_orchestrator.h"
#include <sqlite3.h>
#include <unordered_map>
#include <iostream>
#include <string>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;

DpxOp parse_op_code(const std::string& op) {
    if (op == "MatMulNBits" || op == "MatMul" || op == "Gemm" || op == "QLinearMatMul") return DpxOp::MatMulQ4;
    if (op == "Add" || op == "Sum") return DpxOp::Add;
    if (op == "SimplifiedLayerNormalization" || op == "LayerNormalization" || op == "SkipSimplifiedLayerNormalization") return DpxOp::RMSNorm;
    if (op == "GroupQueryAttention" || op == "Attention" || op == "MultiHeadAttention") return DpxOp::AttentionGQA;
    if (op == "Gather" || op == "GatherElements" || op == "GatherND" || op == "GatherBlockQuantized" || op == "EmbedLayerNormalization") return DpxOp::Gather;
    if (op == "Mul") return DpxOp::Mul;
    if (op == "Div") return DpxOp::Div;
    if (op == "Sub") return DpxOp::Sub;
    if (op == "Sigmoid" || op == "Silu" || op == "Gelu" || op == "FastGelu" || op == "QuickGelu") return DpxOp::Silu;
    if (op == "RotaryEmbedding" || op == "RoPE") return DpxOp::RoPE;
    if (op == "Reshape" || op == "Squeeze" || op == "Unsqueeze" || op == "Identity" || op == "Cast" || op == "Concat" || op == "Transpose" || op == "Expand") return DpxOp::Passthrough;
    return DpxOp::Unknown;
}

void SysGraphOrchestrator::load_from_db(const char* db_path, int target_sig) {
    sqlite3* db;
    if (sqlite3_open(db_path, &db) != SQLITE_OK) {
        std::cerr << "Failed to mount engine core DB: " << db_path << std::endl;
        return;
    }
                            
    tensor_name_to_index.clear(); // Populate class member
    int next_index = 0;
    input_tensor_index = -1;
    output_tensor_index = -1;
            
    auto get_or_alloc_index = [&](const std::string& name) -> int {
        if (tensor_name_to_index.find(name) == tensor_name_to_index.end()) {
            tensor_name_to_index[name] = next_index++;
            active_tensors.push_back({});
            active_tensors.back().name = name;
            
            if (name == "input_ids" || name == "inputs_embeds") {
                input_tensor_index = tensor_name_to_index[name];
            }
            if (name == "logits" || name == "output_logits") {
                output_tensor_index = tensor_name_to_index[name];
            }
        }
        return tensor_name_to_index[name];
    };
    // 0. READ GRAPH I/O TOPOLOGY FROM DB
    sqlite3_stmt* stmt_io;
    if (sqlite3_prepare_v2(db, "SELECT kind, name, elem_type, shape FROM graph_io", -1, &stmt_io, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt_io) == SQLITE_ROW) {
            const char* kind_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt_io, 0));
            const char* name_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt_io, 1));
            const char* shape_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt_io, 3));
            if (kind_text && name_text) {
                std::string kind = kind_text;
                std::string name = name_text;
                int idx = get_or_alloc_index(name);
                if (kind == "in" && (name == "inputs_embeds" || name == "input_ids" || (input_tensor_index == -1 && name.find("past") == std::string::npos))) {
                    input_tensor_index = idx;
                }
                if (kind == "out" && (name == "logits" || (output_tensor_index == -1 && name.find("present") == std::string::npos))) {
                    output_tensor_index = idx;
                }
                if (shape_text) {
                    std::string s(shape_text);
                    size_t pos = 0;
                    while (pos < s.length()) {
                        size_t comma = s.find(',', pos);
                        if (comma == std::string::npos) comma = s.length();
                        if (comma > pos) {
                            std::string dim_str = s.substr(pos, comma - pos);
                            try {
                                active_tensors[idx].shape.push_back(std::stoi(dim_str));
                            } catch (...) {
                                active_tensors[idx].shape.push_back(1);
                            }
                        }
                        pos = comma + 1;
                    }
                }
            }
        }
        sqlite3_finalize(stmt_io);
    }

    // 1. EXTRACT WEIGHT BLOBs INTO D3D12 VRAM
    // dtype column from DB provides the ONNX data type code:
    //   1=FLOAT32, 10=FLOAT16, 16=BFLOAT16, 2=UINT8, 3=INT8, 7=INT64
    sqlite3_stmt* stmt_tensors;
    sqlite3_prepare_v2(db, "SELECT name, data, dims, dtype FROM tensor WHERE data IS NOT NULL", -1, &stmt_tensors, nullptr);
    size_t loaded_bytes = 0;
    int allocated_count = 0;

    // ONNX data type constants
    constexpr int ONNX_FLOAT32  = 1;
    constexpr int ONNX_UINT8    = 2;
    constexpr int ONNX_INT8     = 3;
    constexpr int ONNX_FLOAT16  = 10;
    constexpr int ONNX_BFLOAT16 = 16;

    // FP16->FP32 expansion helper (portable bit-manipulation, no F16C intrinsic needed)
    auto expand_fp16_to_fp32 = [](const void* src_blob, int src_bytes) -> void* {
        int num_elems = src_bytes / 2;
        float* fp32_buf = (float*)_aligned_malloc((size_t)num_elems * sizeof(float), 64);
        if (!fp32_buf) return nullptr;
        const uint16_t* src = (const uint16_t*)src_blob;
        for (int i = 0; i < num_elems; i++) {
            uint32_t h = src[i];
            uint32_t sign     = (h & 0x8000u) << 16;
            uint32_t exponent = (h & 0x7C00u) >> 10;
            uint32_t mantissa = (h & 0x03FFu);
            uint32_t f32;
            if (exponent == 0) {
                if (mantissa == 0) { f32 = sign; }
                else {
                    exponent = 1;
                    while (!(mantissa & 0x0400)) { mantissa <<= 1; exponent--; }
                    mantissa &= 0x03FF;
                    f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
            } else if (exponent == 31) {
                f32 = sign | 0x7F800000u | (mantissa << 13);
            } else {
                f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
            memcpy(&fp32_buf[i], &f32, sizeof(float));
        }
        return fp32_buf;
    };

    // BF16->FP32 expansion helper (BF16 = top 16 bits of FP32)
    auto expand_bf16_to_fp32 = [](const void* src_blob, int src_bytes) -> void* {
        int num_elems = src_bytes / 2;
        float* fp32_buf = (float*)_aligned_malloc((size_t)num_elems * sizeof(float), 64);
        if (!fp32_buf) return nullptr;
        const uint16_t* src = (const uint16_t*)src_blob;
        for (int i = 0; i < num_elems; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&fp32_buf[i], &bits, sizeof(float));
        }
        return fp32_buf;
    };
    
    while (sqlite3_step(stmt_tensors) == SQLITE_ROW) {
        std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt_tensors, 0));
        const void* blob = sqlite3_column_blob(stmt_tensors, 1);
        int bytes = sqlite3_column_bytes(stmt_tensors, 1);
        const char* dims_cstr = reinterpret_cast<const char*>(sqlite3_column_text(stmt_tensors, 2));
        std::string dims_str = dims_cstr ? dims_cstr : "";
        int dtype = sqlite3_column_int(stmt_tensors, 3);  // ONNX data type from DB
        loaded_bytes += bytes;
        
        std::string base_name = name;
        bool is_scale = false, is_zp = false;
        if (name.find("_scale") != std::string::npos) {
            base_name = name.substr(0, name.find("_scale"));
            is_scale = true;
        } else if (name.find("_zp") != std::string::npos) {
            base_name = name.substr(0, name.find("_zp"));
            is_zp = true;
        }
        
        int idx = get_or_alloc_index(base_name);
        auto& t = active_tensors[idx];

        // Register exact name alias for scale/zp so node_io lookups resolve correctly
        if ((is_scale || is_zp) && tensor_name_to_index.find(name) == tensor_name_to_index.end()) {
            tensor_name_to_index[name] = idx;
        }

        if (t.shape.empty() && !dims_str.empty()) {
            size_t pos = 0;
            while(pos < dims_str.length()) {
                size_t comma = dims_str.find(',', pos);
                if(comma == std::string::npos) comma = dims_str.length();
                if(comma > pos) {
                    try { t.shape.push_back(std::stoi(dims_str.substr(pos, comma - pos))); }
                    catch (...) { t.shape.push_back(1); }
                }
                pos = comma + 1;
            }
        }
        
        extern bool g_dpx_cpu_only;
        if (bytes <= 0) {
            std::cerr << ">>> [WARNING] Skip empty weight allocation: " << name << std::endl;
            continue;
        }

        // --- Scale and ZP tensors: always decode to FP32 on CPU ---
        if (is_scale) {
            if (dtype == ONNX_FLOAT16) {
                t.cpu_scales = expand_fp16_to_fp32(blob, bytes);
            } else if (dtype == ONNX_BFLOAT16) {
                t.cpu_scales = expand_bf16_to_fp32(blob, bytes);
            } else {
                void* buf = _aligned_malloc(bytes, 64);
                if (buf) memcpy(buf, blob, bytes);
                t.cpu_scales = buf;
            }
            allocated_count++;
            continue;
        } else if (is_zp) {
            void* cpu_blob = _aligned_malloc(bytes, 64);
            if (cpu_blob) memcpy(cpu_blob, blob, bytes);
            t.cpu_zp = cpu_blob;
            allocated_count++;
            continue;
        }

        // --- Regular weight tensors: use dtype to determine handling ---
        if (dtype == ONNX_FLOAT16) {
            // FP16 weight (e.g. RMSNorm gamma, bias, embedding) -> expand to FP32
            t.cpu_data = expand_fp16_to_fp32(blob, bytes);
            t.is_q4 = false;
            allocated_count++;
        } else if (dtype == ONNX_BFLOAT16) {
            // BF16 weight -> expand to FP32
            t.cpu_data = expand_bf16_to_fp32(blob, bytes);
            t.is_q4 = false;
            allocated_count++;
        } else if (dtype == ONNX_UINT8 || dtype == ONNX_INT8) {
            // Quantized packed weight blob — load raw bytes
            if (g_device && !g_dpx_cpu_only) {
                D3D12_HEAP_PROPERTIES heapProps = {};
                heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC resDesc = {};
                resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                resDesc.Width = bytes;
                resDesc.Height = 1; resDesc.DepthOrArraySize = 1; resDesc.MipLevels = 1;
                resDesc.SampleDesc.Count = 1; resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                
                Microsoft::WRL::ComPtr<ID3D12Resource> temp_res;
                HRESULT hr = g_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&temp_res));
                if (FAILED(hr)) {
                    std::cerr << ">>> [CRITICAL] CreateCommittedResource FAILED for " << name << " HRESULT: " << hr << std::endl;
                    break;
                }
                void* pData = nullptr;
                temp_res->Map(0, nullptr, &pData);
                if (pData) memcpy(pData, blob, bytes);
                t.resource = temp_res;
                t.gpu_va = temp_res->GetGPUVirtualAddress();
                t.cpu_data = pData;
            } else {
                void* cpu_blob = _aligned_malloc(bytes, 64);
                if (cpu_blob) memcpy(cpu_blob, blob, bytes);
                t.cpu_data = cpu_blob;
            }
            t.is_q4 = true;
            allocated_count++;
        } else {
            // FP32 / INT64 / other — load as-is
            if (g_device && !g_dpx_cpu_only) {
                D3D12_HEAP_PROPERTIES heapProps = {};
                heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC resDesc = {};
                resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                resDesc.Width = bytes;
                resDesc.Height = 1; resDesc.DepthOrArraySize = 1; resDesc.MipLevels = 1;
                resDesc.SampleDesc.Count = 1; resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                
                Microsoft::WRL::ComPtr<ID3D12Resource> temp_res;
                HRESULT hr = g_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&temp_res));
                if (FAILED(hr)) {
                    std::cerr << ">>> [CRITICAL] CreateCommittedResource FAILED for " << name << " HRESULT: " << hr << std::endl;
                    break;
                }
                void* pData = nullptr;
                temp_res->Map(0, nullptr, &pData);
                if (pData) memcpy(pData, blob, bytes);
                t.resource = temp_res;
                t.gpu_va = temp_res->GetGPUVirtualAddress();
                t.cpu_data = pData;
            } else {
                void* cpu_blob = _aligned_malloc(bytes, 64);
                if (cpu_blob) memcpy(cpu_blob, blob, bytes);
                t.cpu_data = cpu_blob;
            }
            t.is_q4 = false;
            allocated_count++;
        }
    }
    sqlite3_finalize(stmt_tensors);

    // 2. LOAD NODE TOPOLOGY
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, "SELECT id, op_type FROM node ORDER BY ord", -1, &stmt, nullptr);
                
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int node_id = sqlite3_column_int(stmt, 0);
        const char* op_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        std::string op = op_text ? op_text : "";
                            
        NodeTopology topo;
        topo.id = node_id;
        topo.op_code = parse_op_code(op);
                        
        sqlite3_stmt* stmt2;
        sqlite3_prepare_v2(db, "SELECT kind, value_name FROM node_io WHERE node_id = ?", -1, &stmt2, nullptr);
        sqlite3_bind_int(stmt2, 1, node_id);
                                        
        while (sqlite3_step(stmt2) == SQLITE_ROW) {
            const char* kind_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt2, 0));
            const char* val_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt2, 1));
            std::string kind = kind_text ? kind_text : "";
            std::string value_name = val_text ? val_text : "";
                                            
            if (value_name.empty()) continue;
                        
            int resolved_index = get_or_alloc_index(value_name);
            if (kind == "in") topo.input_registry_indices.push_back(resolved_index);
            if (kind == "out") topo.output_registry_indices.push_back(resolved_index);
        }
        sqlite3_finalize(stmt2);
        compute_sequence.push_back(topo);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
                    
    std::cout << ">> DB LOADED. " << compute_sequence.size() << " nodes bound. " 
              << (loaded_bytes / 1024 / 1024) << " MB weights cached natively." << std::endl;
    std::cout << ">> Detected Graph Inputs/Outputs: Input Index=" << input_tensor_index << ", Output Index=" << output_tensor_index << std::endl;
    if (input_tensor_index >= 0 && input_tensor_index < (int)active_tensors.size()) {
        std::cout << ">> Input Tensor Name: " << active_tensors[input_tensor_index].name << std::endl;
    }
    if (output_tensor_index >= 0 && output_tensor_index < (int)active_tensors.size()) {
        std::cout << ">> Output Tensor Name: " << active_tensors[output_tensor_index].name << std::endl;
    }

    allocate_placed_resources();
}


