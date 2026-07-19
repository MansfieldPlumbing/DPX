#include "sys_graph_orchestrator.h"
#include <sqlite3.h>
#include <unordered_map>
#include <iostream>
#include <string>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;

DpxOp parse_op_code(const std::string& op) {
    if (op == "MatMulNBits" || op == "MatMul") return DpxOp::MatMulQ4;
    if (op == "Add") return DpxOp::Add;
    if (op == "SimplifiedLayerNormalization" || op == "LayerNormalization") return DpxOp::RMSNorm;
    if (op == "GroupQueryAttention") return DpxOp::AttentionGQA;
    return DpxOp::Unknown;
}

void SysGraphOrchestrator::load_from_db(const char* db_path, int target_sig) {
    sqlite3* db;
    if (sqlite3_open(db_path, &db) != SQLITE_OK) {
        std::cerr << "Failed to mount engine core DB: " << db_path << std::endl;
        return;
    }
                            
    std::unordered_map<std::string, int> tensor_name_to_index;
    int next_index = 0;
            
    auto get_or_alloc_index = [&](const std::string& name) {
        if (tensor_name_to_index.find(name) == tensor_name_to_index.end()) {
            tensor_name_to_index[name] = next_index++;
            active_tensors.push_back({});
        }
        return tensor_name_to_index[name];
    };

    // 1. EXTRACT WEIGHT BLOBs INTO D3D12 VRAM
    sqlite3_stmt* stmt_tensors;
    sqlite3_prepare_v2(db, "SELECT name, data FROM tensor WHERE data IS NOT NULL", -1, &stmt_tensors, nullptr);
    size_t loaded_bytes = 0;
    
    while (sqlite3_step(stmt_tensors) == SQLITE_ROW) {
        std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt_tensors, 0));
        const void* blob = sqlite3_column_blob(stmt_tensors, 1);
        int bytes = sqlite3_column_bytes(stmt_tensors, 1);
        loaded_bytes += bytes;
        
        int idx = get_or_alloc_index(name);
        auto& t = active_tensors[idx];
        
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD; // Direct upload for weights
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width = bytes;
        resDesc.Height = 1; resDesc.DepthOrArraySize = 1; resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1; resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        
        g_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&t.resource));
        
        void* pData;
        t.resource->Map(0, nullptr, &pData);
        memcpy(pData, blob, bytes);
        t.resource->Unmap(0, nullptr);
        
        t.gpu_va = t.resource->GetGPUVirtualAddress();
        
        // Tag Q4 auxiliary bindings
        if (name.find("_scale") != std::string::npos) t.scales_resource = t.resource;
        else if (name.find("_zp") != std::string::npos) t.zp_resource = t.resource;
        else t.is_q4 = true; // Assume main weight blob
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
}
