#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "sys_types.h"
#include "sys_memory_arena.h"

struct NodeTopology {
    int id;
    DpxOp op_code;
    std::vector<int> input_registry_indices;
    std::vector<int> output_registry_indices;
    int axis = 0;
    std::vector<int> perm;
};

class SysGraphOrchestrator {
public:
    std::vector<DpxGpuTensor> active_tensors;
    std::vector<NodeTopology> compute_sequence;
    std::unordered_map<std::string, int> tensor_name_to_index; // <-- Exposed mapping
    SysMemoryArena memory_arena;
    int input_tensor_index = -1;
    int output_tensor_index = -1;

    // Shared D3D12 Placed Heap
    Microsoft::WRL::ComPtr<ID3D12Heap> d3d12_placed_heap;

    SysGraphOrchestrator() : memory_arena(1536 * 1024 * 1024) {} // 1.5GB Bump Alloc
    
    void load_from_db(const char* db_path, int target_sig);
    void allocate_placed_resources();
    void process_token_frame(uint64_t sequence_id, class PolicyLatencyDropper& current_dropper);
};
