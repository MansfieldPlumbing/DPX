#pragma once
#include <vector>
#include <string>
#include "sys_types.h"
#include "sys_memory_arena.h"

enum class DpxOp { 
    MatMulQ4, Add, RMSNorm, AttentionGQA, Silu, Unknown
};

// MECHANISTIC PIPELINE MAP 
// Hardware integer execution offsets instead of 'Dictionary<string, Tensor>'
struct NodeTopology {
    int id;
    DpxOp op_code;
    std::vector<int> input_registry_indices;
    std::vector<int> output_registry_indices;
};

class SysGraphOrchestrator {
public:
    std::vector<DpxGpuTensor> active_tensors;
    std::vector<NodeTopology> compute_sequence;
    SysMemoryArena memory_arena;

    SysGraphOrchestrator() : memory_arena(1536 * 1024 * 1024) {} // 1.5GB Bump Alloc
    
    // Topology is loaded at boot
    void load_from_db(const char* db_path, int target_sig);
    
    // Evaluation avoids string searches entirely
    void process_token_frame(uint64_t sequence_id, class PolicyLatencyDropper& current_dropper);
};
