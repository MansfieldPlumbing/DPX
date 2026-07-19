#include "sys_types.h"
#include "sys_graph_orchestrator.h"
#include "sys_spsc_ring_buffer.h"
#include "eval_sentencepiece.h"
#include <atomic>
#include <vector>

extern "C" __declspec(dllexport) void dpx_engine_start();
extern "C" __declspec(dllexport) void dpx_engine_stop();

extern SysGraphOrchestrator g_embed_orchestrator;
extern SysGraphOrchestrator g_decoder_orchestrator;
extern SysSPSCRingBuffer* g_ring_buffer;
extern SentencePieceFastUnigram g_tokenizer;

std::atomic<int> g_realtime_budget_ms{30};
std::atomic<bool> g_light_up_experts{false};

extern "C" __declspec(dllexport) void dpx_set_budget(int target_ms) {
    g_realtime_budget_ms.store(target_ms, std::memory_order_relaxed);
}

extern "C" __declspec(dllexport) void dpx_toggle_background_experts(bool enabled) {
    g_light_up_experts.store(enabled, std::memory_order_relaxed);
}

extern "C" __declspec(dllexport) void dpx_load_models(const char* embed_db, const char* decoder_db, const char* spm_path) {
    g_embed_orchestrator.load_from_db(embed_db, 0);
    g_decoder_orchestrator.load_from_db(decoder_db, 0);
    g_tokenizer.load_from_file(spm_path);
}

extern "C" __declspec(dllexport) void dpx_push_prompt(const char* text) {
    if (!g_ring_buffer) return;
    std::vector<int> tokens = g_tokenizer.encode(text);
    uint64_t frame_id = g_ring_buffer->consumer_get_freshest() + 1;
    for (int t : tokens) {
        float token_val = static_cast<float>(t);
        g_ring_buffer->producer_push(frame_id++, &token_val, sizeof(float));
    }
}
