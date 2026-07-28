#include "sys_token_sampler.h"
#include <cmath>
#include <iostream>

int sys_sample_argmax(const float* logits, const std::vector<int>& shape) {
    if (!logits) return 0;

    uint32_t vocab_size = 256000;
    if (!shape.empty() && shape.back() > 0) {
        vocab_size = static_cast<uint32_t>(shape.back());
    }

    uint64_t total_elements = vocab_size;
    if (!shape.empty()) {
        total_elements = 1;
        for (int d : shape) total_elements *= d;
    }

    uint64_t last_token_offset = (total_elements >= vocab_size) ? (total_elements - vocab_size) : 0;
    const float* last_logits = logits + last_token_offset;

    int best_token = 0;
    float best_score = -1e9f;
    std::vector<std::pair<float, int>> top_k;
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (last_logits[i] > best_score) {
            best_score = last_logits[i];
            best_token = static_cast<int>(i);
        }
        if (top_k.size() < 5) {
            top_k.push_back({last_logits[i], (int)i});
        } else {
            for (size_t k = 0; k < 5; k++) {
                if (last_logits[i] > top_k[k].first) {
                    top_k[k] = {last_logits[i], (int)i};
                    break;
                }
            }
        }
    }
    std::cout << "\n[TOP 5 LOGITS] Argmax Token=" << best_token << " Score=" << best_score << std::endl;
    for (size_t k = 0; k < top_k.size(); k++) {
        std::cout << "  #" << k << ": Token=" << top_k[k].second << " Score=" << top_k[k].first << std::endl;
    }
    std::cout << "\n[DEBUG SAMPLER] total=" << total_elements << " vocab=" << vocab_size << " offset=" << last_token_offset << " best_token=" << best_token << " best_score=" << best_score << std::endl;
    return best_token;
}
