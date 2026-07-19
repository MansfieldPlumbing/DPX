#pragma once
#include <chrono>
#include <string>

// KOKORO DEADLINE DROPS
class PolicyLatencyDropper {
    std::chrono::high_resolution_clock::time_point frame_deadline;
    int budget_ms;

public:
    PolicyLatencyDropper(int target_ms) : budget_ms(target_ms) {}

    void begin_frame_clock() {
        frame_deadline = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(budget_ms);
    }

    bool evaluate_drop_exceeds_budget() const {
        return std::chrono::high_resolution_clock::now() > frame_deadline;
    }

    std::chrono::milliseconds check_idle_budget() const {
        auto now = std::chrono::high_resolution_clock::now();
        if (now < frame_deadline) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(frame_deadline - now);
        }
        return std::chrono::milliseconds(0);
    }
};
