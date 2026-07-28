#include "sys_kv_cache_manager.h"

SysKvCacheManager g_kv_cache_manager;

void SysKvCacheManager::reset() {
    cache.clear();
    current_step = 0;
}

void SysKvCacheManager::update_cache(const std::string& present_name, const float* data, const std::vector<int>& shape) {
    if (!data || shape.empty()) return;

    std::string past_name = present_name;
    size_t pos = past_name.find("present");
    if (pos != std::string::npos) {
        past_name.replace(pos, 7, "past");
    }

    uint64_t total = 1;
    for (int d : shape) total *= d;

    auto& kv = cache[past_name];
    kv.name = past_name;
    kv.shape = shape;
    kv.data.assign(data, data + total);
}

const float* SysKvCacheManager::get_cache(const std::string& past_name) const {
    auto it = cache.find(past_name);
    if (it != cache.end() && !it->second.data.empty()) {
        return it->second.data.data();
    }
    return nullptr;
}
