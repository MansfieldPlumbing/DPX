#pragma once
#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

struct KvCacheTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
};

class SysKvCacheManager {
public:
    std::unordered_map<std::string, KvCacheTensor> cache;
    uint32_t current_step = 0;

    void reset();
    void update_cache(const std::string& present_name, const float* data, const std::vector<int>& shape);
    const float* get_cache(const std::string& past_name) const;
};

extern SysKvCacheManager g_kv_cache_manager;
