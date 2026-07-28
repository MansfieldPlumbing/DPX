#pragma once
#include <cstdint>

// ZERO-GC BUMP ALLOCATOR
class SysMemoryArena {
    uint8_t* memory_base;
    uint64_t total_capacity;
    uint64_t current_offset;
public:
    SysMemoryArena(uint64_t size_bytes);
    ~SysMemoryArena();
    void* allocate_transient(uint64_t size_bytes, uint64_t alignment = 256);
    void reset_frame_allocations();
};
