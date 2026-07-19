#include "sys_memory_arena.h"
#include <stdexcept>
#include <malloc.h>

SysMemoryArena::SysMemoryArena(uint64_t size_bytes) : total_capacity(size_bytes), current_offset(0) {
    memory_base = static_cast<uint8_t*>(_aligned_malloc(size_bytes, 256));
    if (!memory_base) throw std::runtime_error("FATAL: Memory Arena OOM pre-allocating Memory.");
}

SysMemoryArena::~SysMemoryArena() {
    _aligned_free(memory_base);
}

void* SysMemoryArena::allocate_transient(uint64_t size_bytes, uint64_t alignment) {
    uint64_t remainder = current_offset % alignment;
    uint64_t padding = (remainder == 0) ? 0 : (alignment - remainder);
    if (current_offset + padding + size_bytes > total_capacity) {
        throw std::runtime_error("FATAL: Frame Limits OOM Exceeded.");
    }
    current_offset += padding;
    void* ptr = memory_base + current_offset;
    current_offset += size_bytes;
    return ptr;
}

// 0-cost reclaim! Memory simply rolls back its target
void SysMemoryArena::reset_frame_allocations() {
    current_offset = 0;
}
