#pragma once
#include "sys_types.h"
#include <atomic>
#include <string.h>
#include <vector>

extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;

struct ActiveFrame {
    uint64_t monotonic_id = 0;
    DpxGpuTensor input_activations;
    void* mapped_data = nullptr;
};

extern bool g_dpx_cpu_only;
class SysSPSCRingBuffer {
private:
    std::vector<ActiveFrame> slots;
    uint32_t ring_capacity;
    std::atomic<uint64_t> head{0};
    std::atomic<uint64_t> tail{0};

public:
    SysSPSCRingBuffer(uint32_t width, uint32_t height, uint32_t depth = 1024) : ring_capacity(depth) {
        slots.resize(ring_capacity);
        if (g_dpx_cpu_only) {
            for (uint32_t i = 0; i < ring_capacity; ++i) {
                slots[i].input_activations.shape = { (int)width, (int)height };
                slots[i].mapped_data = calloc(width * height, sizeof(float));
                slots[i].input_activations.cpu_data = slots[i].mapped_data;
            }
            return;
        }
        if (!g_device) return;
        D3D12_HEAP_PROPERTIES hp = {}; 
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        
        D3D12_RESOURCE_DESC rd = {}; 
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = width * height * sizeof(float);
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        
        for (uint32_t i = 0; i < ring_capacity; ++i) {
            slots[i].input_activations.shape = { (int)width, (int)height };
            g_device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd, 
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, 
                IID_PPV_ARGS(&slots[i].input_activations.resource)
            );
            if (slots[i].input_activations.resource) {
                slots[i].input_activations.gpu_va = slots[i].input_activations.resource->GetGPUVirtualAddress();
                slots[i].input_activations.resource->Map(0, nullptr, &slots[i].mapped_data);
            }
        }
    }

    ~SysSPSCRingBuffer() {
        for (uint32_t i = 0; i < ring_capacity; ++i) {
            if (slots[i].input_activations.resource) {
                slots[i].input_activations.resource->Unmap(0, nullptr);
            }
        }
    }

    void producer_push(uint64_t frame_id, const void* incoming_data, size_t bytes = sizeof(float)) {
        uint32_t s = frame_id % ring_capacity;
        if (slots[s].mapped_data && incoming_data && bytes > 0) {
            memcpy(slots[s].mapped_data, incoming_data, bytes);
        }
        slots[s].monotonic_id = frame_id;
        uint64_t h = head.load(std::memory_order_relaxed);
        if (frame_id + 1 > h) {
            head.store(frame_id + 1, std::memory_order_release);
        }
    }

    uint64_t consumer_get_freshest() const {
        return head.load(std::memory_order_acquire);
    }

    bool consumer_pop(ActiveFrame& out_frame) {
        uint64_t t = tail.load(std::memory_order_relaxed);
        uint64_t h = head.load(std::memory_order_acquire);
        if (t >= h) return false;
        out_frame = slots[t % ring_capacity];
        tail.store(t + 1, std::memory_order_release);
        return true;
    }

    ActiveFrame& get_frame_memory(uint64_t frame_id) {
        return slots[frame_id % ring_capacity];
    }
};


