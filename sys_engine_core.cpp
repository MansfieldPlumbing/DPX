#include "sys_types.h"
#include "sys_graph_orchestrator.h"
#include "sys_spsc_ring_buffer.h"
#include "eval_sentencepiece.h"
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <dxgi1_6.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

typedef void (*DpxTokenCallback)(const char*);

Microsoft::WRL::ComPtr<ID3D12Device> g_device;
Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_compute_queue;
Microsoft::WRL::ComPtr<ID3D12RootSignature> g_root;
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pso_gemv;
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pso_tiled;

SysSPSCRingBuffer* g_ring_buffer = nullptr;
std::thread g_consumer_thread;
bool g_engine_running = false;

// Global settings defined in the shared core
int g_dpx_device_idx = -1;
bool g_dpx_cpu_only = false;
int g_dpx_ctx_size = 4096;

// Allocate real storage for the UI callback
DpxTokenCallback g_ui_callback = nullptr;

SysGraphOrchestrator g_embed_orchestrator;
SysGraphOrchestrator g_decoder_orchestrator;
SentencePieceFastUnigram g_tokenizer;

// Fixed signature with default argument to satisfy both old 1-arg calls and new 2-arg loops
void sys_run_consumer_loop(SysSPSCRingBuffer& ring, int prompt_size = 1);

static std::vector<uint8_t> read_dxil_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    f.read((char*)buffer.data(), size);
    return buffer;
}

void dpx_init_d3d12() {
    if (g_dpx_cpu_only) return;

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << ">>> [ERROR] CreateDXGIFactory1 failed! HRESULT: " << hr << std::endl;
        return;
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        
        if (g_dpx_device_idx >= 0 && i != (UINT)g_dpx_device_idx) continue;

        hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_device));
        if (SUCCEEDED(hr)) {
            std::wcout << L">>> D3D12 GPU Mapped: " << desc.Description << std::endl;
            break;
        }
    }

    if (!g_device) {
        std::cerr << ">>> [WARNING] Failed to map requested D3D12 GPU! Falling back to CPU.\n";
        g_dpx_cpu_only = true;
        return;
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    hr = g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_compute_queue));
    if (FAILED(hr)) {
        std::cerr << ">>> [ERROR] CreateCommandQueue failed! HRESULT: " << hr << std::endl;
        return;
    }

    D3D12_ROOT_PARAMETER rp[6] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.Num32BitValues = 5;
    rp[0].Constants.ShaderRegister = 0;

    for (int i = 1; i <= 4; ++i) {
        rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[i].Descriptor.ShaderRegister = i - 1;
        rp[i].Descriptor.RegisterSpace = 0;
    }
    rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[5].Descriptor.ShaderRegister = 0;
    rp[5].Descriptor.RegisterSpace = 0;

    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 6;
    rsd.pParameters = rp;
    Microsoft::WRL::ComPtr<ID3DBlob> sig, err;
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (SUCCEEDED(hr)) {
        hr = g_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&g_root));
        if (FAILED(hr)) {
            std::cerr << ">>> [ERROR] CreateRootSignature failed! HRESULT: " << hr << std::endl;
        }
    } else {
        std::cerr << ">>> [ERROR] D3D12SerializeRootSignature failed! HRESULT: " << hr << std::endl;
    }

    auto gemv_dxil = read_dxil_file("shaders/gemm_q4_gemv.dxil");
    auto tiled_dxil = read_dxil_file("shaders/gemm_q4_tiled.dxil");

    if (gemv_dxil.empty()) {
        std::cerr << ">>> [ERROR] shaders/gemm_q4_gemv.dxil not found or empty!" << std::endl;
    } else {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g_root.Get();
        pd.CS.pShaderBytecode = gemv_dxil.data();
        pd.CS.BytecodeLength = gemv_dxil.size();
        hr = g_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_pso_gemv));
        if (FAILED(hr)) {
            std::cerr << ">>> [ERROR] CreateComputePipelineState for GEMV failed! HRESULT: " << hr << std::endl;
        } else {
            std::cout << ">>> [OK] GEMV Compute Pipeline State successfully bound." << std::endl;
        }
    }

    if (tiled_dxil.empty()) {
        std::cerr << ">>> [ERROR] shaders/gemm_q4_tiled.dxil not found or empty!" << std::endl;
    } else {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g_root.Get();
        pd.CS.pShaderBytecode = tiled_dxil.data();
        pd.CS.BytecodeLength = tiled_dxil.size();
        hr = g_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_pso_tiled));
        if (FAILED(hr)) {
            std::cerr << ">>> [ERROR] CreateComputePipelineState for Tiled failed! HRESULT: " << hr << std::endl;
        } else {
            std::cout << ">>> [OK] Tiled Compute Pipeline State successfully bound." << std::endl;
        }
    }
}

extern "C" __declspec(dllexport) void dpx_engine_start() {
    if (g_engine_running) return;
    dpx_init_d3d12();
    g_ring_buffer = new SysSPSCRingBuffer(4096, 4096);
    g_engine_running = true;
    g_consumer_thread = std::thread([]() { sys_run_consumer_loop(*g_ring_buffer); });
}

extern "C" __declspec(dllexport) void dpx_engine_stop() {
    g_engine_running = false;
    if (g_consumer_thread.joinable()) g_consumer_thread.join();
    delete g_ring_buffer;
}
