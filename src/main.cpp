#include "sys_types.h"
#include "sys_precision_config.h"
#include "sys_kernel_dispatch.h"
#include "sys_graph_orchestrator.h"
#include "sys_spsc_ring_buffer.h"
#include "eval_sentencepiece.h"
#include <iostream>
#include <string>
#include <exception>
#include <dxgi1_6.h>

#define NOMINMAX
#include <windows.h>

extern void dpx_init_d3d12();
extern SysGraphOrchestrator g_embed_orchestrator;
extern SysGraphOrchestrator g_decoder_orchestrator;
extern SentencePieceFastUnigram g_tokenizer;
extern Microsoft::WRL::ComPtr<ID3D12Device> g_device;

extern bool g_engine_running;
extern int g_dpx_device_idx;
extern bool g_dpx_cpu_only;
extern int g_dpx_ctx_size;

void sys_run_consumer_loop(SysSPSCRingBuffer& ring, int prompt_size);

// Windows VT100 Neon color initialization
void dpx_enable_ansi_colors() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
}

// DXGI physical hardware adapter scanning
void dpx_list_devices() {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cout << "\033[1;31m[!] Failed to initialize DXGI subsystem.\033[0m\n";
        return;
    }
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    std::cout << "\033[1;33m=== Available Hardware Devices ===\033[0m\n";
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        std::wcout << L"  Device [" << i << L"]: " << desc.Description 
                   << L" (" << (desc.DedicatedVideoMemory / 1024 / 1024) << L" MB dedicated VRAM)\n";
    }
}

int main(int argc, char** argv) {
    dpx_enable_ansi_colors();

    std::string embed_db, decoder_db, spm_path, prompt;
    bool list_devices = false;
    bool server_mode = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-e" && i + 1 < argc) embed_db = argv[++i];
            else if (arg == "-m" && i + 1 < argc) decoder_db = argv[++i];
            else if (arg == "-v" && i + 1 < argc) spm_path = argv[++i];
            else if (arg == "-p" && i + 1 < argc) prompt = argv[++i];
            else if (arg == "-s" || arg == "--server") server_mode = true;
            else if (arg == "-l" || arg == "--list-devices") list_devices = true;
            else if ((arg == "-dev" || arg == "--device") && i + 1 < argc) g_dpx_device_idx = std::stoi(argv[++i]);
            else if (arg == "-ctx" && i + 1 < argc) g_dpx_ctx_size = std::stoi(argv[++i]);
            else if (arg == "-cpu") g_dpx_cpu_only = true;
            else if (arg == "-fp32") g_dpx_precision = DpxPrecisionMode::FP32;
            else if (arg == "-fp16") g_dpx_precision = DpxPrecisionMode::FP16;
            else if (arg == "-int8") g_dpx_precision = DpxPrecisionMode::INT8;
        }

        if (list_devices) {
            dpx_list_devices();
            return 0;
        }

        if (decoder_db.empty() || spm_path.empty()) {
            std::cerr << "\033[1;33m=================================================================\033[0m\n";
            std::cerr << " \033[1;32mDPX NATIVE RUNTIME ENGINE\033[0m\n";
            std::cerr << "\033[1;33m=================================================================\033[0m\n";
            std::cerr << "Usage: dpx.exe -m <decoder.db> -v <vocab.spm> [options]\n\n";
            std::cerr << "Options:\n";
            std::cerr << "  -p \"<text>\"        One-shot generation prompt\n";
            std::cerr << "  -s, --server       Interactive stdio server mode (infinite loop)\n";
            std::cerr << "  -l, --list-devices List DXGI hardware adapters\n";
            std::cerr << "  -dev <id>          Bind custom hardware adapter index\n";
            std::cerr << "  -ctx <size>        Sequence context window (default 4096)\n";
            std::cerr << "  -cpu               Disable GPU, force CPU fallbacks\n";
            std::cerr << "  -int8              8-bit quantized precision mode\n";
            std::cerr << "  -fp16              16-bit half-precision mode\n";
            std::cerr << "  -fp32              32-bit float-precision mode\n";
            std::cerr << "  -e <embed.db>      Embedding DB block\n";
            return 1;
        }

        std::cout << "\033[1;32mDPX model:\033[0m " << decoder_db << "\n";
        std::cout << "\033[1;32mDPX vocab:\033[0m " << spm_path << "\n";
        std::cout.flush();

        dpx_init_cpu_dispatch();

        if (!g_dpx_cpu_only) {
            dpx_init_d3d12();
        } else {
            std::cout << "\033[1;33m[!] Hardware acceleration bypassed. Running CPU threadpool.\033[0m\n";
        }

        if (!embed_db.empty()) g_embed_orchestrator.load_from_db(embed_db.c_str(), 0);
        g_decoder_orchestrator.load_from_db(decoder_db.c_str(), 0);
        g_tokenizer.load_from_file(spm_path.c_str());

        SysSPSCRingBuffer ring(g_dpx_ctx_size, g_dpx_ctx_size);
        g_engine_running = true;

        if (server_mode) {
            std::cout << "\n\033[1;32mDPX Standby (s-mode). Enter prompt:\033[0m\n> ";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "exit" || line == "quit") break;
                if (line.empty()) { std::cout << "> "; continue; }

                std::vector<int> tokens = g_tokenizer.encode(line);
                uint64_t frame_id = ring.consumer_get_freshest() + 1;
                for (int t : tokens) {
                    float token_val = static_cast<float>(t);
                    ring.producer_push(frame_id++, &token_val, sizeof(float));
                }

                sys_run_consumer_loop(ring, tokens.size());
                std::cout << "\n\033[1;32mDPX Standby (s-mode). Enter prompt:\033[0m\n> ";
            }
        } else if (!prompt.empty()) {
            std::string formatted_prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
            std::vector<int> tokens;
            tokens.push_back(2); // <bos>
            std::vector<int> encoded = g_tokenizer.encode(formatted_prompt);
            tokens.insert(tokens.end(), encoded.begin(), encoded.end());

            uint64_t frame_id = 1;
            for (int t : tokens) {
                float token_val = static_cast<float>(t);
                ring.producer_push(frame_id++, &token_val, sizeof(float));
            }
            std::cout << "\033[1;33m" << prompt << "\033[0m\n";
            std::cout.flush();

            sys_run_consumer_loop(ring, tokens.size());
        } else {
            std::cerr << "\033[1;31m[!] Error: Please provide either a prompt (-p) or launch server mode (-s).\033[0m\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n\033[1;31m[FATAL EXECUTION FAULT] " << e.what() << "\033[0m\n";
        return 1;
    } catch (...) {
        std::cerr << "\n\033[1;31m[FATAL EXECUTION FAULT] Unknown memory exception caught.\033[0m\n";
        return 1;
    }
    return 0;
}
