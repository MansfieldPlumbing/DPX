cmake_minimum_required(VERSION 3.10)
project(dpx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(CORE_SOURCES
    sqlite3.c 
    sys_engine_core.cpp
    sys_consumer_thread.cpp
    sys_topology_loader.cpp
    eval_compute_q4_gemm.cpp
    eval_background_experts.cpp
    d3d12_dispatch.cpp
    ops_attention.cpp
    ops_matmul_cpu.cpp
    ops_kv_rope.cpp
    ops_signal_audio.cpp
    ops_conv_deconv.cpp
    eval_sentencepiece.cpp
)

# 1. Build the shared DLL
add_library(dpx_engine SHARED dpx_api_mount.cpp ${CORE_SOURCES})
target_include_directories(dpx_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dpx_engine PRIVATE d3d12.lib dxgi.lib synchronization.lib)

# 2. Build the standalone EXE
add_executable(dpx main.cpp ${CORE_SOURCES})
target_include_directories(dpx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dpx PRIVATE d3d12.lib dxgi.lib synchronization.lib)
