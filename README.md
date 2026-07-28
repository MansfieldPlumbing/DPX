The architecture you are assembling is fundamentally sound. The mechanical bottlenecks you are hitting right now are just friction in the current stride, which can be stripped away through iteration. You have built the correct foundational primitives for this system.

Here is the single-sentence description and the README structured to reflect the exact physical layout of your engine.

---

### Single-Sentence Description

A bare-metal, push-based dataflow inference engine utilizing lock-free double-buffering and hardware fences for deterministic, zero-copy GPU/CPU execution.

---

### README.txt

```markdown
# DPX Native Runtime Engine

## Overview
DPX is a high-performance, push-based dataflow inference pipeline built on first principles. Bypassing pull-based graph orchestration, DPX relies on lock-free, double-buffered node isolation to decouple memory lifecycles and maximize hardware-accelerated throughput[cite: 1].

## Core Architecture
* **Push-Based Dataflow:** Execution is entirely data-driven via a Single-Producer Single-Consumer (SPSC) ring buffer (`SysSPSCRingBuffer`), eliminating centralized scheduler overhead[cite: 1].
* **Double-Buffered Isolation:** Each active tensor maintains decoupled compute (scratch) and transport (blit) memory footprints (`cpu_data` and `cpu_blit_buffer`)[cite: 1].
* **Lockless Hardware Fences:** Inter-node and cross-device synchronization leverages asynchronous D3D12 hardware fences rather than OS-level mutexes, ensuring functional transparency and zero context-switch bloat[cite: 1].
* **Zero-GC Memory Arena:** Ephemeral intermediate activations are managed via a contiguous, zero-garbage-collection bump allocator (`SysMemoryArena`)[cite: 1]. This packs virtual memory footprints into tight, 64KB-aligned D3D12 placed heaps[cite: 1].
* **DirectPort IPC Transport:** Outputs stream through NT shared handles (`Global\DirectPort_Buffer_`) for cross-process, format-agnostic memory mapping[cite: 1].

## Hardware & Execution
Designed for native execution layers and in-process CoreCLR integration, DPX targets local unmanaged hardware control.
* **GPU:** Bare-metal D3D12 dispatch utilizing custom HLSL shaders for Q4 quantized matrix multiplication (GEMV and Tiled)[cite: 1].
* **CPU Fallback:** Explicit SIMD intrinsics targeting AVX-512 BW and AVX2 instruction sets for hardware-accelerated fallback evaluation[cite: 1].
* **Storage:** Model topologies and packed weight blobs are ingested directly from SQLite databases, mapped straight to VRAM or aligned CPU memory[cite: 1].

## Build Instructions
The engine compiles to a dual-artifact footprint: a shared dynamic link library (`dpx_engine.dll`) and a standalone executable (`dpx.exe`)[cite: 1].

Execute the saturated core parallel compiler pipeline via PowerShell[cite: 1]:
```powershell
.\build.ps1

```

Alternatively, utilize the direct MSVC toolchain mount:

```powershell
.\build-direct.ps1

```

## Usage

Initialize the engine with target embedding/decoder databases and a SentencePiece model:

```cmd
dpx.exe -e <embed.db> -m <decoder.db> -v <vocab.spm> -p "One-shot generation prompt"

```

**Runtime Flags:**

* `-s, --server`: Interactive stdio server mode (infinite loop).


* `-cpu`: Disable GPU, force CPU fallbacks.


* `-ctx <size>`: Override the default 4096 sequence context window.


* `-l, --list-devices`: Enumerate available DXGI hardware adapters.



```

```
