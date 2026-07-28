#pragma once

struct CpuCapabilities {
    bool avx2 = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool vnni = false;
};

CpuCapabilities dpx_detect_cpu();
