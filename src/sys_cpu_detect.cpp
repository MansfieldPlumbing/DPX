#include "sys_cpu_detect.h"
#include <intrin.h>

CpuCapabilities dpx_detect_cpu() {
    CpuCapabilities cap;
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        cap.avx2     = (cpuInfo[1] & (1 << 5)) != 0;
        cap.avx512f  = (cpuInfo[1] & (1 << 16)) != 0;
        cap.avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
        cap.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
        cap.vnni     = (cpuInfo[2] & (1 << 11)) != 0;
    }
    return cap;
}
