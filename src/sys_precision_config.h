#pragma once
#include <cstdint>

enum class DpxPrecisionMode {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2
};

extern DpxPrecisionMode g_dpx_precision;
