#pragma once
#include <cstdint>
#include <vector>

int sys_sample_argmax(const float* logits, const std::vector<int>& shape);
