#pragma once

#include <filesystem>

#include "matrix.h"

struct FShape
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

FMatrix CPUMatMul(const FMatrix& A, const FMatrix& B);

void SaveTimings(const std::vector<float>& Timings, const std::filesystem::path& FilePath);