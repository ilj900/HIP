#pragma once

#include <filesystem>

#include "matrix.h"

struct FShape
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct FTile
{
    uint32_t X;
    uint32_t Y;
};

struct F1DBlockTilingConfig
{
    uint32_t BM;
    uint32_t BN;
    uint32_t BK;
    uint32_t TM;

    auto operator<=>(const F1DBlockTilingConfig&) const = default;
};

FMatrix CPUMatMul(const FMatrix& A, const FMatrix& B);

void SaveTimings(const std::vector<float>& Timings, const std::filesystem::path& FilePath);