#pragma once

#include <cstdint>
#include <vector>

struct FMatrix
{
    FMatrix(uint32_t MIn, uint32_t KIn, float Min = -20, float Max = 20, uint32_t Seed = 0);
    void Print();
    bool operator==(const FMatrix&) const;

    uint32_t M;
    uint32_t K;
    std::vector<float> Data;
};
