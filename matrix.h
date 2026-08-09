#pragma once

#include "hip_utils.h"
#include <cstdint>
#include <vector>

struct FMatrix
{
    explicit FMatrix(uint32_t MIn, uint32_t KIn, bool Empty);
    FMatrix(uint32_t MIn, uint32_t KIn, uint32_t Seed = 0, float Min = -20, float Max = 20);
    void Print();
    void PrintWolfram();
    bool operator==(const FMatrix& Other) const;
    float AbsDiff(const FMatrix& Other) const;
    float RelDiff(const FMatrix& Other) const;
    uint32_t Size() const;
    uint32_t SizeB() const;
    FDeviceMemory<> ToDevice() const;
    void* data();

    uint32_t M;
    uint32_t K;
    std::vector<float> Data;
};
