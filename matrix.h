#pragma once

#include "hip_utils.h"
#include <cstdint>
#include <vector>

struct FMatrix
{
    static FMatrix Zeros(uint32_t M, uint32_t K);

    FMatrix(uint32_t MIn, uint32_t KIn, uint32_t Seed = 0, float Min = -20, float Max = 20);
    void Print() const;
    void PrintWolfram() const;
    bool operator==(const FMatrix& Other) const;
    float AbsDiff(const FMatrix& Other) const;
    float RelDiff(const FMatrix& Other) const;
    uint32_t Size() const;
    uint32_t SizeB() const;
    FDeviceMemory<> ToDevice() const;
    FDeviceMemory<> AllocateDevice() const;
    void* GetData();

    uint32_t M;
    uint32_t K;
    std::vector<float> Data;

private:
    struct FZeroTag {};
    FMatrix(uint32_t MIn, uint32_t KIn, FZeroTag) : M(MIn), K(KIn), Data(M * K) {}
};
