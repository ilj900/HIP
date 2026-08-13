#include <print>
#include <random>
#include <cassert>

#include "matrix.h"

FMatrix FMatrix::Zeros(uint32_t M, uint32_t K)
{
    return FMatrix(M, K, FZeroTag{});
}

FMatrix::FMatrix(uint32_t MIn, uint32_t KIn,  uint32_t Seed, float Min, float Max)
    : M (MIn), K (KIn), Data(M * K)
{
    std::mt19937 Engine{Seed};
    std::uniform_real_distribution Dist(Min, Max);

    for (auto & Entry : Data)
    {
        Entry = Dist(Engine);
    }
}

void FMatrix::Print() const
{
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            std::print("{:10.3f}", Data[i * K + j]);
        }
        std::println();
    }
}

void FMatrix::PrintWolfram() const
{
    std::print("{{");
    for (uint32_t i = 0; i < M; ++i)
    {
        if (i > 0)
        {
            std::print(", ");
        }
        std::print("{{");
        for (uint32_t j = 0; j < K; ++j)
        {
            if (j > 0)
            {
                std::print(", ");
            }
            std::print("{:.9g}", Data[i * K + j]);
        }
        std::print("}}");
    }
    std::println("}}");
}

bool FMatrix::operator==(const FMatrix& Other) const
{
    return M == Other.M && K == Other.K && Data == Other.Data;
}

float FMatrix::AbsDiff(const FMatrix& Other) const
{
    assert(M == Other.M && K == Other.K);
    float Diff = 0.f;

    for (size_t i = 0; i < Data.size(); ++i)
        Diff += std::abs(Other.Data[i] - Data[i]);

    return Diff;
}

float FMatrix::RelDiff(const FMatrix& Other) const
{
    assert(M == Other.M && K == Other.K);

    float Diff = 0.f;
    float Sum = 0.f;

    for (uint32_t i = 0; i < Data.size(); ++i)
    {
        Diff += std::abs(Other.Data[i] - Data[i]);
        Sum += std::abs(Data[i]);
    }

    return Sum > 0.f ? Diff / Sum : Diff;
}

uint32_t FMatrix::Size() const
{
    return Data.size();
}

uint32_t FMatrix::SizeB() const
{
    return Data.size() * sizeof(float);
}

FDeviceMemory<> FMatrix::ToDevice() const
{
    FDeviceMemory Memory(Data.size());
    Memory.ToDevice(Data.data(), SizeB());
    return Memory;
}

FDeviceMemory<> FMatrix::AllocateDevice() const
{
    FDeviceMemory Memory(Data.size());
    return Memory;
}

void* FMatrix::GetData()
{
    return Data.data();
}

