#include <print>
#include <random>

#include "matrix.h"

FMatrix::FMatrix(uint32_t MIn, uint32_t KIn, bool Empty)
    : M (MIn), K (KIn), Data(M * K)
{}

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

void FMatrix::Print()
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

void FMatrix::PrintWolfram()
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
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            uint32_t Index = i * K + j;
            if (Other.Data[Index] != Data[Index])
            {
                return false;
            }
        }
    }
    return true;
}

float FMatrix::AbsDiff(const FMatrix& Other) const
{
    float Diff = 0.f;
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            uint32_t Index = i * K + j;
            Diff += std::abs(Other.Data[Index] - Data[Index]);
        }
    }

    return Diff;
}

float FMatrix::RelDiff(const FMatrix& Other) const
{
    float Diff = 0.f;
    float Sum = 0.f;
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            uint32_t Index = i * K + j;
            Diff += std::abs(Other.Data[Index] - Data[Index]);
            Sum += std::abs(Data[Index]);
        }
    }

    return Diff / Sum;
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

FDeviceMemory<> FMatrix::AllocateGMem() const
{
    FDeviceMemory Memory(Data.size());
    return Memory;
}

void* FMatrix::data()
{
    return Data.data();
}

