#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul.h"
#include "utils.h"
#include <array>

std::filesystem::path BenchPath = "bench_data";

// M x K x N
constexpr std::array ValidationShapes =
{
    FShape{1, 1, 1},
    FShape{4, 4, 4},
    FShape(13, 17, 19),
    FShape(251, 251, 251),
    FShape{512, 4, 4},
    FShape{4, 4, 512},
    FShape{4, 512, 4},
    FShape(257, 256, 256),
    FShape(256, 257, 256),
    FShape(256, 256, 257),
    FShape{1, 256, 256},
    FShape{256, 256, 1},
    FShape{256, 1, 256},
    FShape{1, 1, 256},
    FShape{256, 1, 1},
};

constexpr std::array BenchShapes =
{
    FShape{256, 256, 256},
    FShape{1024, 1024, 1024},
    FShape{4096, 4096, 4096},
    FShape{10000, 10000, 10000},
    FShape{8192, 128, 8192},
    FShape{128, 8192, 128},
    FShape{1, 4096, 4096},
    FShape{8, 4096, 4096},
    FShape{32, 4096, 4096},
};

constexpr std::array BlockSizes1D =
{
    64u,
    128u,
    255u,
    256u,
    257u,
    1024u
};

class Matmul1DSanity : public ::testing::TestWithParam<FShape>
{
public:
    static constexpr uint32_t BlockSize = 256;
};

class Matmul1DBench : public ::testing::TestWithParam<std::tuple<FShape, uint32_t>>
{};

std::string GetName1D(const FShape& Shape, uint32_t BlockSize)
{
    return std::format("Matmul1D_GroupSize_{}_Shape_{}x{}x{}", BlockSize, Shape.M, Shape.N, Shape.K);
}

TEST(matmul, SmokeTest)
{
    FMatrix A(13, 17);
    FMatrix B(17, 19, 1u);
    FMatrix C = FMatrix::Zeros(13, 19);   // line 71
    MatMul1D(A, B, C, 16);
    auto D = CPUMatMul(A, B);
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST_P(Matmul1DBench, Bench)
{
    auto [Shape, BlockSize] = GetParam();
    auto& [M, N, K] = Shape;
    auto Name = GetName1D(Shape, BlockSize);

    std::println("Bench 1D: {}", Name);

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C(M, N, true);

    uint32_t Runs = 100;
    std::vector<float> Values(Runs);
    for (int i = 0; i < Runs; ++i)
    {
        Values[i] = MatMul1D(A, B, C, BlockSize);
    }

    SaveTimings(Values, BenchPath / ("Bench1D" + Name + ".dat"));
    std::println("{}ms\n", Values.back());
}

TEST_P(Matmul1DSanity, Sanity)
{
    auto Shape = GetParam();
    auto& [M, N, K] = Shape;

    std::println("Sanity: {}", GetName1D(Shape, BlockSize));

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);
    MatMul1D(A, B, C, BlockSize);
    auto D = CPUMatMul(A, B);
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    std::println();
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

INSTANTIATE_TEST_SUITE_P(, Matmul1DSanity, ::testing::ValuesIn(ValidationShapes),
[](const ::testing::TestParamInfo<FShape>& Info)
{
        const auto& S = Info.param;
        return std::format("Sanity_{}", GetName1D(S, Matmul1DSanity::BlockSize));
});

INSTANTIATE_TEST_SUITE_P(, Matmul1DBench, ::testing::Combine(::testing::ValuesIn(BenchShapes), ::testing::ValuesIn(BlockSizes1D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, uint32_t>>& Info)
{
        const auto& S = std::get<0>(Info.param);
        const auto  B = std::get<1>(Info.param);
        return std::format("Bench_{}", GetName1D(S, B));
});
