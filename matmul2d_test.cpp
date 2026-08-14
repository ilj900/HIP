#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul1d.h"
#include "matmul2d.h"
#include "utils.h"
#include <array>

namespace
{
std::filesystem::path BenchPath = "bench_data";
}

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
    FShape{4096, 4096, 4096},
    FShape{10000, 10000, 10000},
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

constexpr std::array BlockSizes2D =
{
    FTile{256, 1},
    FTile{1, 256},
    FTile{16, 16},
    FTile{32, 32},
    FTile{64, 16},
    FTile{64, 4},
    FTile{8, 8},
    FTile{17, 17},
};

class Matmul2DSanity : public ::testing::TestWithParam<std::tuple<FShape, FTile>>
{
public:
    static constexpr uint32_t BlockSize = 256;
};

class Matmul2DBench : public ::testing::TestWithParam<std::tuple<FShape, FTile>>
{};

std::string GetName2D(const FShape& Shape, const FTile& Tile)
{
    return std::format("Matmul2D_TileSize_{}x{}_Shape_{}x{}x{}", Tile.X, Tile.Y, Shape.M, Shape.N, Shape.K);
}

TEST_P(Matmul2DSanity, Sanity)
{
    auto [Shape, Tile] = GetParam();
    auto& [M, N, K] = Shape;

    std::println("Sanity: {}", GetName2D(Shape, Tile));

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);
    FMatrix D = FMatrix::Zeros(M, N);
    MatMul1D(A, B, C, BlockSize);
    MatMul2D(A, B, D, Tile);
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    std::println();
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST_P(Matmul2DBench, Bench)
{
    auto [Shape, Tile] = GetParam();
    auto& [M, N, K] = Shape;
    auto Name = GetName2D(Shape, Tile);

    std::println("Bench 2D: {}", Name);

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C(M, N, true);

    uint32_t Runs = 50;
    std::vector<float> Values(Runs);
    for (int i = 0; i < Runs; ++i)
    {
        Values[i] = MatMul2D(A, B, C, Tile);
    }

    SaveTimings(Values, BenchPath / ("Bench2D" + Name + ".dat"));
    std::println("{}ms\n", Values.back());
}

INSTANTIATE_TEST_SUITE_P(Matmul, Matmul2DSanity, ::testing::Combine(::testing::ValuesIn(ValidationShapes), ::testing::ValuesIn(BlockSizes2D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, FTile>>& Info)
{
    const auto& S = std::get<0>(Info.param);
    const auto& T = std::get<1>(Info.param);
    return std::format("Sanity_{}", GetName2D(S, T));
});

INSTANTIATE_TEST_SUITE_P(Matmul, Matmul2DBench, ::testing::Combine(::testing::ValuesIn(BenchShapes), ::testing::ValuesIn(BlockSizes2D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, FTile>>& Info)
{
    const auto& S = std::get<0>(Info.param);
    const auto  T = std::get<1>(Info.param);
    return std::format("Bench_{}", GetName2D(S, T));
});
