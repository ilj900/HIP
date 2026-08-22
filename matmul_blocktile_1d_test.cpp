#include "matmul_blocktile_1d.h"

#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul2d.h"
#include "matmul_blocktile_1d.h"
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

constexpr std::array BlockTileSizes1D =
{
    F1DBlockTilingConfig{64, 64, 8, 8},
    F1DBlockTilingConfig{64, 64, 8, 16},
    F1DBlockTilingConfig{64, 64, 8, 32},
};

class Matmul1DBlockTilingSanity : public ::testing::TestWithParam<std::tuple<FShape, F1DBlockTilingConfig>>
{
public:
    static constexpr FTile TileSize = {64, 4};
};

class Matmul1DBlockTilingBench : public ::testing::TestWithParam<std::tuple<FShape, F1DBlockTilingConfig>>
{};

std::string GetName1DBlockTiling(const FShape& Shape, const F1DBlockTilingConfig& Params)
{
    return std::format("MatmulBlokTile_BlockDims_{}x{}_{}_TM_{}_Shape_{}x{}x{}",
        Params.BM, Params.BN, Params.BK, Params.TM, Shape.M, Shape.N, Shape.K);
}

TEST_P(Matmul1DBlockTilingSanity, Sanity)
{
    auto [Shape, Tile] = GetParam();
    auto& [M, N, K] = Shape;
    auto& [BM, BN, BK, TM] = Tile;

    std::println("Sanity: {}", GetName1DBlockTiling(Shape, Tile));

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);
    FMatrix D = FMatrix::Zeros(M, N);
    auto Time1 = MatMulBlockTile1D(A, B, C, BM, BN, BK, TM);
    auto Time2 = MatMul2D(A, B, D, TileSize);
    std::println("MatMulBlockTile1D time: {}", Time1);
    std::println("MatMul2D time: {}", Time2);
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    std::println();
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST_P(Matmul1DBlockTilingBench, Bench)
{
    auto [Shape, Tile] = GetParam();
    auto& [M, N, K] = Shape;
    auto& [BM, BN, BK, TM] = Tile;
    auto Name = GetName1DBlockTiling(Shape, Tile);

    std::println("Bench: {}", Name);

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);

    uint32_t Runs = 50;
    std::vector<float> Values(Runs);
    for (int i = 0; i < Runs; ++i)
    {
        Values[i] = MatMulBlockTile1D(A, B, C, BM, BN, BK, TM);
    }

    SaveTimings(Values, BenchPath / ("Bench" + Name + ".dat"));
    std::println("{}ms\n", Values.back());
}

INSTANTIATE_TEST_SUITE_P(Matmul, Matmul1DBlockTilingSanity, ::testing::Combine(::testing::ValuesIn(ValidationShapes), ::testing::ValuesIn(BlockTileSizes1D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, F1DBlockTilingConfig>>& Info)
{
    const auto& S = std::get<0>(Info.param);
    const auto& T = std::get<1>(Info.param);
    return std::format("Sanity_{}", GetName1DBlockTiling(S, T));
});

INSTANTIATE_TEST_SUITE_P(Matmul, Matmul1DBlockTilingBench, ::testing::Combine(::testing::ValuesIn(BenchShapes), ::testing::ValuesIn(BlockTileSizes1D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, F1DBlockTilingConfig>>& Info)
{
    const auto& S = std::get<0>(Info.param);
    const auto  T = std::get<1>(Info.param);
    return std::format("Bench_{}", GetName1DBlockTiling(S, T));
});
