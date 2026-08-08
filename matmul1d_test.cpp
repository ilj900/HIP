#include <gtest/gtest.h>

#include "matrix.h"
#include "matmul.h"

TEST(matmul, ReturnsSuccess)
{
    MatMul1D(4, 4, 4, 4);
}

TEST(Sanity, MatGen)
{
    FMatrix A(10, 10);
    A.Print();
}