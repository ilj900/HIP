#pragma once

#include "vector_add.h"

#include <gtest/gtest.h>

TEST(VectorAdd, ReturnsSuccess)
{
    EXPECT_EQ(vector_add(), true);
}