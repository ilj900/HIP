#pragma once

#include "matrix.h"

FMatrix CPUMatMul(const FMatrix& A, const FMatrix& B);

void SaveTimings(const std::vector<float>& Timings, const std::string& FileName);