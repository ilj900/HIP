#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <format>
#include <source_location>
#include <stdexcept>

#include "hip_utils.h"
#include "matmul_rocblas.h"

namespace
{
void RocblasCheck(rocblas_status Status,
                  std::source_location Location = std::source_location::current())
{
    if (Status != rocblas_status_success)
    {
        throw std::runtime_error(std::format(
            "rocBLAS error {} ({}) at {}:{} in `{}`",
            static_cast<int>(Status), rocblas_status_to_string(Status),
            Location.file_name(), Location.line(), Location.function_name()));
    }
}

// Handle creation is expensive, so keep one for the whole process and out of
// the timed path. Reused across every MatMulRocblas call.
rocblas_handle GetHandle()
{
    static rocblas_handle Handle = []
    {
        rocblas_handle H = nullptr;
        RocblasCheck(rocblas_create_handle(&H));
        return H;
    }();
    return Handle;
}
}

float MatMulRocblas(const FMatrix& A, const FMatrix& B, FMatrix& C)
{
    auto HipA = A.ToDevice();
    auto HipB = B.ToDevice();
    auto HipC = C.AllocateDevice();

    uint32_t M = C.M;   // rows of C (row-major)
    uint32_t N = C.K;   // cols of C
    uint32_t K = A.K;   // shared dim

    const float Alpha = 1.f;
    const float Beta  = 0.f;

    rocblas_handle Handle = GetHandle();

    FHipTimer Timer;
    Timer.Start();

    // rocBLAS is column-major; our buffers are row-major. A row-major MxK buffer
    // *is* a column-major KxM buffer (its transpose), so row-major C = A*B equals
    // the column-major Cᵀ = Bᵀ*Aᵀ. Concretely: pass B and A swapped, no transpose.
    //   op(A_blas) is m×k, op(B_blas) is k×n, C_blas is m×n, all column-major.
    RocblasCheck(rocblas_sgemm(Handle,
        rocblas_operation_none, rocblas_operation_none,
        N, M, K,          // m, n, k of the column-major problem
        &Alpha,
        HipB.Data, N,     // A_blas = B  (col-major NxK), lda = N
        HipA.Data, K,     // B_blas = A  (col-major KxM), ldb = K
        &Beta,
        HipC.Data, N));   // C_blas = C  (col-major NxM), ldc = N

    float Ms = Timer.Stop();

    HipC.FromDevice(C.GetData(), C.SizeB());
    return Ms;
}