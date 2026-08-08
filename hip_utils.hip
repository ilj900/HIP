#include "hip/hip_runtime.h"

#include "hip_utils.h"

void* HipAlloc(uint32_t SizeInBytes)
{
    void* Data = nullptr;
    hipMalloc(&Data, SizeInBytes);
    return Data;
}

void HipFree(void* Data)
{
    hipFree(Data);
}

void CopyFromDevice(const void* DeviceM, void* HostM, uint32_t Size)
{
    hipMemcpy(HostM, DeviceM, Size, hipMemcpyDeviceToHost);
}

void CopyToDevice(void* DeviceM, const void* HostM, uint32_t Size)
{
    hipMemcpy(DeviceM, HostM, Size, hipMemcpyHostToDevice);
}
