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

FHipTimer::FHipTimer()
{
    hipEvent_t Start, Stop;
    hipEventCreate(&Start);
    hipEventCreate(&Stop);
    StartEvent = Start;
    StopEvent  = Stop;
}

FHipTimer::~FHipTimer()
{
    if (StartEvent) hipEventDestroy(static_cast<hipEvent_t>(StartEvent));
    if (StopEvent)  hipEventDestroy(static_cast<hipEvent_t>(StopEvent));
}

void FHipTimer::Start()
{
    hipEventRecord(static_cast<hipEvent_t>(StartEvent), 0);
}

float FHipTimer::Stop()
{
    hipEventRecord(static_cast<hipEvent_t>(StopEvent), 0);
    hipEventSynchronize(static_cast<hipEvent_t>(StopEvent));
    float Ms = 0.f;
    hipEventElapsedTime(&Ms, static_cast<hipEvent_t>(StartEvent), static_cast<hipEvent_t>(StopEvent));
    return Ms;
}