#include "hip/hip_runtime.h"

#include <format>
#include <stdexcept>

#include "hip_utils.h"

void HipCheck(uint32_t Status, std::source_location Location)
  {
      if (Status != hipSuccess)
      {
          auto Err = static_cast<hipError_t>(Status);
          throw std::runtime_error(std::format(
              "HIP error {} ({}) at {}:{} in `{}`",
              Status, hipGetErrorString(Err),
              Location.file_name(), Location.line(), Location.function_name()));
      }
  }

void* HipAlloc(uint32_t SizeInBytes)
{
    void* Data = nullptr;
    HipCheck(hipMalloc(&Data, SizeInBytes));
    return Data;
}

void HipFree(void* Data)
{
    HipCheck(hipFree(Data));
}

void CopyFromDevice(const void* DeviceM, void* HostM, uint32_t Size)
{
    HipCheck(hipMemcpy(HostM, DeviceM, Size, hipMemcpyDeviceToHost));
}

void CopyToDevice(void* DeviceM, const void* HostM, uint32_t Size)
{
    HipCheck(hipMemcpy(DeviceM, HostM, Size, hipMemcpyHostToDevice));
}

FHipTimer::FHipTimer()
{
    hipEvent_t Start, Stop;
    HipCheck(hipEventCreate(&Start));
    HipCheck(hipEventCreate(&Stop));
    StartEvent = Start;
    StopEvent  = Stop;
}

FHipTimer::~FHipTimer()
{
    if (StartEvent)
    {
        HipCheck(hipEventDestroy(static_cast<hipEvent_t>(StartEvent)));
    }
    if (StopEvent)
    {
        HipCheck(hipEventDestroy(static_cast<hipEvent_t>(StopEvent)));
    }
}

void FHipTimer::Start()
{
    HipCheck(hipEventRecord(static_cast<hipEvent_t>(StartEvent), 0));
}

float FHipTimer::Stop()
{
    HipCheck(hipEventRecord(static_cast<hipEvent_t>(StopEvent), 0));
    HipCheck(hipEventSynchronize(static_cast<hipEvent_t>(StopEvent)));
    float Ms = 0.f;
    HipCheck(hipEventElapsedTime(&Ms, static_cast<hipEvent_t>(StartEvent), static_cast<hipEvent_t>(StopEvent)));
    return Ms;
}