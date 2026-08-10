#pragma once

#include <memory>

void* HipAlloc(uint32_t SizeInBytes);
void HipFree(void* Data);
void CopyFromDevice(const void* DeviceM, void* HostM, uint32_t Size);
void CopyToDevice(void* DeviceM, const void* HostM, uint32_t Size);

// Helper that allocates some device memory and frees is after it goes out of scope.
template <typename T = float>
struct FDeviceMemory
{
    // Size is the number of element of the template type
    FDeviceMemory(uint32_t Size)
    {
        Data = static_cast<T*>(HipAlloc(Size * sizeof(T)));
    }

    ~FDeviceMemory()
    {
        if (Data)
        {
            HipFree(static_cast<T*>(Data));
        }
    }

    FDeviceMemory(FDeviceMemory&& Other) noexcept
    {
        Data = Other.Data;
        Other.Data = nullptr;
    }

    FDeviceMemory& operator=(FDeviceMemory&& Other) noexcept
    {
        if (Data)
        {
            HipFree(static_cast<T*>(Data));
        }
        Data = Other.Data;
        Other.Data = nullptr;
        return *this;
    }

    FDeviceMemory(const FDeviceMemory &) = delete;
    FDeviceMemory& operator=(const FDeviceMemory&) = delete;

    void FromDevice(void* ToHost, uint32_t SizeB)
    {
        CopyFromDevice(Data, ToHost, SizeB);
    }

    void ToDevice(const void* FromHost, uint32_t SizeB)
    {
        CopyToDevice(Data, FromHost, SizeB);
    }

    T* Data = nullptr;
};

struct FHipTimer
{
    FHipTimer();
    ~FHipTimer();

    FHipTimer(const FHipTimer&) = delete;
    FHipTimer& operator=(const FHipTimer&) = delete;

    void Start();
    float Stop();

    void* StartEvent = nullptr;
    void* StopEvent = nullptr;
};
