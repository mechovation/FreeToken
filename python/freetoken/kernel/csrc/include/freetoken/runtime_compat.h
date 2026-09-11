#pragma once

#if defined(FREETOKEN_USE_ROCM)
#include <hip/hip_runtime_api.h>

#define CUDART_CB
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaFreeHost hipHostFree
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaLaunchHostFunc hipLaunchHostFunc
#else
#include <cuda_runtime_api.h>
#endif
