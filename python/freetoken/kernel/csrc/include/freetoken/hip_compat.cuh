#pragma once

// Adapted from Maxritz/FreeToken-ROCm (Apache-2.0), issue #82.
#include <hip/hip_runtime.h>
#include <tuple>

#ifndef __grid_constant__
#define __grid_constant__
#endif

using cudaError_t = ::hipError_t;
using cudaStream_t = ::hipStream_t;
using cudaLaunchConfig_t = ::hipLaunchConfig_t;
using cudaLaunchAttribute = ::hipLaunchAttribute;
inline constexpr auto cudaSuccess = ::hipSuccess;
inline constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize =
    ::hipFuncAttributeMaxDynamicSharedMemorySize;
inline constexpr auto cudaDevAttrUnifiedAddressing =
    ::hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    ::hipDeviceAttributeCanUseHostPointerForRegisteredMem;

inline auto cudaGetErrorString(cudaError_t error) {
  return ::hipGetErrorString(error);
}
inline auto cudaGetLastError() { return ::hipGetLastError(); }
inline auto cudaGetDevice(int *device) { return ::hipGetDevice(device); }
inline auto cudaSetDevice(int device) { return ::hipSetDevice(device); }
inline auto cudaDeviceGetAttribute(int *value, hipDeviceAttribute_t attr, int device) {
  return ::hipDeviceGetAttribute(value, attr, device);
}
inline auto cudaHostGetDevicePointer(void **device, void *host, unsigned flags) {
  return ::hipHostGetDevicePointer(device, host, flags);
}

template <typename F>
inline auto cudaFuncSetAttribute(F *func, hipFuncAttribute attr, int value) {
  return ::hipFuncSetAttribute(reinterpret_cast<const void *>(func), attr, value);
}

template <typename F, typename... Args>
inline auto cudaLaunchKernelEx(const cudaLaunchConfig_t *config, F func,
                              Args &&...args) {
  auto storage = std::make_tuple(args...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    void *params[] = {static_cast<void *>(&std::get<I>(storage))...};
    return ::hipLaunchKernelExC(config, reinterpret_cast<const void *>(func), params);
  }(std::index_sequence_for<Args...>{});
}
