#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <c10/core/Device.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime_api.h>

// TaeJun-Ryu
#include <c10/util/custom_logging.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace c10::cuda {

// NB: In the past, we were inconsistent about whether or not this reported
// an error if there were driver problems are not.  Based on experience
// interacting with users, it seems that people basically ~never want this
// function to fail; it should just return zero if things are not working.
// Oblige them.
// It still might log a warning for user first time it's invoked
C10_CUDA_API DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
C10_CUDA_API DeviceIndex device_count_ensure_non_zero();

C10_CUDA_API DeviceIndex current_device();

C10_CUDA_API void set_device(DeviceIndex device);

C10_CUDA_API void device_synchronize();

C10_CUDA_API void warn_or_error_on_sync();

// Raw CUDA device management functions
C10_CUDA_API cudaError_t GetDeviceCount(int* dev_count);

C10_CUDA_API cudaError_t GetDevice(DeviceIndex* device);

C10_CUDA_API cudaError_t SetDevice(DeviceIndex device);

C10_CUDA_API cudaError_t MaybeSetDevice(DeviceIndex device);

C10_CUDA_API DeviceIndex ExchangeDevice(DeviceIndex device);

C10_CUDA_API DeviceIndex MaybeExchangeDevice(DeviceIndex device);

C10_CUDA_API void SetTargetDevice();

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// this is a holder for c10 global state (similar to at GlobalContext)
// currently it's used to store cuda synchronization warning state,
// but can be expanded to hold other related global state, e.g. to
// record stream usage
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

C10_CUDA_API __inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}
// the subsequent functions are defined in the header because for performance
// reasons we want them to be inline
C10_CUDA_API void __inline__ memcpy_and_sync(
    void* dst,
    const void* src,
    int64_t nbytes,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  // TaeJun-Ryu
  // CustomLOG("function called.");

  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        reinterpret_cast<uintptr_t>(stream));
  }
#if defined(TORCH_HIP_VERSION) && (TORCH_HIP_VERSION >= 301)
  C10_CUDA_CHECK(hipMemcpyWithStream(dst, src, nbytes, kind, stream));
#else
  // auto st_cudaMemcpyAsync = std::chrono::high_resolution_clock::now(); // TaeJun-Ryu
  C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
  // auto et_cudaMemcpyAsync = std::chrono::high_resolution_clock::now(); // TaeJun-Ryu
  // auto du_cudaMemcpyAsync = std::chrono::duration_cast<std::chrono::nanoseconds>(et_cudaMemcpyAsync - st_cudaMemcpyAsync).count(); // TaeJun-Ryu
  // CustomLOG("cudaMemcpyAsync | " + std::to_string(du_cudaMemcpyAsync)); // TaeJun-Ryu

  // auto st_cudaStreamSynchronize = std::chrono::high_resolution_clock::now(); // TaeJun-Ryu
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  // auto et_cudaStreamSynchronize = std::chrono::high_resolution_clock::now(); // TaeJun-Ryu
  // auto du_cudaStreamSynchronize = std::chrono::duration_cast<std::chrono::nanoseconds>(et_cudaStreamSynchronize - st_cudaStreamSynchronize).count(); // TaeJun-Ryu
  // CustomLOG("cudaStreamSynchronize | " + std::to_string(du_cudaStreamSynchronize)); // TaeJun-Ryu
#endif
}

C10_CUDA_API void __inline__ stream_synchronize(cudaStream_t stream) {
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        reinterpret_cast<uintptr_t>(stream));
  }

//   CustomLOG("cudaStreamSynchronize | function start"); // TaeJun-Ryu
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
//   CustomLOG("cudaStreamSynchronize | function end"); // TaeJun-Ryu
}

C10_CUDA_API bool hasPrimaryContext(DeviceIndex device_index);
C10_CUDA_API c10::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

} // namespace c10::cuda
