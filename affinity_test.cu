
// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime_api.h>

// C++
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#define cuda_try(call)                                                                \
  do {                                                                                \
    cudaError_t err = static_cast<cudaError_t>(call);                                 \
    if (err != cudaSuccess) {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

CUdevice get_cuda_device(const int device_id, int& sms_count) {
  CUdevice device;
  int device_count = 0;

  cuda_try(cuInit(0));  // Flag parameter must be zero
  cuda_try(cuDeviceGetCount(&device_count));

  if (device_count == 0) {
    std::cout << "No CUDA capable device found." << std::endl;
    std::terminate();
  }

  cuda_try(cuDeviceGet(&device, device_id));

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);

  sms_count = device_prop.multiProcessorCount;

  std::cout << "Device[" << device_id << "]: " << device_prop.name << '\n';
  std::cout << "SMs count: " << sms_count << '\n';

  return device;
}

int main(int argc, char** argv) {
  int device_id = 0;
  int sms_count = 0;
  CUdevice dev = get_cuda_device(device_id, sms_count);

  float load = 0.5f;
  int load_sms = static_cast<int>(load * sms_count);
  CUcontext ctx;
  CUexecAffinityParam_v1 affinity_param{
      CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, load_sms};
  auto flags = CUctx_flags::CU_CTX_SCHED_AUTO;

  cuda_try(cuCtxCreate_v3(&ctx, &affinity_param, 1, flags, dev));

  cuda_try(cuCtxDestroy(ctx));
}
