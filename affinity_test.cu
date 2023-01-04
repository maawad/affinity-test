
// HiP driver & runtime
#include "hip/hip_runtime.h"

// HiP
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// C++
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#define hip_try(call)                                                               \
  do {                                                                              \
    hipError_t err = static_cast<hipError_t>(call);                                 \
    if (err != hipSuccess) {                                                        \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorName(err)); \
      std::terminate();                                                             \
    }                                                                               \
  } while (0)

hipDevice_t get_hip_device(const int device_id, int& sms_count) {
  hipDevice_t device;
  int device_count = 0;

  hip_try(hipInit(0));  // Flag parameter must be zero
  hip_try(hipGetDeviceCount(&device_count));

  if (device_count == 0) {
    std::cout << "No HIP capable device found." << std::endl;
    std::terminate();
  }

  hip_try(hipDeviceGet(&device, device_id));

  hipDeviceProp_t device_prop;
  hipGetDeviceProperties(&device_prop, device_id);

  sms_count = device_prop.multiProcessorCount;

  std::cout << "Device[" << device_id << "]: " << device_prop.name << '\n';
  std::cout << "SMs count: " << sms_count << '\n';

  return device;
}

__global__ void kernel(uint32_t* d_in, uint32_t* d_out, int count) {
  auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < count; i += blockDim.x * gridDim.x) {
    d_out[i] = d_in[i];
  }
}

void one_kernel_test(int argc, char** argv) {
  int device_id = 0;
  int sms_count = 0;
  hipDevice_t dev = get_hip_device(device_id, sms_count);

  float load = 0.5f;
  unsigned load_sms = static_cast<unsigned>(load * sms_count);

  hipCtx_t ctx;
  auto flags = 0;  // FIXME

  std::cout << "Requesting " << load_sms << " SMs (load = " << load << ")\n";
  // FIXME
  /// git/affinity_test.cu:70:11: warning: 'hipCtxCreate' is deprecated: This API is
  /// marked as deprecated and may not be supported in future releases. For more details
  /// please refer
  /// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_deprecated_api_list.md
  /// [-Wdeprecated-declarations]
  hip_try(hipCtxCreate(&ctx, flags, dev));

  uint32_t count = 100'000'000;
  uint32_t* d_in;
  uint32_t* d_out;
  std::size_t bytes_count = sizeof(uint32_t) * count;
  hip_try(hipMalloc(&d_in, bytes_count));
  hip_try(hipMalloc(&d_out, bytes_count));
  hip_try(hipMemset(d_in, 0xff, bytes_count));
  hip_try(hipMemset(d_out, 0x00, bytes_count));

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 512;

  kernel<<<num_blocks, block_size>>>(d_in, d_out, count);
  hip_try(hipDeviceSynchronize());

  std::vector<uint32_t> h_out(count);

  hip_try(hipMemcpy(h_out.data(), d_out, bytes_count, hipMemcpyDeviceToHost));

  for (const auto& k : h_out) {
    assert(k == 0xffffffff);
  }

  hip_try(hipFree(d_in));
  hip_try(hipFree(d_out));
  hip_try(hipCtxDestroy(ctx));

  std::cout << "Ok\n";
}

template <typename T>
__device__ inline T load_acquire(T* ptr) {
  T old = *reinterpret_cast<volatile T*>(ptr);
  __threadfence();
  // asm volatile("ld.acquire.gpu.b32 %0,[%1];" : "=r"(old) : "l"(ptr) : "memory");
  return old;
}

template <typename T>
__device__ inline void store_release(T* ptr, T value) {
  __threadfence();
  *reinterpret_cast<volatile T*>(ptr) = value;
  // asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
}

__managed__ bool success_;

__global__ void producer_kernel(uint32_t* data,
                                uint32_t* flags,
                                uint32_t count,
                                uint32_t produce) {
  auto stride = blockDim.x * gridDim.x;
  auto thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  for (auto tid = thread_index; tid < count; tid += stride) {
    data[tid] = produce;
    uint32_t* line_flag = flags + tid / 32;
    store_release(line_flag, 1u);
  }
}
__global__ void consumer_kernel(uint32_t* data,
                                uint32_t* flags,
                                uint32_t count,
                                uint32_t consume) {
  auto stride = blockDim.x * gridDim.x;
  auto thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  for (auto tid = thread_index; tid < count; tid += stride) {
    uint32_t* line_flag = flags + tid / 32;
    while (!load_acquire(line_flag))
      ;  // spin till flag is set
    auto found = data[tid];
    if (consume != found) {
      success_ = false;
    }
  }
}

void test_producer_consumer(int argc, char** argv) {
  uint32_t count = (argc >= 2) ? std::atoi(argv[1]) : 100'000'000;
  float producer_sms_ratio = (argc >= 3) ? std::atof(argv[2]) : 0.0125f;
  float consumer_sms_ratio = (argc >= 4) ? std::atoi(argv[3]) : 1.0f - producer_sms_ratio;

  int device_id = 0;
  int sms_count = 0;
  hipDevice_t device = get_hip_device(device_id, sms_count);

  unsigned producer_sms = static_cast<unsigned>(producer_sms_ratio * sms_count);
  unsigned consumer_sms = static_cast<unsigned>(consumer_sms_ratio * sms_count);
  producer_sms = std::max(producer_sms, 1u);
  consumer_sms = std::max(consumer_sms, 1u);

  std::cout << "Producer SMs = " << producer_sms << "/" << sms_count << '\n';
  std::cout << "Consumer SMs = " << consumer_sms << "/" << sms_count << '\n';

  // Setup context and affinity parameters
  // CUexecAffinityParam_v1 producer_affinity_param{
  //     CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, producer_sms};
  // CUexecAffinityParam_v1 consumer_affinity_param{
  //     CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, consumer_sms};

  auto affinity_flags = 0;

  hipCtx_t producer_ctx, consumer_ctx;
  std::cout << "Creating producer_ctx\n";
  hip_try(hipCtxCreate(&producer_ctx, affinity_flags, device));
  std::cout << "Creating consumer_ctx\n";
  hip_try(hipCtxCreate(&consumer_ctx, affinity_flags, device));

  // Allocate memory
  uint32_t* data;
  uint32_t* flags;
  std::size_t bytes_count = sizeof(uint32_t) * count;
  hip_try(hipMalloc(&data, bytes_count));
  hip_try(hipMalloc(&flags, bytes_count));
  hip_try(hipMemset(data, 0x00, bytes_count));
  hip_try(hipMemset(flags, 0x00, bytes_count));
  success_ = true;
  hip_try(hipDeviceSynchronize());
  std::cout << "Launching kernels\n";
  // launch kernels
  const uint32_t block_size = 128;
  const uint32_t num_blocks = 512;

  // producer kernel
  std::thread producer_thread{[&] {
    hipCtxSetCurrent(producer_ctx);
    producer_kernel<<<num_blocks, block_size>>>(data, flags, count, 512);
  }};

  // consumer kernel
  std::thread consumer_thread{[&] {
    hipCtxSetCurrent(consumer_ctx);
    consumer_kernel<<<num_blocks, block_size>>>(data, flags, count, 512);
  }};
  producer_thread.join();
  consumer_thread.join();

  hip_try(hipDeviceSynchronize());
  if (success_) {
    std::cout << "Ok.\n";
  } else {
    std::cout << "Error\n";
  }
  hip_try(hipFree(data));
  hip_try(hipFree(flags));

  hip_try(hipCtxDestroy(producer_ctx));
  hip_try(hipCtxDestroy(consumer_ctx));
}

int main(int argc, char** argv) {
  // one_kernel_test(argc, argv);
  test_producer_consumer(argc, argv);
}
