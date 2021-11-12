
// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// C++
#include <assert.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#define cuda_try(call)                                                                \
  do {                                                                                \
    cudaError_t err = static_cast<cudaError_t>(call);                                 \
    if (err != cudaSuccess) {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

struct gpu_timer {
  gpu_timer(cudaStream_t stream = 0) : start_{}, stop_{}, stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  void start_timer() { cudaEventRecord(start_, stream_); }
  void stop_timer() { cudaEventRecord(stop_, stream_); }
  float get_elapsed_ms() {
    compute_ms();
    return elapsed_time_;
  }

  float get_elapsed_s() {
    compute_ms();
    return elapsed_time_ * 0.001f;
  }
  ~gpu_timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  };

 private:
  void compute_ms() {
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_time_, start_, stop_);
  }
  cudaEvent_t start_, stop_;
  cudaStream_t stream_;
  float elapsed_time_ = 0.0f;
};

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

__global__ void kernel(uint32_t* d_in, uint32_t* d_out, int count) {
  auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < count; i += blockDim.x * gridDim.x) {
    d_out[i] = d_in[i];
  }
}

struct bench_result {
  unsigned requested_sms;
  unsigned acquired_sms;
  unsigned total_sms;
  float time_s;

  bench_result& operator+=(const bench_result& rhs) {
    requested_sms += rhs.requested_sms;
    acquired_sms += rhs.acquired_sms;
    total_sms += rhs.total_sms;
    time_s += rhs.time_s;
    return *this;
  }

  template <typename T>
  bench_result& operator/=(const T& rhs) {
    requested_sms /= rhs;
    acquired_sms /= rhs;
    total_sms /= rhs;
    time_s /= rhs;
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, const bench_result& res) {
  std::cout << "requested_sms = " << res.requested_sms << '\n';
  std::cout << "acquired_sms = " << res.acquired_sms << '\n';
  std::cout << "total_sms = " << res.total_sms << '\n';
  std::cout << "time_s = " << res.time_s << '\n';
  return os;
}

float one_kernel_reference() {
  uint32_t count = 100'000'000;
  uint32_t* d_in;
  uint32_t* d_out;
  std::size_t bytes_count = sizeof(uint32_t) * count;
  cuda_try(cudaMalloc(&d_in, bytes_count));
  cuda_try(cudaMalloc(&d_out, bytes_count));
  cuda_try(cudaMemset(d_in, 0xff, bytes_count));
  cuda_try(cudaMemset(d_out, 0x00, bytes_count));

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 512;

  gpu_timer timer;
  timer.start_timer();
  kernel<<<num_blocks, block_size>>>(d_in, d_out, count);
  timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());

  float elapsed_seconds = timer.get_elapsed_s();
  std::vector<uint32_t> h_out(count);

  cuda_try(cudaMemcpy(h_out.data(), d_out, bytes_count, cudaMemcpyDeviceToHost));

  for (const auto& k : h_out) {
    assert(k == 0xffffffff);
  }

  cuda_try(cudaFree(d_in));
  cuda_try(cudaFree(d_out));

  std::cout << "Ok\n";

  return elapsed_seconds;
}

bench_result one_kernel_test(float load) {
  int device_id = 0;
  int sms_count = 0;
  CUdevice dev = get_cuda_device(device_id, sms_count);

  unsigned load_sms = static_cast<unsigned>(load * sms_count);
  CUcontext ctx;
  CUexecAffinityParam_v1 affinity_param{
      CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, load_sms};
  auto flags = CUctx_flags::CU_CTX_SCHED_AUTO;

  std::cout << "Requesting " << load_sms << " SMs (load = " << load << ")\n";
  cuda_try(cuCtxCreate_v3(&ctx, &affinity_param, 1, flags, dev));

  CUexecAffinityParam acquired_affinity_param;
  cuda_try(
      cuCtxGetExecAffinity(&acquired_affinity_param, CU_EXEC_AFFINITY_TYPE_SM_COUNT));
  unsigned acquired_sms = acquired_affinity_param.param.smCount.val;
  std::cout << "Acquired " << acquired_sms << " SMs\n";

  uint32_t count = 100'000'000;
  uint32_t* d_in;
  uint32_t* d_out;
  std::size_t bytes_count = sizeof(uint32_t) * count;
  cuda_try(cudaMalloc(&d_in, bytes_count));
  cuda_try(cudaMalloc(&d_out, bytes_count));
  cuda_try(cudaMemset(d_in, 0xff, bytes_count));
  cuda_try(cudaMemset(d_out, 0x00, bytes_count));

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 512;

  gpu_timer timer;
  timer.start_timer();
  kernel<<<num_blocks, block_size>>>(d_in, d_out, count);
  timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());

  float elapsed_seconds = timer.get_elapsed_s();
  std::vector<uint32_t> h_out(count);

  cuda_try(cudaMemcpy(h_out.data(), d_out, bytes_count, cudaMemcpyDeviceToHost));

  for (const auto& k : h_out) {
    assert(k == 0xffffffff);
  }

  cuda_try(cudaFree(d_in));
  cuda_try(cudaFree(d_out));
  cuda_try(cuCtxDestroy(ctx));

  std::cout << "Ok\n";

  return {load_sms, acquired_sms, (unsigned)sms_count, elapsed_seconds};
}

void test_valid_sm_configs() {
  int device_id = 0;
  int sms_count = 0;
  CUdevice dev = get_cuda_device(device_id, sms_count);

  std::cout << "requested -> acquired \n";

  for (unsigned load_sms = 1; load_sms <= sms_count; load_sms++) {
    CUcontext ctx;
    CUexecAffinityParam_v1 affinity_param{
        CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, load_sms};
    auto flags = CUctx_flags::CU_CTX_SCHED_AUTO;

    cuda_try(cuCtxCreate_v3(&ctx, &affinity_param, 1, flags, dev));

    CUexecAffinityParam acquired_affinity_param;
    cuda_try(
        cuCtxGetExecAffinity(&acquired_affinity_param, CU_EXEC_AFFINITY_TYPE_SM_COUNT));
    unsigned acquired_sms = acquired_affinity_param.param.smCount.val;

    std::cout << load_sms;
    std::cout << ",";
    std::cout << acquired_sms;
    std::cout << "\n";

    cuda_try(cuCtxDestroy(ctx));
  }
}

template <typename T>
__device__ inline T load_acquire(T* ptr) {
  T old;
  asm volatile("ld.acquire.gpu.b32 %0,[%1];" : "=r"(old) : "l"(ptr) : "memory");
  return old;
}

template <typename T>
__device__ inline void store_release(T* ptr, T value) {
  asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
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
  CUdevice device = get_cuda_device(device_id, sms_count);

  unsigned producer_sms = static_cast<unsigned>(producer_sms_ratio * sms_count);
  unsigned consumer_sms = static_cast<unsigned>(consumer_sms_ratio * sms_count);
  producer_sms = std::max(producer_sms, 1u);
  consumer_sms = std::max(consumer_sms, 1u);

  std::cout << "Producer SMs = " << producer_sms << "/" << sms_count << '\n';
  std::cout << "Consumer SMs = " << consumer_sms << "/" << sms_count << '\n';

  // Setup context and affinity parameters
  CUexecAffinityParam_v1 producer_affinity_param{
      CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, producer_sms};
  CUexecAffinityParam_v1 consumer_affinity_param{
      CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, consumer_sms};

  auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;

  CUcontext producer_ctx, consumer_ctx;
  std::cout << "Creating producer_ctx\n";
  cuda_try(
      cuCtxCreate_v3(&producer_ctx, &producer_affinity_param, 1, affinity_flags, device));
  std::cout << "Creating consumer_ctx\n";
  cuda_try(
      cuCtxCreate_v3(&consumer_ctx, &consumer_affinity_param, 1, affinity_flags, device));

  // Allocate memory
  uint32_t* data;
  uint32_t* flags;
  std::size_t bytes_count = sizeof(uint32_t) * count;
  cuda_try(cudaMalloc(&data, bytes_count));
  cuda_try(cudaMalloc(&flags, bytes_count));
  cuda_try(cudaMemset(data, 0x00, bytes_count));
  cuda_try(cudaMemset(flags, 0x00, bytes_count));
  success_ = true;
  cuda_try(cudaDeviceSynchronize());
  std::cout << "Launching kernels\n";
  // launch kernels
  const uint32_t block_size = 128;
  const uint32_t num_blocks = 512;

  // producer kernel
  std::thread producer_thread{[&] {
    cuCtxSetCurrent(producer_ctx);
    producer_kernel<<<num_blocks, block_size>>>(data, flags, count, 512);
  }};

  // consumer kernel
  std::thread consumer_thread{[&] {
    cuCtxSetCurrent(consumer_ctx);
    consumer_kernel<<<num_blocks, block_size>>>(data, flags, count, 512);
  }};
  producer_thread.join();
  consumer_thread.join();

  cuda_try(cudaDeviceSynchronize());
  if (success_) {
    std::cout << "Ok.\n";
  } else {
    std::cout << "Error\n";
  }
  cuda_try(cudaFree(data));
  cuda_try(cudaFree(flags));

  cuda_try(cuCtxDestroy(producer_ctx));
  cuda_try(cuCtxDestroy(consumer_ctx));
}

void bench(int argc, char** argv) {
  int num_experiments = (argc >= 2) ? std::atoi(argv[1]) : 10;
  std::vector<float> loads_vector{
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

  std::fstream output("bench.csv", std::ios::out);

  float refence_time = one_kernel_reference();
  std::cout << "refence_time = " << refence_time << std::endl;
  std::vector<bench_result> load_results(loads_vector.size(), {0, 0, 0, 0});

  for (std::size_t load_index = 0; load_index < loads_vector.size(); load_index++) {
    std::cout << "Benchmarking load = " << loads_vector[load_index] << '\n';
    for (int i = 0; i < num_experiments; i++) {
      auto res = one_kernel_test(loads_vector[load_index]);
      load_results[load_index] += res;
    }
    std::cout << load_results[load_index];
    load_results[load_index] /= num_experiments;
    std::cout << load_results[load_index];
  }

  output << "acquired_sms, requested_sms, time_seconds, pct,\n";
  for (std::size_t load_index = 0; load_index < loads_vector.size(); load_index++) {
    output << load_results[load_index].requested_sms << ',';
    output << load_results[load_index].acquired_sms << ',';
    output << load_results[load_index].time_s << ',';
    output << refence_time / load_results[load_index].time_s * 100 << ',';
    output << '\n';
  }
}
int main(int argc, char** argv) {
  // one_kernel_test(argc, argv);
  // test_producer_consumer(argc, argv);
  // bench(argc, argv);
  test_valid_sm_configs();
}
