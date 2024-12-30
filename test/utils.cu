#include <glog/logging.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <memory>
#include "utils.cuh"

__global__ void test_function_cu(float* cu_arr, int32_t size, float value) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  cu_arr[tid] = value;
}

void test_function(float* arr, int32_t size, float value) {
  if (!arr) {
    return;
  }
  float* cu_arr = nullptr;
  cudaError_t err = cudaMalloc(&cu_arr, sizeof(float) * size);
  CHECK_EQ(err, cudaSuccess);  // 检查 cudaMalloc 错误

  test_function_cu<<<1, size>>>(cu_arr, size, value);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);

  err = cudaMemcpy(arr, cu_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_EQ(err, cudaSuccess);  // 检查 cudaMemcpy 错误
  
  cudaFree(cu_arr);
}

void set_value_cu(float* arr_cu, int32_t size, float value) {
  int32_t threads_num = 512;
  int32_t block_num = (size + threads_num - 1) / threads_num;
  
  test_function_cu<<<block_num, threads_num>>>(arr_cu, size, value);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);
}

__global__ void setup_kernel(curandState* states, unsigned int seed, int32_t size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    curand_init(seed, tid, 0, &states[tid]);
  }
}

__global__ void generate_random_data_kernel(float* data, int32_t size, curandState* states) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  
  // Generate random number using cuRAND
  float random = curand_uniform(&states[tid]);
  data[tid] = random * 10.0f; // Generate random number between 0 and 10
}

void generate_random_data_cu(float* arr_cu, int32_t size, unsigned int seed) {
  int32_t threads_num = 512;
  int32_t block_num = (size + threads_num - 1) / threads_num;

  // Allocate curandState for each thread
  curandState* dev_states = nullptr;
  cudaError_t err = cudaMalloc(&dev_states, size * sizeof(curandState));
  CHECK_EQ(err, cudaSuccess);

  // 修改 lambda 的写法，接受一个参数
  auto cleanup = [](curandState* ptr) {
    if (ptr) {
      cudaFree(ptr);
    }
  };
  // 使用 dev_states 作为管理的指针
  std::unique_ptr<curandState, decltype(cleanup)> cleanup_guard(dev_states, cleanup);

  // Initialize random number generator states
  setup_kernel<<<block_num, threads_num>>>(dev_states, seed, size);
  err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);
  
  // Generate random numbers
  generate_random_data_kernel<<<block_num, threads_num>>>(arr_cu, size, dev_states);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);
}
