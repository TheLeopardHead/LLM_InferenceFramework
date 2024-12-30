#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <memory>
#include <iostream>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "../utils.h"
#include "base/buffer.h"

TEST (test_add_cpu, add1) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;
  int32_t repeat_times = 100;

  // Record total durations
  std::chrono::microseconds total_data_prep_duration(0);
  std::chrono::microseconds total_compute_duration(0);
  std::chrono::microseconds total_copy_duration(0);

  // Perform random tests repeat_times times
  for (int test = 0; test < repeat_times; ++test) {
    // Prepare input and output tensors
    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

    // Record data preparation start time
    auto data_prep_start = std::chrono::high_resolution_clock::now();

    // Generate random data for input tensors
    float* t1_data = static_cast<float*>(t1.get_buffer()->ptr());
    float* t2_data = static_cast<float*>(t2.get_buffer()->ptr());
    auto expected_output = std::make_unique<float[]>(size);  // For storing expected results

    // Use modern C++ random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    for (int i = 0; i < size; ++i) {
      t1_data[i] = dis(gen);
      t2_data[i] = dis(gen);
      expected_output[i] = t1_data[i] + t2_data[i];  // Calculate expected results
    }

    auto output = std::make_unique<float[]>(size);
    auto data_prep_end = std::chrono::high_resolution_clock::now();
    auto data_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_prep_end - data_prep_start);
    total_data_prep_duration += data_prep_duration;

    // Record computation time
    auto compute_start = std::chrono::high_resolution_clock::now();
    kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(t1, t2, out, nullptr);
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
    total_compute_duration += compute_duration;

    // Record data copy time
    auto copy_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
      output[i] = out.index<float>(i);
    }
    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start);
    total_copy_duration += copy_duration;

    // Verify results correctness
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(output[i], expected_output[i], 1e-5f);
    }
  }

  // Output average performance statistics
  LOG(INFO) << "CPU Add Performance (Average over " << repeat_times << " tests):";
  LOG(INFO) << "Average data preparation time: " << total_data_prep_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average computation time: " << total_compute_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average data copy time: " << total_copy_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average total time: " << (total_data_prep_duration + total_compute_duration + total_copy_duration).count() / 1000.0 / repeat_times << " ms";
}


TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;
  int32_t repeat_times = 100;

  // Record total durations
  std::chrono::microseconds total_data_prep_duration(0);
  std::chrono::microseconds total_compute_duration(0);
  std::chrono::microseconds total_copy_duration(0);

  // Perform random tests repeat_times times
  for (int test = 0; test < repeat_times; ++test) {
    // Prepare input and output tensors
    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

    // Record data preparation start time
    auto data_prep_start = std::chrono::high_resolution_clock::now();

    // Generate random data for input tensors with different seeds each time
    generate_random_data_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, test * 12345);
    generate_random_data_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, test * 67890);

    // For result verification
    auto t1_host = std::make_unique<float[]>(size);
    auto t2_host = std::make_unique<float[]>(size);
    auto expected_output = std::make_unique<float[]>(size);  // For storing expected results

    cudaMemcpy(t1_host.get(), t1.get_buffer()->ptr(), size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(t2_host.get(), t2.get_buffer()->ptr(), size * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate expected results
    for (int i = 0; i < size; ++i) {
      expected_output[i] = t1_host[i] + t2_host[i];
    }

    auto output = std::make_unique<float[]>(size);
    auto data_prep_end = std::chrono::high_resolution_clock::now();
    auto data_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_prep_end - data_prep_start);
    total_data_prep_duration += data_prep_duration;

    // Record computation time
    auto compute_start = std::chrono::high_resolution_clock::now();
    kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
    cudaDeviceSynchronize();
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
    total_compute_duration += compute_duration;

    // Record data copy time
    auto copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(output.get(), out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start);
    total_copy_duration += copy_duration;

    // Verify results correctness
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(output[i], expected_output[i], 1e-5f);
    }
  }

  // Output average performance statistics
  LOG(INFO) << "GPU Add Performance (No Stream, Average over " << repeat_times << " tests):";
  LOG(INFO) << "Average data preparation time: " << total_data_prep_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average computation time: " << total_compute_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average data copy time: " << total_copy_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average total time: " << (total_data_prep_duration + total_compute_duration + total_copy_duration).count() / 1000.0 / repeat_times << " ms";
}

TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;
  int32_t repeat_times = 100;

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Record total durations
  std::chrono::microseconds total_data_prep_duration(0);
  std::chrono::microseconds total_compute_duration(0);
  std::chrono::microseconds total_copy_duration(0);

  // Perform random tests repeat_times times
  for (int test = 0; test < repeat_times; ++test) {
    // Prepare input and output tensors
    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

    // Record data preparation start time
    auto data_prep_start = std::chrono::high_resolution_clock::now();

    // Generate random data for input tensors with different seeds each time
    generate_random_data_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, test * 12345);
    generate_random_data_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, test * 67890);

    // For result verification
    auto t1_host = std::make_unique<float[]>(size);
    auto t2_host = std::make_unique<float[]>(size);
    auto expected_output = std::make_unique<float[]>(size);  // For storing expected results

    cudaMemcpyAsync(t1_host.get(), t1.get_buffer()->ptr(), size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(t2_host.get(), t2.get_buffer()->ptr(), size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate expected results
    for (int i = 0; i < size; ++i) {
      expected_output[i] = t1_host[i] + t2_host[i];
    }

    auto output = std::make_unique<float[]>(size);
    auto data_prep_end = std::chrono::high_resolution_clock::now();
    auto data_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_prep_end - data_prep_start);
    total_data_prep_duration += data_prep_duration;

    // Record computation time
    auto compute_start = std::chrono::high_resolution_clock::now();
    kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
    cudaStreamSynchronize(stream);
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
    total_compute_duration += compute_duration;

    // Record data copy time
    auto copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(output.get(), out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start);
    total_copy_duration += copy_duration;

    // Verify results correctness
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(output[i], expected_output[i], 1e-5f);
    }
  }

  // Output average performance statistics
  LOG(INFO) << "GPU Add Performance (With Stream, Average over " << repeat_times << " tests):";
  LOG(INFO) << "Average data preparation time: " << total_data_prep_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average computation time: " << total_compute_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average data copy time: " << total_copy_duration.count() / 1000.0 / repeat_times << " ms";
  LOG(INFO) << "Average total time: " << (total_data_prep_duration + total_compute_duration + total_copy_duration).count() / 1000.0 / repeat_times << " ms";

  cudaStreamDestroy(stream);
}

TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)( t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);
  }

  delete[] output;
}