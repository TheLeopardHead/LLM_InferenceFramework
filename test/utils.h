#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cstdint>
#include <glog/logging.h>
#include <gtest/gtest.h>

void set_value_cpu(float* arr, int32_t size, float value = 1.f);
#endif  // TEST_UTILS_H
