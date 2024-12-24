#include "utils.h"

void set_value_cpu(float* arr, int32_t size, float value) {
  for (int32_t i = 0; i < size; ++i) {
    arr[i] = value;
  }
}
