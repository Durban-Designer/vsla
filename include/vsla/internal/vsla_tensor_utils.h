/**
 * @file vsla_tensor_utils.h
 * @brief Utility functions for accessing tensor data
 * @copyright MIT License
 */

#ifndef VSLA_TENSOR_UTILS_H
#define VSLA_TENSOR_UTILS_H

#include "../vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void* vsla_tensor_get_gpu_data(const vsla_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_TENSOR_UTILS_H */
