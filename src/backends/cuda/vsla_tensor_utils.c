/**
 * @file vsla_tensor_utils.c
 * @brief Implementation of tensor utility functions
 * @copyright MIT License
 */

#include "vsla/vsla_tensor_utils.h"
#include "vsla/internal/vsla_tensor_internal.h"

void* vsla_tensor_get_gpu_data(const vsla_tensor_t* tensor) {
    if (!tensor) {
        return NULL;
    }
    return tensor->gpu_data;
}
