/**
 * @file vsla_tensor_adapter.h
 * @brief Adapter functions for converting between different tensor types in VSLA.
 *
 * This module provides functions to convert between the basic `vsla_tensor_t`,
 * the `vsla_gpu_tensor_t`, and the `vsla_tensor_t`. This is a temporary
 * solution to the tensor type compatibility crisis, allowing the unified API
 * to work with the basic and GPU-specific operations.
 *
 * @copyright MIT License
 */

#ifndef VSLA_TENSOR_ADAPTER_H
#define VSLA_TENSOR_ADAPTER_H

#include "vsla_tensor.h"
#include "vsla_gpu.h"
#include "vsla_unified.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts a unified tensor to a basic tensor.
 *
 * This function extracts the CPU-specific data from a unified tensor
 * and creates a new basic tensor. The data is not copied.
 *
 * @param unified_tensor The unified tensor to convert.
 * @return A new basic tensor, or NULL on error.
 */
vsla_tensor_t* vsla_unified_to_basic_tensor(vsla_tensor_t* unified_tensor);

/**
 * @brief Converts a basic tensor to a unified tensor.
 *
 * This function creates a new unified tensor from a basic tensor.
 * The data is not copied.
 *
 * @param basic_tensor The basic tensor to convert.
 * @param ctx The VSLA context.
 * @return A new unified tensor, or NULL on error.
 */
vsla_tensor_t* vsla_basic_to_unified_tensor(vsla_tensor_t* basic_tensor, vsla_context_t* ctx);

/**
 * @brief Converts a unified tensor to a GPU tensor.
 *
 * This function extracts the GPU-specific data from a unified tensor
 * and creates a new GPU tensor. The data is not copied.
 *
 * @param unified_tensor The unified tensor to convert.
 * @return A new GPU tensor, or NULL on error.
 */
vsla_gpu_tensor_t* vsla_unified_to_gpu_tensor(vsla_tensor_t* unified_tensor);

/**
 * @brief Converts a GPU tensor to a unified tensor.
 *
 * This function creates a new unified tensor from a GPU tensor.
 * The data is not copied.
 *
 * @param gpu_tensor The GPU tensor to convert.
 * @param ctx The VSLA context.
 * @return A new unified tensor, or NULL on error.
 */
vsla_tensor_t* vsla_gpu_to_unified_tensor(vsla_gpu_tensor_t* gpu_tensor, vsla_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // VSLA_TENSOR_ADAPTER_H
