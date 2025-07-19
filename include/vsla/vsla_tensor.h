/**
 * @file vsla_tensor.h
 * @brief Opaque tensor handle definition and basic property accessors
 * 
 * This header only defines the opaque tensor type and read-only property
 * accessors. All tensor operations must go through the unified interface.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_TENSOR_H
#define VSLA_TENSOR_H

#include "vsla_core.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a VSLA tensor.
 *
 * The internal structure is not exposed to ensure ABI stability.
 * All operations on tensors must go through the unified interface.
 */
typedef struct vsla_tensor vsla_tensor_t;

/**
 * @brief Opaque handle to a VSLA context.
 */
typedef struct vsla_context vsla_context_t;

/* Read-only property accessors */

/**
 * @brief Get the rank (number of dimensions) of a tensor
 * 
 * @param tensor Input tensor
 * @return Rank of the tensor, or 0 if tensor is NULL
 */
uint8_t vsla_get_rank(const vsla_tensor_t* tensor);

/**
 * @brief Get the shape of a tensor
 * 
 * @param tensor Input tensor
 * @param shape Output array to fill with shape (must have space for rank elements)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_get_shape(const vsla_tensor_t* tensor, uint64_t* shape);

/**
 * @brief Get the model type of a tensor
 * 
 * @param tensor Input tensor
 * @return Model type, or VSLA_MODEL_NONE if tensor is NULL
 */
vsla_model_t vsla_get_model(const vsla_tensor_t* tensor);

/**
 * @brief Get the data type of a tensor
 * 
 * @param tensor Input tensor
 * @return Data type, or VSLA_DTYPE_NONE if tensor is NULL
 */
vsla_dtype_t vsla_get_dtype(const vsla_tensor_t* tensor);

/**
 * @brief Get the total number of elements in the tensor
 * 
 * @param tensor Input tensor
 * @return Number of elements, or 0 if tensor is NULL
 */
uint64_t vsla_numel(const vsla_tensor_t* tensor);

/**
 * @brief Check if two tensors have the same shape
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return 1 if shapes are equal, 0 otherwise
 */
int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * @brief Get the backend location of a tensor
 * 
 * @param tensor Input tensor
 * @return Backend location
 */
vsla_backend_t vsla_get_location(const vsla_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_TENSOR_H */