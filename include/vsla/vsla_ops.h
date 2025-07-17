/**
 * @file vsla_ops.h
 * @brief Basic operations on VSLA tensors
 * 
 * @copyright MIT License
 */

#ifndef VSLA_OPS_H
#define VSLA_OPS_H

#include "vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Expand the rank of a tensor by adding new dimensions
 * 
 * This is a zero-copy operation that increases the rank of a tensor by
 * appending new dimensions. The original data is preserved, and the new
 * dimensions are implicitly zero-padded.
 * 
 * @param tensor Input tensor
 * @param new_rank New rank (must be >= current rank)
 * @param target_cap Array of target capacities for new dimensions (can be NULL)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_pad_rank(vsla_tensor_t* tensor, uint8_t new_rank, 
                           const uint64_t target_cap[]);

/**
 * @brief Add two tensors element-wise
 * 
 * Performs element-wise addition after automatic padding to compatible shapes.
 * The output tensor must be pre-allocated with sufficient capacity.
 * 
 * @param out Output tensor (pre-allocated)
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_add(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b);

/**
 * @brief Subtract two tensors element-wise
 * 
 * @param out Output tensor (pre-allocated)
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_sub(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b);

/**
 * @brief Scale a tensor by a scalar
 * 
 * @param out Output tensor (can be same as input for in-place operation)
 * @param tensor Input tensor
 * @param scalar Scalar multiplier
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, 
                        double scalar);

/**
 * @brief Element-wise multiplication (Hadamard product)
 * 
 * @param out Output tensor (pre-allocated)
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, 
                           const vsla_tensor_t* b);

/**
 * @brief Matrix multiplication for 2D tensors
 * 
 * @param out Output tensor (pre-allocated)
 * @param a First matrix
 * @param b Second matrix
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, 
                         const vsla_tensor_t* b);

/**
 * @brief Transpose a 2D tensor (matrix)
 * 
 * @param out Output tensor (pre-allocated)
 * @param tensor Input tensor (must be rank 2)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor);

/**
 * @brief Reshape a tensor (must preserve total number of elements)
 * 
 * @param tensor Tensor to reshape
 * @param new_rank New rank
 * @param new_shape New shape array
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_reshape(vsla_tensor_t* tensor, uint8_t new_rank, 
                         const uint64_t new_shape[]);

/**
 * @brief Create a view (slice) of a tensor
 * 
 * @param tensor Source tensor
 * @param start Start indices for each dimension
 * @param end End indices for each dimension (exclusive)
 * @return New tensor view, or NULL on error
 */
vsla_tensor_t* vsla_slice(const vsla_tensor_t* tensor, const uint64_t start[], 
                          const uint64_t end[]);

/**
 * @brief Compute the Frobenius norm of a tensor
 * 
 * @param tensor Input tensor
 * @param norm Output norm value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_norm(const vsla_tensor_t* tensor, double* norm);

/**
 * @brief Compute the sum of all elements
 * 
 * @param tensor Input tensor
 * @param sum Output sum value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_sum(const vsla_tensor_t* tensor, double* sum);

/**
 * @brief Find the maximum element
 * 
 * @param tensor Input tensor
 * @param max Output maximum value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_max(const vsla_tensor_t* tensor, double* max);

/**
 * @brief Find the minimum element
 * 
 * @param tensor Input tensor
 * @param min Output minimum value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_min(const vsla_tensor_t* tensor, double* min);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_OPS_H */