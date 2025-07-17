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
 * @brief Create a view (slice) of a tensor
 * 
 * @param tensor Source tensor
 * @param start Start indices for each dimension
 * @param end End indices for each dimension (exclusive)
 * @return New tensor view, or NULL on error
 */
vsla_tensor_t* vsla_slice(const vsla_tensor_t* tensor, const uint64_t start[], 
                          const uint64_t end[]);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_OPS_H */
