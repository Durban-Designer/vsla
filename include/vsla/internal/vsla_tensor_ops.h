/**
 * @file vsla_tensor_ops.h
 * @brief Internal tensor operations API (not for public use)
 * 
 * This header contains direct tensor operations that bypass the unified
 * interface. These are used internally by the library but should not be
 * exposed to end users.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_INTERNAL_TENSOR_OPS_H
#define VSLA_INTERNAL_TENSOR_OPS_H

#include "../vsla_core.h"
#include "../vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct tensor operations (internal use only) */
vsla_error_t vsla_add_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_sub_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_scale_direct(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t vsla_hadamard_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_matmul_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_transpose_direct(vsla_tensor_t* out, const vsla_tensor_t* in);
vsla_error_t vsla_conv_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_kron_direct(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

/* Direct reduction operations */
vsla_error_t vsla_sum_direct(const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_mean_direct(const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_norm_direct(const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_max_direct(const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_min_direct(const vsla_tensor_t* tensor, double* result);

/* Direct shape operations */
vsla_error_t vsla_reshape_direct(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* new_shape);
vsla_error_t vsla_pad_rank_direct(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* target_cap);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_INTERNAL_TENSOR_OPS_H */