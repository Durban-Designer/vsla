/**
 * @file vsla_backend_cpu.h
 * @brief CPU backend for VSLA operations.
 *
 * @copyright MIT License
 */

#ifndef VSLA_BACKEND_CPU_H
#define VSLA_BACKEND_CPU_H

#include "vsla/vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

vsla_error_t vsla_cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, double scalar);
vsla_error_t vsla_cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cpu_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cpu_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor);
vsla_error_t vsla_cpu_reshape(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]);
vsla_error_t vsla_cpu_sum(const vsla_tensor_t* tensor, double* sum);
vsla_error_t vsla_cpu_norm(const vsla_tensor_t* tensor, double* norm);
vsla_error_t vsla_cpu_max(const vsla_tensor_t* tensor, double* max);
vsla_error_t vsla_cpu_min(const vsla_tensor_t* tensor, double* min);
vsla_error_t vsla_cpu_fill(vsla_tensor_t* tensor, double value);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_BACKEND_CPU_H */
