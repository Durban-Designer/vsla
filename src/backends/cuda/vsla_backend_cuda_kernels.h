/**
 * @file vsla_backend_cuda_kernels.h
 * @brief C-callable wrappers for CUDA kernels
 * @copyright MIT License
 */

#ifndef VSLA_BACKEND_CUDA_KERNELS_H
#define VSLA_BACKEND_CUDA_KERNELS_H

#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"

#ifdef __cplusplus
extern "C" {
#endif

vsla_error_t vsla_cuda_kernel_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cuda_kernel_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cuda_kernel_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t vsla_cuda_kernel_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cuda_kernel_fill(vsla_tensor_t* tensor, double value);
vsla_error_t vsla_cuda_kernel_sum(const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_cuda_kernel_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t vsla_cuda_kernel_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_cuda_kernel_fill(vsla_tensor_t* tensor, double value);
vsla_error_t vsla_cuda_kernel_sum(const vsla_tensor_t* tensor, double* result);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_BACKEND_CUDA_KERNELS_H */
