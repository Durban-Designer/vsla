/**
 * @file vsla_tensor_create.h
 * @brief Internal tensor creation API (not for public use)
 * 
 * This header contains direct tensor creation functions that bypass the
 * unified interface. These are used internally by the library.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_INTERNAL_TENSOR_CREATE_H
#define VSLA_INTERNAL_TENSOR_CREATE_H

#include "../vsla_core.h"
#include "../vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct tensor creation (internal use only) */
vsla_tensor_t* vsla_new_direct(uint8_t rank, const uint64_t shape[], 
                               vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_zeros_direct(uint8_t rank, const uint64_t shape[],
                                 vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_ones_direct(uint8_t rank, const uint64_t shape[],
                                vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_copy_direct(const vsla_tensor_t* tensor);
vsla_tensor_t* vsla_zero_element_direct(vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_one_element_direct(vsla_model_t model, vsla_dtype_t dtype);

/* Direct tensor data access */
vsla_error_t vsla_get_f64_direct(const vsla_tensor_t* tensor, const uint64_t indices[], double* value);
vsla_error_t vsla_set_f64_direct(vsla_tensor_t* tensor, const uint64_t indices[], double value);
vsla_error_t vsla_get_f32_direct(const vsla_tensor_t* tensor, const uint64_t indices[], float* value);
vsla_error_t vsla_set_f32_direct(vsla_tensor_t* tensor, const uint64_t indices[], float value);
vsla_error_t vsla_fill_direct(vsla_tensor_t* tensor, double value);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_INTERNAL_TENSOR_CREATE_H */