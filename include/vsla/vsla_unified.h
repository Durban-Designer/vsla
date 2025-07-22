/**
 * @file vsla_unified.h
 * @brief Hardware-agnostic unified interface for VSLA operations
 * 
 * @copyright MIT License
 */

#ifndef VSLA_UNIFIED_H
#define VSLA_UNIFIED_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include "vsla_backend.h"
#include "vsla_context.h"
#include <stdbool.h>
#include <stddef.h>

/* vsla_context_t is defined in vsla_tensor.h */

typedef enum {
    VSLA_HINT_NONE = 0,
    VSLA_HINT_LATENCY = 1,
    VSLA_HINT_THROUGHPUT = 2,
    VSLA_HINT_MEMORY = 3,
    VSLA_HINT_ENERGY = 4
} vsla_hint_t;

typedef struct {
    vsla_backend_t backend;
    int device_id;
    size_t memory_limit;
    vsla_hint_t optimization_hint;
    bool enable_profiling;
    bool verbose;
} vsla_config_t;

/* Context management */
vsla_context_t* vsla_init(const vsla_config_t* config);
void vsla_cleanup(vsla_context_t* ctx);
vsla_error_t vsla_set_backend(vsla_context_t* ctx, vsla_backend_t backend);
vsla_backend_t vsla_get_backend(const vsla_context_t* ctx);

/* Tensor creation and destruction */
vsla_tensor_t* vsla_tensor_create(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                  vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_tensor_zeros(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                 vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_tensor_ones(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_tensor_copy(vsla_context_t* ctx, const vsla_tensor_t* tensor);
vsla_tensor_t* vsla_zero_element(vsla_context_t* ctx, vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* vsla_one_element(vsla_context_t* ctx, vsla_model_t model, vsla_dtype_t dtype);
void vsla_tensor_free(vsla_tensor_t* tensor);

/* Data access */
vsla_error_t vsla_get_f64(vsla_context_t* ctx, const vsla_tensor_t* tensor, const uint64_t indices[], double* value);
vsla_error_t vsla_set_f64(vsla_context_t* ctx, vsla_tensor_t* tensor, const uint64_t indices[], double value);
vsla_error_t vsla_get_f32(vsla_context_t* ctx, const vsla_tensor_t* tensor, const uint64_t indices[], float* value);
vsla_error_t vsla_set_f32(vsla_context_t* ctx, vsla_tensor_t* tensor, const uint64_t indices[], float value);
vsla_error_t vsla_fill(vsla_context_t* ctx, vsla_tensor_t* tensor, double value);
const void* vsla_tensor_data(const vsla_tensor_t* tensor, size_t* size);
void* vsla_tensor_data_mut(vsla_tensor_t* tensor, size_t* size);

/* Basic operations */
vsla_error_t vsla_add(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_sub(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_scale(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t vsla_hadamard(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

/* Linear algebra operations */
vsla_error_t vsla_matmul(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_transpose(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in);

/* Model-specific operations */
vsla_error_t vsla_conv(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_kron(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

/* Shape operations */
vsla_error_t vsla_reshape(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* new_shape);
vsla_error_t vsla_pad_rank(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* target_cap);

/* Reduction operations */
vsla_error_t vsla_sum(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_mean(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_max(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_min(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_norm(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);

/* Structural operations */
vsla_error_t vsla_stack(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* const* tensors, size_t k);
vsla_error_t vsla_shrink(vsla_context_t* ctx, vsla_tensor_t* tensor);

/* Window stacking structures and functions */
typedef struct vsla_window_s vsla_window_t;
vsla_window_t* vsla_window_create(vsla_context_t* ctx, size_t window_size, uint8_t rank, vsla_dtype_t dtype);
void vsla_window_destroy(vsla_window_t* window);
vsla_tensor_t* vsla_window_push(vsla_window_t* window, vsla_tensor_t* tensor);

/* Synchronization */
vsla_error_t vsla_synchronize(vsla_context_t* ctx);

#endif /* VSLA_UNIFIED_H */
