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

vsla_context_t* vsla_init(const vsla_config_t* config);
void vsla_cleanup(vsla_context_t* ctx);
vsla_error_t vsla_add(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_sub(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_scale(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t vsla_hadamard(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_matmul(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t vsla_transpose(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in);
vsla_error_t vsla_reshape(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* new_shape);
vsla_error_t vsla_sum(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_mean(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_max(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_min(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_error_t vsla_norm(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result);
vsla_tensor_t* vsla_tensor_create(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, vsla_model_t model, vsla_dtype_t dtype);
void vsla_tensor_free(vsla_tensor_t* tensor);
const void* vsla_tensor_data(const vsla_tensor_t* tensor, size_t* size);
void* vsla_tensor_data_mut(vsla_tensor_t* tensor, size_t* size);

#endif /* VSLA_UNIFIED_H */
