/**
 * @file vsla_unified_minimal.c
 * @brief Minimal implementation of unified interface for testing
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_backend.h"
#include <stdlib.h>
#include <string.h>

/* Simple context structure definition */
struct vsla_context {
    vsla_backend_t active_backend_type;
    vsla_backend_interface_t* active_backend;
    int device_id;
    bool auto_migration;
    bool enable_profiling;
    bool verbose;
};

/* Minimal context management */
vsla_context_t* vsla_init(const vsla_config_t* config) {
    vsla_context_t* ctx = (vsla_context_t*)malloc(sizeof(vsla_context_t));
    if (!ctx) return NULL;
    
    /* Set defaults */
    ctx->active_backend_type = config ? config->backend : VSLA_BACKEND_CPU;
    ctx->device_id = config ? config->device_id : 0;
    ctx->auto_migration = false;
    ctx->enable_profiling = config ? config->enable_profiling : false;
    ctx->verbose = config ? config->verbose : false;
    
    /* For now, just use CPU backend */
    extern vsla_backend_interface_t* vsla_backend_cpu_create(void);
    ctx->active_backend = vsla_backend_cpu_create();
    
    return ctx;
}

void vsla_cleanup(vsla_context_t* ctx) {
    if (ctx) {
        /* Clean up backend if needed */
        if (ctx->active_backend && ctx->active_backend->cleanup) {
            ctx->active_backend->cleanup();
        }
        free(ctx);
    }
}

vsla_error_t vsla_set_backend(vsla_context_t* ctx, vsla_backend_t backend) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    ctx->active_backend_type = backend;
    return VSLA_SUCCESS;
}

vsla_backend_t vsla_get_backend(const vsla_context_t* ctx) {
    return ctx ? ctx->active_backend_type : VSLA_BACKEND_CPU;
}

/* Tensor creation functions */
vsla_tensor_t* vsla_tensor_create(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                  vsla_model_t model, vsla_dtype_t dtype) {
    if (!ctx) return NULL;
    
    /* Use existing tensor creation function */
    extern vsla_tensor_t* vsla_new(uint8_t rank, const uint64_t shape[], 
                                   vsla_model_t model, vsla_dtype_t dtype);
    return vsla_new(rank, shape, model, dtype);
}

vsla_tensor_t* vsla_tensor_zeros(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                 vsla_model_t model, vsla_dtype_t dtype) {
    if (!ctx) return NULL;
    
    extern vsla_tensor_t* vsla_zeros(uint8_t rank, const uint64_t shape[],
                                     vsla_model_t model, vsla_dtype_t dtype);
    return vsla_zeros(rank, shape, model, dtype);
}

vsla_tensor_t* vsla_tensor_ones(vsla_context_t* ctx, uint8_t rank, const uint64_t* shape, 
                                vsla_model_t model, vsla_dtype_t dtype) {
    if (!ctx) return NULL;
    
    extern vsla_tensor_t* vsla_ones(uint8_t rank, const uint64_t shape[],
                                    vsla_model_t model, vsla_dtype_t dtype);
    return vsla_ones(rank, shape, model, dtype);
}

vsla_tensor_t* vsla_tensor_copy(vsla_context_t* ctx, const vsla_tensor_t* tensor) {
    if (!ctx || !tensor) return NULL;
    
    extern vsla_tensor_t* vsla_copy_basic(const vsla_tensor_t* tensor);
    return vsla_copy_basic(tensor);
}

/* Data access functions for unified interface - these take context parameters */
vsla_error_t vsla_get_f64(vsla_context_t* ctx, const vsla_tensor_t* tensor, const uint64_t indices[], double* value) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    
    extern vsla_error_t vsla_get_f64_basic(const vsla_tensor_t* tensor, const uint64_t indices[], double* value);
    return vsla_get_f64_basic(tensor, indices, value);
}

vsla_error_t vsla_set_f64(vsla_context_t* ctx, vsla_tensor_t* tensor, const uint64_t indices[], double value) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    
    extern vsla_error_t vsla_set_f64_basic(vsla_tensor_t* tensor, const uint64_t indices[], double value);
    return vsla_set_f64_basic(tensor, indices, value);
}

vsla_error_t vsla_get_f32(vsla_context_t* ctx, const vsla_tensor_t* tensor, const uint64_t indices[], float* value) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    
    extern vsla_error_t vsla_get_f32_basic(const vsla_tensor_t* tensor, const uint64_t indices[], float* value);
    return vsla_get_f32_basic(tensor, indices, value);
}

vsla_error_t vsla_set_f32(vsla_context_t* ctx, vsla_tensor_t* tensor, const uint64_t indices[], float value) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    
    extern vsla_error_t vsla_set_f32_basic(vsla_tensor_t* tensor, const uint64_t indices[], float value);
    return vsla_set_f32_basic(tensor, indices, value);
}

vsla_error_t vsla_fill(vsla_context_t* ctx, vsla_tensor_t* tensor, double value) {
    if (!ctx) return VSLA_ERROR_NULL_POINTER;
    
    extern vsla_error_t vsla_fill_basic(vsla_tensor_t* tensor, double value);
    return vsla_fill_basic(tensor, value);
}

/* Basic operations using backend */
vsla_error_t vsla_add(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!ctx || !ctx->active_backend) return VSLA_ERROR_NULL_POINTER;
    
    if (ctx->active_backend->add) {
        return ctx->active_backend->add(out, a, b);
    }
    
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_sub(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!ctx || !ctx->active_backend) return VSLA_ERROR_NULL_POINTER;
    
    if (ctx->active_backend->sub) {
        return ctx->active_backend->sub(out, a, b);
    }
    
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_scale(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    if (!ctx || !ctx->active_backend) return VSLA_ERROR_NULL_POINTER;
    
    if (ctx->active_backend->scale) {
        return ctx->active_backend->scale(out, in, scalar);
    }
    
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_hadamard(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!ctx || !ctx->active_backend) return VSLA_ERROR_NULL_POINTER;
    
    if (ctx->active_backend->hadamard) {
        return ctx->active_backend->hadamard(out, a, b);
    }
    
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

/* Placeholder implementations for other functions */
vsla_error_t vsla_matmul(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_transpose(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_conv(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_kron(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_reshape(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* new_shape) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_pad_rank(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t* target_cap) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_sum(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_mean(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_max(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_min(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_norm(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_synchronize(vsla_context_t* ctx) {
    if (!ctx || !ctx->active_backend) return VSLA_ERROR_NULL_POINTER;
    
    if (ctx->active_backend->synchronize) {
        return ctx->active_backend->synchronize();
    }
    
    return VSLA_SUCCESS;
}