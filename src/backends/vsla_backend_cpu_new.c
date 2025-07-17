/**
 * @file vsla_backend_cpu_new.c
 * @brief New CPU backend implementation following the unified backend interface
 *
 * @copyright MIT License
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

/* CPU Backend Memory Management */
static vsla_error_t cpu_allocate(vsla_tensor_t* tensor) {
    if (!tensor || tensor->data_size == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    /* For CPU backend, cpu_data and data point to the same memory */
    if (!tensor->cpu_data) {
        tensor->cpu_data = calloc(1, tensor->data_size);
        if (!tensor->cpu_data) {
            return VSLA_ERROR_MEMORY;
        }
        tensor->data = tensor->cpu_data;
    }
    
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_deallocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    /* Note: Actual deallocation is handled by vsla_free in vsla_tensor.c */
    tensor->cpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_copy_to_device(vsla_tensor_t* tensor) {
    /* CPU backend: data is already on "device" (CPU) */
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_copy_to_host(vsla_tensor_t* tensor) {
    /* CPU backend: data is already on host */
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_synchronize(void) {
    /* CPU backend: no synchronization needed */
    return VSLA_SUCCESS;
}

/* CPU Backend Arithmetic Operations */
static vsla_error_t cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] - b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] - b_data[i];
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    if (!out || !in) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (in->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t n = vsla_numel(in);
    if (n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (in->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)in->cpu_data;
        double* out_data = (double*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] * scalar;
        }
    } else if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)in->cpu_data;
        float* out_data = (float*)out->cpu_data;
        float fscalar = (float)scalar;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] * fscalar;
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;
        
        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_fill(vsla_tensor_t* tensor, double value) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_SUCCESS;
    }
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = value;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->cpu_data;
        float fvalue = (float)value;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = fvalue;
        }
    }
    
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

/* Reduction operations */
static vsla_error_t cpu_sum(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        *result = 0.0;
        return VSLA_SUCCESS;
    }
    
    *result = 0.0;
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            *result += data[i];
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            *result += (double)data[i];
        }
    }
    
    return VSLA_SUCCESS;
}

static vsla_error_t cpu_mean(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    vsla_error_t err = cpu_sum(tensor, result);
    if (err != VSLA_SUCCESS) {
        return err;
    }
    
    *result /= (double)n;
    return VSLA_SUCCESS;
}

/* Stub implementations for operations not yet implemented */
static vsla_error_t cpu_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    (void)out; (void)tensor;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_reshape(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]) {
    (void)tensor; (void)new_rank; (void)new_shape;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_broadcast(vsla_tensor_t* out, const vsla_tensor_t* in) {
    (void)out; (void)in;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_norm(const vsla_tensor_t* tensor, double* norm) {
    (void)tensor; (void)norm;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_max(const vsla_tensor_t* tensor, double* max) {
    (void)tensor; (void)max;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_min(const vsla_tensor_t* tensor, double* min) {
    (void)tensor; (void)min;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_conv(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_kron(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_init(void* config) {
    (void)config;
    return VSLA_SUCCESS;
}

static void cpu_cleanup(void) {
    /* Nothing to cleanup for CPU backend */
}

/* Backend interface creation */
vsla_backend_interface_t* vsla_backend_cpu_create(void) {
    static vsla_backend_interface_t cpu_backend = {
        .caps = {
            .supports_gpu = false,
            .supports_multi_gpu = false,
            .supports_unified_memory = false,
            .supports_async = false,
            .max_tensor_size = SIZE_MAX,
            .name = "CPU",
            .version = "1.0.0"
        },
        
        /* Memory management */
        .allocate = cpu_allocate,
        .deallocate = cpu_deallocate,
        .copy_to_device = cpu_copy_to_device,
        .copy_to_host = cpu_copy_to_host,
        .synchronize = cpu_synchronize,
        
        /* Basic arithmetic operations */
        .add = cpu_add,
        .sub = cpu_sub,
        .scale = cpu_scale,
        .hadamard = cpu_hadamard,
        .fill = cpu_fill,
        
        /* Linear algebra operations */
        .matmul = cpu_matmul,
        .transpose = cpu_transpose,
        
        /* Tensor operations */
        .reshape = cpu_reshape,
        .broadcast = cpu_broadcast,
        
        /* Reduction operations */
        .sum = cpu_sum,
        .mean = cpu_mean,
        .norm = cpu_norm,
        .max = cpu_max,
        .min = cpu_min,
        
        /* Advanced operations */
        .conv = cpu_conv,
        .kron = cpu_kron,
        
        /* Backend lifecycle */
        .init = cpu_init,
        .cleanup = cpu_cleanup
    };
    
    return &cpu_backend;
}