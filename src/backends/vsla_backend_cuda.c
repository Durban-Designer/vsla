/**
 * @file vsla_backend_cuda.c
 * @brief CUDA GPU backend implementation with single-kernel operations
 *
 * @copyright MIT License
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/vsla_tensor.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include "vsla/vsla_core.h"
#include "cuda/vsla_backend_cuda_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>

/* CUDA Backend Memory Management */
static vsla_error_t cuda_allocate(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    size_t capacity_bytes = vsla_dtype_size(tensor->dtype);
    for (uint8_t i = 0; i < tensor->rank; ++i) {
        capacity_bytes *= tensor->cap[i];
    }
    if (capacity_bytes == 0) {
        tensor->cpu_data = NULL;
        tensor->gpu_data = NULL;
        return VSLA_SUCCESS;
    }
    // Allocate both CPU and GPU memory
    tensor->cpu_data = malloc(capacity_bytes);
    if (!tensor->cpu_data) {
        return VSLA_ERROR_MEMORY;
    }
    cudaError_t err = cudaMalloc(&tensor->gpu_data, capacity_bytes);
    if (err != cudaSuccess) {
        free(tensor->cpu_data);
        return VSLA_ERROR_MEMORY;
    }
    tensor->data = tensor->cpu_data; // Keep `data` pointing to CPU memory for compatibility
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_deallocate(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    if (tensor->cpu_data) {
        free(tensor->cpu_data);
        tensor->cpu_data = NULL;
    }
    if (tensor->gpu_data) {
        cudaFree(tensor->gpu_data);
        tensor->gpu_data = NULL;
    }
    tensor->data = NULL;
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_copy_to_device(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    if (!tensor || !tensor->cpu_data || !tensor->gpu_data) {
        return VSLA_ERROR_NULL_POINTER;
    }
    size_t capacity_bytes = vsla_dtype_size(tensor->dtype);
    for (uint8_t i = 0; i < tensor->rank; ++i) {
        capacity_bytes *= tensor->cap[i];
    }
    cudaError_t err = cudaMemcpy(tensor->gpu_data, tensor->cpu_data, capacity_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return VSLA_ERROR_CUDA;
    }
    tensor->gpu_valid = true;
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_copy_to_host(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    if (!tensor || !tensor->cpu_data || !tensor->gpu_data) {
        return VSLA_ERROR_NULL_POINTER;
    }
    size_t capacity_bytes = vsla_dtype_size(tensor->dtype);
    for (uint8_t i = 0; i < tensor->rank; ++i) {
        capacity_bytes *= tensor->cap[i];
    }
    cudaError_t err = cudaMemcpy(tensor->cpu_data, tensor->gpu_data, capacity_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return VSLA_ERROR_CUDA;
    }
    tensor->cpu_valid = true;
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_synchronize(vsla_context_t* ctx) {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? VSLA_SUCCESS : VSLA_ERROR_CUDA;
}

/* Stub implementations - TODO: Implement actual CUDA kernels */
static vsla_error_t cuda_add(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_add(out, a, b);
}

static vsla_error_t cuda_sub(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_sub(out, a, b);
}

static vsla_error_t cuda_scale(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    return vsla_cuda_kernel_scale(out, in, scalar);
}

static vsla_error_t cuda_hadamard(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_hadamard(out, a, b);
}

static vsla_error_t cuda_fill(vsla_context_t* ctx, vsla_tensor_t* tensor, double value) {
    return vsla_cuda_kernel_fill(tensor, value);
}

static vsla_error_t cuda_matmul(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_transpose(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    (void)out; (void)tensor;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_reshape(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]) {
    (void)tensor; (void)new_rank; (void)new_shape;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_broadcast(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in) {
    (void)out; (void)in;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_sum(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    return vsla_cuda_kernel_sum(tensor, result);
}

static vsla_error_t cuda_mean(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    (void)tensor; (void)result;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_norm(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* norm) {
    (void)tensor; (void)norm;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_max(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* max) {
    (void)tensor; (void)max;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_min(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* min) {
    (void)tensor; (void)min;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_conv(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_kron(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_init(void* config) {
    (void)config;
    
    /* Initialize CUDA context */
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return VSLA_ERROR_CUDA;
    }
    
    return VSLA_SUCCESS;
}

static void cuda_cleanup(void) {
    cudaDeviceReset();
}

/* Backend interface creation */
vsla_backend_interface_t* vsla_backend_cuda_create(void) {
    static vsla_backend_interface_t cuda_backend = {
        .caps = {
            .supports_gpu = true,
            .supports_multi_gpu = true,
            .supports_unified_memory = true,
            .supports_async = true,
            .max_tensor_size = SIZE_MAX,
            .name = "CUDA",
            .version = "1.0.0"
        },
        
        /* Memory management */
        .allocate = cuda_allocate,
        .deallocate = cuda_deallocate,
        .copy_to_device = cuda_copy_to_device,
        .copy_to_host = cuda_copy_to_host,
        .synchronize = cuda_synchronize,
        
        /* Basic arithmetic operations */
        .add = cuda_add,
        .sub = cuda_sub,
        .scale = cuda_scale,
        .hadamard = cuda_hadamard,
        .fill = cuda_fill,
        
        /* Linear algebra operations */
        .matmul = cuda_matmul,
        .transpose = cuda_transpose,
        
        /* Tensor operations */
        .reshape = cuda_reshape,
        .broadcast = cuda_broadcast,
        
        /* Reduction operations */
        .sum = cuda_sum,
        .mean = cuda_mean,
        .norm = cuda_norm,
        .max = cuda_max,
        .min = cuda_min,
        
        /* Advanced operations */
        .conv = cuda_conv,
        .kron = cuda_kron,
        
        /* Backend lifecycle */
        .init = cuda_init,
        .cleanup = cuda_cleanup
    };
    
    return &cuda_backend;
}

#else /* !VSLA_ENABLE_CUDA */

/* Stub implementation when CUDA is not available */
vsla_backend_interface_t* vsla_backend_cuda_create(void) {
    return NULL;
}

#endif /* VSLA_ENABLE_CUDA */