/**
 * @file vsla_backend_cuda_new.c
 * @brief CUDA GPU backend implementation with single-kernel operations
 *
 * @copyright MIT License
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla_backend_cuda_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>

/* CUDA Backend Memory Management */
static vsla_error_t cuda_allocate(vsla_tensor_t* tensor) {
    if (!tensor || tensor->data_size == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    /* Allocate GPU memory if not already allocated */
    if (!tensor->gpu_data) {
        cudaError_t err = cudaMalloc(&tensor->gpu_data, tensor->data_size);
        if (err != cudaSuccess) {
            return VSLA_ERROR_MEMORY;
        }
    }
    
    /* Ensure CPU memory exists for data transfers */
    if (!tensor->cpu_data) {
        tensor->cpu_data = calloc(1, tensor->data_size);
        if (!tensor->cpu_data) {
            cudaFree(tensor->gpu_data);
            tensor->gpu_data = NULL;
            return VSLA_ERROR_MEMORY;
        }
        tensor->data = tensor->cpu_data;
    }
    
    tensor->location = VSLA_BACKEND_CUDA;
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_deallocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->gpu_data) {
        cudaFree(tensor->gpu_data);
        tensor->gpu_data = NULL;
    }
    
    tensor->gpu_valid = false;
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_copy_to_device(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (!tensor->gpu_data) {
        vsla_error_t err = cuda_allocate(tensor);
        if (err != VSLA_SUCCESS) {
            return err;
        }
    }
    
    if (tensor->cpu_valid && tensor->cpu_data) {
        cudaError_t err = cudaMemcpy(tensor->gpu_data, tensor->cpu_data, 
                                     tensor->data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return VSLA_ERROR_CUDA;
        }
        tensor->gpu_valid = true;
        tensor->location = VSLA_BACKEND_CUDA;
    }
    
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_copy_to_host(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->gpu_valid && tensor->gpu_data && tensor->cpu_data) {
        cudaError_t err = cudaMemcpy(tensor->cpu_data, tensor->gpu_data, 
                                     tensor->data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return VSLA_ERROR_CUDA;
        }
        tensor->cpu_valid = true;
        tensor->location = VSLA_BACKEND_CPU;
    }
    
    return VSLA_SUCCESS;
}

static vsla_error_t cuda_synchronize(void) {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? VSLA_SUCCESS : VSLA_ERROR_CUDA;
}

/* Stub implementations - TODO: Implement actual CUDA kernels */
static vsla_error_t cuda_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_add(out, a, b);
}

static vsla_error_t cuda_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_sub(out, a, b);
}

static vsla_error_t cuda_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    return vsla_cuda_kernel_scale(out, in, scalar);
}

static vsla_error_t cuda_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    return vsla_cuda_kernel_hadamard(out, a, b);
}

static vsla_error_t cuda_fill(vsla_tensor_t* tensor, double value) {
    return vsla_cuda_kernel_fill(tensor, value);
}

static vsla_error_t cuda_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    (void)out; (void)tensor;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_reshape(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]) {
    (void)tensor; (void)new_rank; (void)new_shape;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_broadcast(vsla_tensor_t* out, const vsla_tensor_t* in) {
    (void)out; (void)in;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_sum(const vsla_tensor_t* tensor, double* result) {
    return vsla_cuda_kernel_sum(tensor, result);
}

static vsla_error_t cuda_mean(const vsla_tensor_t* tensor, double* result) {
    (void)tensor; (void)result;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_norm(const vsla_tensor_t* tensor, double* norm) {
    (void)tensor; (void)norm;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_max(const vsla_tensor_t* tensor, double* max) {
    (void)tensor; (void)max;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_min(const vsla_tensor_t* tensor, double* min) {
    (void)tensor; (void)min;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_conv(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cuda_kron(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
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