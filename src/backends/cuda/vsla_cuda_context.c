/**
 * @file vsla_gpu.c
 * @brief GPU acceleration implementation for VSLA using CUDA
 * 
 * @copyright MIT License
 */

#include "vsla/internal/vsla_gpu.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return VSLA_ERROR_GPU_FAILURE; \
    } \
} while(0)

// CUDA kernels for VSLA variable-shape operations
__global__ void vsla_gpu_add_variable_shape_f32(float* result, const float* a, const float* b,
                                                 const uint64_t* shape_a, const uint64_t* shape_b,
                                                 const uint64_t* shape_result, uint8_t rank) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total result size
    size_t total_size = 1;
    for (int i = 0; i < rank; i++) {
        total_size *= shape_result[i];
    }
    
    if (idx < total_size) {
        // Convert linear index to multi-dimensional coordinates
        size_t coords[8]; // Support up to 8 dimensions
        size_t temp_idx = idx;
        for (int i = rank - 1; i >= 0; i--) {
            coords[i] = temp_idx % shape_result[i];
            temp_idx /= shape_result[i];
        }
        
        // Calculate corresponding indices in input tensors (with zero-padding)
        size_t idx_a = 0, idx_b = 0;
        size_t stride_a = 1, stride_b = 1;
        
        for (int i = rank - 1; i >= 0; i--) {
            // Zero-pad if coordinate exceeds tensor dimension
            if (coords[i] < shape_a[i]) {
                idx_a += coords[i] * stride_a;
            }
            if (coords[i] < shape_b[i]) {
                idx_b += coords[i] * stride_b;
            }
            stride_a *= shape_a[i];
            stride_b *= shape_b[i];
        }
        
        // Perform addition with automatic zero-padding
        float val_a = (coords[0] < shape_a[0]) ? a[idx_a] : 0.0f;
        float val_b = (coords[0] < shape_b[0]) ? b[idx_b] : 0.0f;
        result[idx] = val_a + val_b;
    }
}

__global__ void vsla_gpu_add_kernel_f32(float* result, const float* a, const float* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void vsla_gpu_add_kernel_f64(double* result, const double* a, const double* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void vsla_gpu_scale_kernel_f32(float* result, const float* tensor, float scale, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = tensor[idx] * scale;
    }
}

__global__ void vsla_gpu_scale_kernel_f64(double* result, const double* tensor, double scale, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = tensor[idx] * scale;
    }
}

// VSLA-specific GPU matrix multiplication (our own implementation)
__global__ void vsla_gpu_matmul_kernel_f32(float* result, const float* a, const float* b,
                                            int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        result[row * n + col] = sum;
    }
}

__global__ void vsla_gpu_matmul_kernel_f64(double* result, const double* a, const double* b,
                                            int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        result[row * n + col] = sum;
    }
}

// Simple GPU-based FFT implementation for demonstration
// In production, we'd implement a more sophisticated FFT
__global__ void vsla_gpu_fft_1d_kernel_f32(float* real, float* imag, int n, int inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // This is a simplified demonstration - real FFT implementation would be more complex
    // For now, just demonstrate the concept
    if (inverse) {
        real[idx] = real[idx] / n;
        imag[idx] = imag[idx] / n;
    }
}

#endif // VSLA_ENABLE_CUDA

// GPU Context Management
vsla_gpu_context_t* vsla_gpu_init(int device_id) {
#ifdef VSLA_ENABLE_CUDA
    // Check for CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        return NULL;
    }
    
    // Select device
    if (device_id < 0) {
        device_id = 0; // Auto-select first device
    }
    if (device_id >= device_count) {
        return NULL;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Allocate context
    vsla_gpu_context_t* ctx = (vsla_gpu_context_t*)malloc(sizeof(vsla_gpu_context_t));
    if (!ctx) {
        return NULL;
    }
    
    ctx->device_id = device_id;
    
    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&ctx->default_stream));
    
    // Get memory information
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    ctx->free_memory = free_mem;
    ctx->total_memory = total_mem;
    
    // Initialize VSLA-specific GPU resources
    ctx->fft_workspace = NULL;
    ctx->fft_workspace_size = 0;
    ctx->temp_buffer = NULL;
    ctx->temp_buffer_size = 0;
    
    return ctx;
#else
    (void)device_id;
    return NULL;
#endif
}

void vsla_gpu_destroy(vsla_gpu_context_t* ctx) {
    if (!ctx) return;
    
#ifdef VSLA_ENABLE_CUDA
    // Free VSLA-specific GPU resources
    if (ctx->fft_workspace) {
        cudaFree(ctx->fft_workspace);
    }
    if (ctx->temp_buffer) {
        cudaFree(ctx->temp_buffer);
    }
    
    // Destroy CUDA stream
    if (ctx->default_stream) {
        cudaStreamDestroy(ctx->default_stream);
    }
    
    cudaDeviceReset();
#endif
    
    free(ctx);
}

bool vsla_gpu_is_available(void) {
#ifdef VSLA_ENABLE_CUDA
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

vsla_error_t vsla_gpu_get_device_info(int device_id, char* name, double* memory_gb) {
    if (!name || !memory_gb) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_id >= device_count) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    strncpy(name, prop.name, 255);
    name[255] = '\0';
    *memory_gb = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    
    return VSLA_SUCCESS;
#else
    strcpy(name, "No CUDA support");
    *memory_gb = 0.0;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

// GPU Memory Management
vsla_gpu_tensor_t* vsla_gpu_tensor_from_cpu(const vsla_tensor_t* cpu_tensor, 
                                             vsla_gpu_context_t* ctx) {
    if (!cpu_tensor || !ctx) {
        return NULL;
    }
    
    vsla_gpu_tensor_t* gpu_tensor = (vsla_gpu_tensor_t*)malloc(sizeof(vsla_gpu_tensor_t));
    if (!gpu_tensor) {
        return NULL;
    }
    
    // Copy CPU tensor fields
    gpu_tensor->rank = cpu_tensor->rank;
    gpu_tensor->model = cpu_tensor->model;
    gpu_tensor->dtype = cpu_tensor->dtype;
    gpu_tensor->flags = cpu_tensor->flags;
    
    // Allocate and copy shape arrays
    size_t shape_size = cpu_tensor->rank * sizeof(uint64_t);
    gpu_tensor->shape = (uint64_t*)malloc(shape_size);
    gpu_tensor->cap = (uint64_t*)malloc(shape_size);
    gpu_tensor->stride = (uint64_t*)malloc(shape_size);
    
    if (!gpu_tensor->shape || !gpu_tensor->cap || !gpu_tensor->stride) {
        free(gpu_tensor->shape);
        free(gpu_tensor->cap);
        free(gpu_tensor->stride);
        free(gpu_tensor);
        return NULL;
    }
    
    memcpy(gpu_tensor->shape, cpu_tensor->shape, shape_size);
    memcpy(gpu_tensor->cap, cpu_tensor->cap, shape_size);
    memcpy(gpu_tensor->stride, cpu_tensor->stride, shape_size);
    
    // Set initial GPU fields
    gpu_tensor->data = NULL;
    gpu_tensor->gpu_data = NULL;
    gpu_tensor->location = VSLA_GPU_LOCATION_CPU;
    gpu_tensor->gpu_id = ctx->device_id;
    gpu_tensor->gpu_capacity = 0;
    
#ifdef VSLA_ENABLE_CUDA
    gpu_tensor->stream = ctx->default_stream;
#else
    gpu_tensor->stream = NULL;
#endif
    
    return gpu_tensor;
}

vsla_tensor_t* vsla_gpu_tensor_to_cpu(const vsla_gpu_tensor_t* gpu_tensor) {
    if (!gpu_tensor) {
        return NULL;
    }
    
    // Create CPU tensor with same parameters
    vsla_tensor_t* cpu_tensor = vsla_new(gpu_tensor->rank, gpu_tensor->shape, 
                                        (vsla_model_t)gpu_tensor->model, 
                                        (vsla_dtype_t)gpu_tensor->dtype);
    if (!cpu_tensor) {
        return NULL;
    }
    
    // Copy data from GPU to CPU if needed
    if (gpu_tensor->location == VSLA_GPU_LOCATION_GPU && gpu_tensor->gpu_data) {
        size_t data_size = 1;
        for (uint8_t i = 0; i < gpu_tensor->rank; i++) {
            data_size *= gpu_tensor->cap[i];
        }
        data_size *= (gpu_tensor->dtype == VSLA_DTYPE_F32) ? sizeof(float) : sizeof(double);
        
#ifdef VSLA_ENABLE_CUDA
        CUDA_CHECK(cudaMemcpy(cpu_tensor->data, gpu_tensor->gpu_data, 
                             data_size, cudaMemcpyDeviceToHost));
#endif
    }
    
    return cpu_tensor;
}

vsla_error_t vsla_gpu_tensor_alloc(vsla_gpu_tensor_t* tensor, vsla_gpu_context_t* ctx) {
    if (!tensor || !ctx) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // Calculate required memory
    size_t data_size = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        data_size *= tensor->cap[i];
    }
    data_size *= (tensor->dtype == VSLA_DTYPE_F32) ? sizeof(float) : sizeof(double);
    
    // Allocate GPU memory
    CUDA_CHECK(cudaSetDevice(ctx->device_id));
    CUDA_CHECK(cudaMalloc(&tensor->gpu_data, data_size));
    
    tensor->gpu_capacity = data_size;
    tensor->location = VSLA_GPU_LOCATION_GPU;
    tensor->gpu_id = ctx->device_id;
    
    return VSLA_SUCCESS;
#else
    (void)tensor;
    (void)ctx;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_tensor_free(vsla_gpu_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    if (tensor->gpu_data) {
        cudaFree(tensor->gpu_data);
        tensor->gpu_data = NULL;
    }
#endif
    
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    free(tensor);
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_gpu_tensor_copy_to_gpu(vsla_gpu_tensor_t* tensor, 
                                          const void* cpu_data, bool async) {
    if (!tensor || !cpu_data) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    if (!tensor->gpu_data) {
        return VSLA_ERROR_INVALID_STATE;
    }
    
    cudaMemcpyKind kind = async ? cudaMemcpyHostToDevice : cudaMemcpyHostToDevice;
    
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(tensor->gpu_data, cpu_data, 
                                  tensor->gpu_capacity, kind, 
                                  (cudaStream_t)tensor->stream));
    } else {
        CUDA_CHECK(cudaMemcpy(tensor->gpu_data, cpu_data, 
                             tensor->gpu_capacity, kind));
    }
    
    return VSLA_SUCCESS;
#else
    (void)tensor;
    (void)cpu_data;
    (void)async;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_tensor_copy_to_cpu(const vsla_gpu_tensor_t* tensor, 
                                          void* cpu_data, bool async) {
    if (!tensor || !cpu_data) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    if (!tensor->gpu_data) {
        return VSLA_ERROR_INVALID_STATE;
    }
    
    cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
    
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(cpu_data, tensor->gpu_data, 
                                  tensor->gpu_capacity, kind, 
                                  (cudaStream_t)tensor->stream));
    } else {
        CUDA_CHECK(cudaMemcpy(cpu_data, tensor->gpu_data, 
                             tensor->gpu_capacity, kind));
    }
    
    return VSLA_SUCCESS;
#else
    (void)tensor;
    (void)cpu_data;
    (void)async;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_tensor_sync(const vsla_gpu_tensor_t* tensor) {
#ifdef VSLA_ENABLE_CUDA
    if (tensor && tensor->stream) {
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)tensor->stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return VSLA_SUCCESS;
#else
    (void)tensor;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

// GPU Operations
vsla_error_t vsla_gpu_add(vsla_gpu_tensor_t* result, 
                          const vsla_gpu_tensor_t* a, 
                          const vsla_gpu_tensor_t* b, 
                          vsla_gpu_context_t* ctx) {
    if (!result || !a || !b || !ctx) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // Calculate total size
    size_t total_elements = 1;
    for (uint8_t i = 0; i < result->rank; i++) {
        total_elements *= result->cap[i];
    }
    
    // Launch configuration
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;
    
    // Launch appropriate kernel based on data type
    if (result->dtype == VSLA_DTYPE_F32) {
        vsla_gpu_add_kernel_f32<<<grid_size, block_size, 0, (cudaStream_t)result->stream>>>(
            (float*)result->gpu_data, (const float*)a->gpu_data, 
            (const float*)b->gpu_data, total_elements);
    } else {
        vsla_gpu_add_kernel_f64<<<grid_size, block_size, 0, (cudaStream_t)result->stream>>>(
            (double*)result->gpu_data, (const double*)a->gpu_data, 
            (const double*)b->gpu_data, total_elements);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return VSLA_SUCCESS;
#else
    (void)result;
    (void)a;
    (void)b;
    (void)ctx;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_scale(vsla_gpu_tensor_t* result, 
                            const vsla_gpu_tensor_t* tensor, 
                            double scale, 
                            vsla_gpu_context_t* ctx) {
    if (!result || !tensor || !ctx) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // Calculate total size
    size_t total_elements = 1;
    for (uint8_t i = 0; i < result->rank; i++) {
        total_elements *= result->cap[i];
    }
    
    // Launch configuration
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;
    
    // Launch appropriate kernel based on data type
    if (result->dtype == VSLA_DTYPE_F32) {
        vsla_gpu_scale_kernel_f32<<<grid_size, block_size, 0, (cudaStream_t)result->stream>>>(
            (float*)result->gpu_data, (const float*)tensor->gpu_data, 
            (float)scale, total_elements);
    } else {
        vsla_gpu_scale_kernel_f64<<<grid_size, block_size, 0, (cudaStream_t)result->stream>>>(
            (double*)result->gpu_data, (const double*)tensor->gpu_data, 
            scale, total_elements);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return VSLA_SUCCESS;
#else
    (void)result;
    (void)tensor;
    (void)scale;
    (void)ctx;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_matmul(vsla_gpu_tensor_t* result, 
                             const vsla_gpu_tensor_t* a, 
                             const vsla_gpu_tensor_t* b, 
                             vsla_gpu_context_t* ctx) {
    if (!result || !a || !b || !ctx) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // For matrix multiplication, we need at least 2D tensors
    if (a->rank < 2 || b->rank < 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Get matrix dimensions
    int m = (int)a->shape[0];
    int k = (int)a->shape[1];
    int n = (int)b->shape[1];
    
    // Check dimension compatibility
    if (a->shape[1] != b->shape[0]) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    // Launch configuration for 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    
    // Launch our own VSLA matrix multiplication kernel
    if (result->dtype == VSLA_DTYPE_F32) {
        vsla_gpu_matmul_kernel_f32<<<gridDim, blockDim, 0, (cudaStream_t)result->stream>>>(
            (float*)result->gpu_data, (const float*)a->gpu_data, 
            (const float*)b->gpu_data, m, n, k);
    } else {
        vsla_gpu_matmul_kernel_f64<<<gridDim, blockDim, 0, (cudaStream_t)result->stream>>>(
            (double*)result->gpu_data, (const double*)a->gpu_data, 
            (const double*)b->gpu_data, m, n, k);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return VSLA_SUCCESS;
#else
    (void)result;
    (void)a;
    (void)b;
    (void)ctx;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_conv_fft(vsla_gpu_tensor_t* result, 
                               const vsla_gpu_tensor_t* signal, 
                               const vsla_gpu_tensor_t* kernel, 
                               vsla_gpu_context_t* ctx) {
    if (!result || !signal || !kernel || !ctx) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // TODO: Implement FFT convolution using cuFFT
    // This is a placeholder for the full implementation
    return VSLA_ERROR_NOT_IMPLEMENTED;
#else
    (void)result;
    (void)signal;
    (void)kernel;
    (void)ctx;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

// Utility Functions
vsla_error_t vsla_gpu_get_memory_usage(vsla_gpu_context_t* ctx, 
                                       size_t* used_mb, 
                                       size_t* total_mb) {
    if (!ctx || !used_mb || !total_mb) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    *used_mb = (total_mem - free_mem) / (1024 * 1024);
    *total_mb = total_mem / (1024 * 1024);
    
    return VSLA_SUCCESS;
#else
    *used_mb = 0;
    *total_mb = 0;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}

vsla_error_t vsla_gpu_get_launch_config(size_t size, 
                                        size_t* block_size, 
                                        size_t* grid_size) {
    if (!block_size || !grid_size) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
#ifdef VSLA_ENABLE_CUDA
    // Default block size
    *block_size = 256;
    
    // Calculate grid size
    *grid_size = (size + *block_size - 1) / *block_size;
    
    return VSLA_SUCCESS;
#else
    *block_size = 1;
    *grid_size = size;
    return VSLA_ERROR_NOT_IMPLEMENTED;
#endif
}