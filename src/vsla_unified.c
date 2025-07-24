/**
 * @file vsla_unified.c
 * @brief Implementation of hardware-agnostic unified VSLA interface
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_tensor.h"
#include "vsla/internal/vsla_tensor_internal.h"
#ifdef VSLA_ENABLE_CUDA
#include "vsla/internal/vsla_gpu.h"
#endif
#include "vsla/vsla_core.h"
#include "vsla/internal/vsla_backend.h"
#if VSLA_BUILD_CPU
#include "vsla/internal/vsla_backend_cpu.h"
#endif
#include "vsla/internal/vsla_window.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// Forward declarations for CPU stacking functions - MODERN API WITH CONTEXT
extern vsla_window_t* cpu_window_create(vsla_context_t* ctx, size_t window_size, uint8_t rank, vsla_dtype_t dtype);
extern void cpu_window_destroy(vsla_window_t* window);
extern vsla_tensor_t* cpu_window_push(vsla_window_t* window, vsla_tensor_t* tensor);
extern vsla_pyramid_t* cpu_pyramid_create(vsla_context_t* ctx, size_t levels, size_t window_size, uint8_t rank, vsla_dtype_t dtype, bool discard_partials);
extern void cpu_pyramid_destroy(vsla_pyramid_t* pyramid);
extern vsla_tensor_t* cpu_pyramid_push(vsla_pyramid_t* pyramid, vsla_tensor_t* tensor);
extern vsla_tensor_t** cpu_pyramid_flush(vsla_pyramid_t* pyramid, size_t* count);

// Forward declarations for vendor FFT backends
typedef struct {
    bool available;
    const char* name;
    const char* version;
    vsla_error_t (*init)(void);
    void (*cleanup)(void);
    vsla_error_t (*conv_fft)(void* out, const void* a, const void* b, 
                             vsla_dtype_t dtype, size_t size);
} vsla_fft_backend_impl_t;

// Note: vsla_tensor structure is now defined in vsla_tensor.h

// Performance statistics structure
typedef struct {
    size_t operations_count;
    double total_time_ms;
    size_t memory_allocated;
    size_t memory_peak;
    size_t cpu_operations;
    size_t gpu_operations;
    size_t total_operations;
} vsla_stats_t;

// VSLA runtime context
struct vsla_context {
    // Configuration
    vsla_config_t config;
    vsla_backend_t active_backend_type;
    const vsla_backend_interface_t* active_backend;
    
    // Hardware info
    bool cuda_available;
    bool rocm_available;
    bool oneapi_available;
    int gpu_device_id;
    size_t gpu_memory_total;
    size_t gpu_memory_free;
    
    // FFT backends
    vsla_fft_backend_impl_t* fft_backends;
    size_t fft_backend_count;
    size_t active_fft_backend;
    
    // Performance statistics
    vsla_stats_t stats;
    clock_t start_time;
    
    // Memory management
    bool auto_migration;
    size_t memory_threshold;  // Threshold for GPU allocation
    
#ifdef VSLA_ENABLE_CUDA
    void* gpu_ctx;
#endif
};

// === Hardware Detection ===

static bool detect_cuda(void) {
#ifdef VSLA_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

static bool detect_rocm(void) {
    // TODO: Implement ROCm detection
    return false;
}

static bool detect_oneapi(void) {
    // TODO: Implement oneAPI detection
    return false;
}

static vsla_backend_t select_best_backend(const vsla_config_t* config) {
    if (config && config->backend != VSLA_BACKEND_AUTO) {
        return config->backend;
    }
    
    // Auto-select based on availability and what's compiled in
    if (detect_cuda()) return VSLA_BACKEND_CUDA;
    if (detect_rocm()) return VSLA_BACKEND_ROCM;
    if (detect_oneapi()) return VSLA_BACKEND_ONEAPI;
    
#if VSLA_BUILD_CPU
    return VSLA_BACKEND_CPU;
#else
    // If no backends are available, return CUDA as fallback
    // (the init function will handle the error if CUDA isn't available)
    return VSLA_BACKEND_CUDA;
#endif
}

// === Context Management ===

vsla_context_t* vsla_init(const vsla_config_t* config) {
    vsla_context_t* ctx = calloc(1, sizeof(vsla_context_t));
    if (!ctx) return NULL;
    
    // Copy configuration or use defaults
    if (config) {
        ctx->config = *config;
    } else {
        ctx->config.backend = VSLA_BACKEND_AUTO;
        ctx->config.device_id = -1;
        ctx->config.memory_limit = 0;
        ctx->config.optimization_hint = VSLA_HINT_NONE;
        ctx->config.enable_profiling = false;
        ctx->config.verbose = false;
    }
    
    // Detect hardware
    ctx->cuda_available = detect_cuda();
    ctx->rocm_available = detect_rocm();
    ctx->oneapi_available = detect_oneapi();
    
    // Select backend
    ctx->active_backend_type = select_best_backend(&ctx->config);

    // Initialize backend function pointers
    ctx->active_backend = NULL; // Initialize to NULL first
    
    switch (ctx->active_backend_type) {
#if VSLA_BUILD_CPU
        case VSLA_BACKEND_CPU:
            ctx->active_backend = vsla_backend_cpu_create();
            break;
#endif
#ifdef VSLA_ENABLE_CUDA
        case VSLA_BACKEND_CUDA:
            ctx->active_backend = vsla_backend_cuda_create();
            break;
#endif
        default:
            // Fallback logic: use the first available backend
#if VSLA_BUILD_CPU
            ctx->active_backend = vsla_backend_cpu_create();
            ctx->active_backend_type = VSLA_BACKEND_CPU;
#elif defined(VSLA_ENABLE_CUDA)
            ctx->active_backend = vsla_backend_cuda_create();
            ctx->active_backend_type = VSLA_BACKEND_CUDA;
#else
            // No backends available - this is an error
            free(ctx);
            return NULL;
#endif
            break;
    }
    
    // Check if backend creation failed
    if (!ctx->active_backend) {
        free(ctx);
        return NULL;
    }
    
    // Initialize GPU context if available
#ifdef VSLA_ENABLE_CUDA
    if (ctx->active_backend_type == VSLA_BACKEND_CUDA) {
        // ctx->gpu_ctx = vsla_gpu_init(ctx->config.device_id);
        // if (ctx->gpu_ctx) {
        //     size_t free_mb, total_mb;
        //     if (vsla_gpu_get_memory_usage(ctx->gpu_ctx, &free_mb, &total_mb) == VSLA_SUCCESS) {
        //         ctx->gpu_memory_total = total_mb * 1024 * 1024;
        //         ctx->gpu_memory_free = free_mb * 1024 * 1024;
        //     }
        // }
    }
#endif
    
    // Default settings
    ctx->auto_migration = true;
    ctx->memory_threshold = 1024 * 1024;  // 1MB threshold for GPU
    
    // Initialize statistics
    ctx->start_time = clock();
    
    return ctx;
}

void vsla_cleanup(vsla_context_t* ctx) {
    if (!ctx) return;
    
#ifdef VSLA_ENABLE_CUDA
    // if (ctx->gpu_ctx) {
    //     vsla_gpu_destroy(ctx->gpu_ctx);
    // }
#endif
    
    free(ctx->fft_backends);
    free(ctx);
}

vsla_error_t vsla_get_runtime_info(const vsla_context_t* ctx,
                                    vsla_backend_t* backend,
                                    char* device_name,
                                    double* memory_gb) {
    if (!ctx) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (backend) *backend = ctx->active_backend_type;
    
    if (device_name) {
        switch (ctx->active_backend_type) {
#if VSLA_BUILD_CPU
            case VSLA_BACKEND_CPU:
                strcpy(device_name, "CPU");
                break;
#endif
#ifdef VSLA_ENABLE_CUDA
            case VSLA_BACKEND_CUDA:
                // if (ctx->gpu_ctx) {
                //     vsla_gpu_get_device_info(ctx->gpu_device_id, device_name, memory_gb);
                //     return VSLA_SUCCESS;
                // }
                strcpy(device_name, "CUDA (not initialized)");
                break;
#endif
            default:
                strcpy(device_name, "Unknown");
        }
    }
    
    if (memory_gb) {
        if (ctx->active_backend_type == VSLA_BACKEND_CUDA) {
            *memory_gb = ctx->gpu_memory_total / (1024.0 * 1024.0 * 1024.0);
        } else {
            *memory_gb = 0.0;  // TODO: Get system memory
        }
    }
    
    return VSLA_SUCCESS;
}

// === Tensor Management ===

static size_t calculate_tensor_size(uint8_t rank, const uint64_t* shape, vsla_dtype_t dtype) {
    size_t elements = 1;
    for (uint8_t i = 0; i < rank; i++) {
        elements *= shape[i];
    }
    return elements * vsla_dtype_size(dtype);
}

static bool should_use_gpu(vsla_context_t* ctx, size_t data_size) {
    if (!ctx || ctx->active_backend_type != VSLA_BACKEND_CUDA) return false;
    if (!ctx->auto_migration) return false;
    if (data_size < ctx->memory_threshold) return false;
    if (ctx->gpu_memory_free < data_size * 2) return false;  // Need space for operations
    return true;
}

vsla_tensor_t* vsla_tensor_create(vsla_context_t* ctx,
                                   uint8_t rank,
                                   const uint64_t* shape,
                                   vsla_model_t model,
                                   vsla_dtype_t dtype) {
    // printf("vsla_tensor_create called: ctx=%p, rank=%d, shape=%p\n", (void*)ctx, rank, (void*)shape);
    // fflush(stdout);
    
    if (!ctx || !shape || rank == 0) {
        // printf("vsla_tensor_create: invalid params\n");
        // fflush(stdout);
        return NULL;
    }
    
    vsla_tensor_t* tensor = calloc(1, sizeof(vsla_tensor_t));
    if (!tensor) return NULL;
    
    // Initialize basic fields
    tensor->rank = rank;
    tensor->model = model;
    tensor->dtype = dtype;
    
    // Initialize reference counting
    tensor->ref_count = 1;        // Start with 1 reference
    tensor->owns_data = true;     // By default, tensor owns its data
    
    // Allocate shape arrays
    size_t shape_size = rank * sizeof(uint64_t);
    tensor->shape = malloc(shape_size);
    tensor->cap = malloc(shape_size);
    
    if (!tensor->shape || !tensor->cap) {
        free(tensor->shape);
        free(tensor->cap);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and calculate strides
    memcpy(tensor->shape, shape, shape_size);
    memcpy(tensor->cap, shape, shape_size);  // Initially no padding
    
    // Decide where to allocate
    bool use_gpu = should_use_gpu(ctx, calculate_tensor_size(rank, shape, dtype));
    
    if (use_gpu) {
#ifdef VSLA_ENABLE_CUDA
        // Allocate on GPU
        // if (vsla_gpu_tensor_alloc(tensor, ctx->gpu_ctx) == VSLA_SUCCESS) {
        //     tensor->gpu_valid = true;
        //     tensor->location = VSLA_BACKEND_CUDA;
        //     ctx->stats.gpu_operations++;
        // } else {
        //     use_gpu = false;  // Fall back to CPU
        // }
#else
        use_gpu = false;
#endif
    }
    
    if (!ctx->active_backend) {
        printf("ERROR: No active backend!\n");
        fflush(stdout);
        free(tensor->shape);
        free(tensor->cap);
        free(tensor);
        return NULL;
    }

    if (!use_gpu) {
        // Allocate stride array
        tensor->stride = malloc(rank * sizeof(uint64_t));
        if (!tensor->stride) {
            free(tensor->shape);
            free(tensor->cap);
            free(tensor);
            return NULL;
        }
        
        // Calculate strides (row-major order)
        size_t dtype_size = vsla_dtype_size(dtype);
        tensor->stride[rank - 1] = dtype_size;
        for (int i = rank - 2; i >= 0; i--) {
            tensor->stride[i] = tensor->stride[i + 1] * tensor->cap[i + 1];
        }
        
        // Allocate on CPU
        // printf("Allocating on CPU, backend=%p, allocate=%p\n", 
        //        (void*)ctx->active_backend, 
        //        (void*)(ctx->active_backend ? ctx->active_backend->allocate : NULL));
        // fflush(stdout);
        
        if (!ctx->active_backend || !ctx->active_backend->allocate) {
            printf("ERROR: No backend or allocate function!\n");
            fflush(stdout);
            free(tensor->shape);
            free(tensor->cap);
            free(tensor->stride);
            free(tensor);
            return NULL;
        }
        
        ctx->active_backend->allocate(ctx, tensor);
        
        // Ensure cpu_data points to data (for compatibility)
        tensor->cpu_data = tensor->data;
        tensor->cpu_valid = (tensor->data != NULL);
        
        ctx->stats.cpu_operations++;
    }
    
    ctx->stats.total_operations++;
    return tensor;
}

void vsla_tensor_free(vsla_tensor_t* tensor) {
    // Use reference counting system for memory management
    vsla_tensor_release(tensor);
}

// === Data Access ===

static vsla_error_t ensure_cpu_valid(vsla_tensor_t* tensor) {
    if (!tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // if (tensor->cpu_valid) return VSLA_SUCCESS;
    
#ifdef VSLA_ENABLE_CUDA
    // if (tensor->gpu_valid && tensor->gpu_data) {
    //     // Allocate CPU memory if needed
    //     if (!tensor->cpu_data) {
    //         tensor->cpu_data = malloc(tensor->data_size);
    //         if (!tensor->cpu_data) return VSLA_ERROR_MEMORY;
    //     }
        
    //     // Copy from GPU to CPU
    //     cudaError_t err = cudaMemcpy(tensor->cpu_data, tensor->gpu_data,
    //                                   tensor->data_size, cudaMemcpyDeviceToHost);
    //     if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
        
    //     tensor->cpu_valid = true;
    //     tensor->ctx->stats.transfer_time_ms += 0.1;  // TODO: Actual timing
    // }
#endif
    
    return VSLA_SUCCESS;
}

static vsla_error_t ensure_gpu_valid(vsla_tensor_t* tensor) {
    if (!tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // if (tensor->gpu_valid) return VSLA_SUCCESS;
    
#ifdef VSLA_ENABLE_CUDA
    // if (tensor->cpu_valid && tensor->cpu_data) {
    //     // Allocate GPU memory if needed
    //     if (!tensor->gpu_data) {
    //         vsla_gpu_tensor_t gpu_temp = {0};
    //         gpu_temp.rank = tensor->rank;
    //         gpu_temp.dtype = tensor->dtype;
    //         gpu_temp.cap = tensor->cap;
            
    //         if (vsla_gpu_tensor_alloc(&gpu_temp, tensor->ctx->gpu_ctx) != VSLA_SUCCESS) {
    //             return VSLA_ERROR_GPU_FAILURE;
    //         }
    //         tensor->gpu_data = gpu_temp.gpu_data;
    //     }
        
    //     // Copy from CPU to GPU
    //     cudaError_t err = cudaMemcpy(tensor->gpu_data, tensor->cpu_data,
    //                                   tensor->data_size, cudaMemcpyHostToDevice);
    //     if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
        
    //     tensor->gpu_valid = true;
    //     tensor->ctx->stats.transfer_time_ms += 0.1;  // TODO: Actual timing
    // }
#endif
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_get_f64(vsla_context_t* ctx, const vsla_tensor_t* tensor, const uint64_t indices[], double* value) {
    if (!ctx || !tensor || !indices || !value) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Ensure data is available on CPU
    vsla_tensor_t* mut_tensor = (vsla_tensor_t*)tensor;
    if (ensure_cpu_valid(mut_tensor) != VSLA_SUCCESS) return VSLA_ERROR_MEMORY;
    
    // Calculate linear index
    if (!tensor->cpu_data || !tensor->stride) return VSLA_ERROR_MEMORY;
    
    size_t linear_idx = 0;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->shape[i]) {
            // Zero-extension for out-of-bounds access (VSLA semantics)
            *value = 0.0;
            return VSLA_SUCCESS;
        }
        linear_idx += indices[i] * (tensor->stride[i] / sizeof(double));
    }
    
    // Extract value
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data_ptr = (double*)tensor->cpu_data;
        *value = data_ptr[linear_idx];
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data_ptr = (float*)tensor->cpu_data;
        *value = (double)data_ptr[linear_idx];
    } else {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_set_f64(vsla_context_t* ctx, vsla_tensor_t* tensor, const uint64_t indices[], double value) {
    if (!ctx || !tensor || !indices) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Ensure data is available on CPU
    if (ensure_cpu_valid(tensor) != VSLA_SUCCESS) return VSLA_ERROR_MEMORY;
    
    // Calculate linear index
    if (!tensor->cpu_data || !tensor->stride) return VSLA_ERROR_MEMORY;
    
    size_t linear_idx = 0;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->shape[i]) {
            // Cannot set out-of-bounds values
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
        linear_idx += indices[i] * (tensor->stride[i] / sizeof(double));
    }
    
    // Set value
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data_ptr = (double*)tensor->cpu_data;
        data_ptr[linear_idx] = value;
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data_ptr = (float*)tensor->cpu_data;
        data_ptr[linear_idx] = (float)value;
    } else {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Mark GPU data as invalid since CPU data was modified
    tensor->gpu_valid = false;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_tensor_copy_to_host(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    printf("Copying to host\n");
    if (!ctx || !tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->copy_to_host(ctx, tensor);
}

const void* vsla_tensor_data(const vsla_tensor_t* tensor, size_t* size) {
    if (!tensor) return NULL;
    
    // Ensure CPU data is valid
    vsla_tensor_t* mut_tensor = (vsla_tensor_t*)tensor;
    if (ensure_cpu_valid(mut_tensor) != VSLA_SUCCESS) return NULL;
    
    if (size) *size = tensor->data_size;
    return tensor->cpu_data;
}

void* vsla_tensor_data_mut(vsla_tensor_t* tensor, size_t* size) {
    if (!tensor) return NULL;
    
    // Ensure CPU data is valid
    if (ensure_cpu_valid(tensor) != VSLA_SUCCESS) return NULL;
    
    // Mark GPU data as invalid since CPU data will be modified
    tensor->gpu_valid = false;
    
    if (size) *size = tensor->data_size;
    return tensor->cpu_data;
}

// === Basic Operations ===

vsla_error_t vsla_add(vsla_context_t* ctx,
                      vsla_tensor_t* out,
                      const vsla_tensor_t* a,
                      const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->add(ctx, out, a, b);
}

vsla_error_t vsla_sub(vsla_context_t* ctx,
                      vsla_tensor_t* out,
                      const vsla_tensor_t* a,
                      const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->sub(ctx, out, a, b);
}

vsla_error_t vsla_scale(vsla_context_t* ctx,
                        vsla_tensor_t* out,
                        const vsla_tensor_t* in,
                        double scalar) {
    if (!ctx || !out || !in) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->scale(ctx, out, in, scalar);
}

vsla_error_t vsla_hadamard(vsla_context_t* ctx,
                           vsla_tensor_t* out,
                           const vsla_tensor_t* a,
                           const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->hadamard(ctx, out, a, b);
}

vsla_error_t vsla_fill(vsla_context_t* ctx, vsla_tensor_t* tensor, double value) {
    if (!ctx || !tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->fill(ctx, tensor, value);
}

vsla_error_t vsla_matmul(vsla_context_t* ctx,
                        vsla_tensor_t* out,
                        const vsla_tensor_t* a,
                        const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    if (!ctx->active_backend->matmul) return VSLA_ERROR_NOT_IMPLEMENTED;
    return ctx->active_backend->matmul(ctx, out, a, b);
}

vsla_error_t vsla_conv(vsla_context_t* ctx,
                      vsla_tensor_t* out,
                      const vsla_tensor_t* a,
                      const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    if (!ctx->active_backend->conv) return VSLA_ERROR_NOT_IMPLEMENTED;
    return ctx->active_backend->conv(ctx, out, a, b);
}

vsla_error_t vsla_sum(vsla_context_t* ctx,
                      const vsla_tensor_t* tensor,
                      double* result) {
    if (!ctx || !tensor || !result) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->sum(ctx, tensor, result);
}

vsla_error_t vsla_norm(vsla_context_t* ctx,
                       const vsla_tensor_t* tensor,
                       double* result) {
    if (!ctx || !tensor || !result) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->norm(ctx, tensor, result);
}

vsla_error_t vsla_kron(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* a,
                       const vsla_tensor_t* b) {
    if (!ctx || !out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->kron(ctx, out, a, b);
}

vsla_error_t vsla_shrink(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    if (!ctx || !tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->shrink(ctx, tensor);
}

vsla_error_t vsla_stack(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* const* tensors, size_t k) {
    if (!ctx || !out || !tensors) return VSLA_ERROR_INVALID_ARGUMENT;
    return ctx->active_backend->stack(ctx, out, tensors, k);
}

// === Window Stacking Functions ===

vsla_window_t* vsla_window_create(vsla_context_t* ctx, size_t window_size, uint8_t rank, vsla_dtype_t dtype) {
    if (!ctx) return NULL; // Context is REQUIRED for proper tensor creation
    return cpu_window_create(ctx, window_size, rank, dtype);
}

void vsla_window_destroy(vsla_window_t* window) {
    cpu_window_destroy(window);
}

vsla_tensor_t* vsla_window_push(vsla_window_t* window, vsla_tensor_t* tensor) {
    if (!window || !window->ctx) return NULL; // Context must be available
    return cpu_window_push(window, tensor);
}

// === Pyramid Stacking Functions ===

vsla_pyramid_t* vsla_pyramid_create(vsla_context_t* ctx, size_t levels, size_t window_size, uint8_t rank, vsla_dtype_t dtype, bool discard_partials) {
    if (!ctx) return NULL; // Context is REQUIRED for proper tensor creation
    return cpu_pyramid_create(ctx, levels, window_size, rank, dtype, discard_partials);
}

void vsla_pyramid_destroy(vsla_pyramid_t* pyramid) {
    cpu_pyramid_destroy(pyramid);
}

vsla_tensor_t* vsla_pyramid_push(vsla_pyramid_t* pyramid, vsla_tensor_t* tensor) {
    return cpu_pyramid_push(pyramid, tensor);
}

vsla_tensor_t** vsla_pyramid_flush(vsla_pyramid_t* pyramid, size_t* count) {
    return cpu_pyramid_flush(pyramid, count);
}

// === Performance and Statistics ===

vsla_error_t vsla_get_stats(const vsla_context_t* ctx, vsla_stats_t* stats) {
    if (!ctx || !stats) return VSLA_ERROR_INVALID_ARGUMENT;
    
    *stats = ctx->stats;
    stats->total_time_ms = (double)(clock() - ctx->start_time) * 1000.0 / CLOCKS_PER_SEC;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_synchronize(vsla_context_t* ctx) {
    if (!ctx) return VSLA_ERROR_INVALID_ARGUMENT;
    
#ifdef VSLA_ENABLE_CUDA
    if (ctx->active_backend_type == VSLA_BACKEND_CUDA) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
    }
#endif
    
    return VSLA_SUCCESS;
}