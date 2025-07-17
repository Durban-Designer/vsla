/**
 * @file vsla_unified.c
 * @brief Implementation of hardware-agnostic unified VSLA interface
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_gpu.h"
#include "vsla/vsla_conv.h"
#include "vsla/vsla_ops.h"
#include "vsla/vsla_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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

// Unified tensor structure that abstracts CPU/GPU
struct vsla_tensor {
    // Core tensor info
    uint8_t rank;
    uint64_t* shape;
    uint64_t* cap;
    uint64_t* stride;
    vsla_model_t model;
    vsla_dtype_t dtype;
    
    // Memory management
    void* cpu_data;              // CPU memory
    void* gpu_data;              // GPU memory (if available)
    size_t data_size;            // Total data size in bytes
    vsla_backend_t location;     // Current data location
    bool cpu_valid;              // CPU data is up-to-date
    bool gpu_valid;              // GPU data is up-to-date
    
    // Context reference
    struct vsla_context* ctx;
};

// VSLA runtime context
struct vsla_context {
    // Configuration
    vsla_config_t config;
    vsla_backend_t active_backend;
    
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
    vsla_gpu_context_t* gpu_ctx;
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
    
    // Auto-select based on availability
    if (detect_cuda()) return VSLA_BACKEND_CUDA;
    if (detect_rocm()) return VSLA_BACKEND_ROCM;
    if (detect_oneapi()) return VSLA_BACKEND_ONEAPI;
    return VSLA_BACKEND_CPU;
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
    ctx->active_backend = select_best_backend(&ctx->config);
    
    // Initialize GPU context if available
#ifdef VSLA_ENABLE_CUDA
    if (ctx->active_backend == VSLA_BACKEND_CUDA) {
        ctx->gpu_ctx = vsla_gpu_init(ctx->config.device_id);
        if (ctx->gpu_ctx) {
            size_t free_mb, total_mb;
            if (vsla_gpu_get_memory_usage(ctx->gpu_ctx, &free_mb, &total_mb) == VSLA_SUCCESS) {
                ctx->gpu_memory_total = total_mb * 1024 * 1024;
                ctx->gpu_memory_free = free_mb * 1024 * 1024;
            }
        }
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
    if (ctx->gpu_ctx) {
        vsla_gpu_destroy(ctx->gpu_ctx);
    }
#endif
    
    free(ctx->fft_backends);
    free(ctx);
}

vsla_error_t vsla_get_runtime_info(const vsla_context_t* ctx,
                                    vsla_backend_t* backend,
                                    char* device_name,
                                    double* memory_gb) {
    if (!ctx) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (backend) *backend = ctx->active_backend;
    
    if (device_name) {
        switch (ctx->active_backend) {
            case VSLA_BACKEND_CPU:
                strcpy(device_name, "CPU");
                break;
            case VSLA_BACKEND_CUDA:
#ifdef VSLA_ENABLE_CUDA
                if (ctx->gpu_ctx) {
                    vsla_gpu_get_device_info(ctx->gpu_device_id, device_name, memory_gb);
                    return VSLA_SUCCESS;
                }
#endif
                strcpy(device_name, "CUDA (not initialized)");
                break;
            default:
                strcpy(device_name, "Unknown");
        }
    }
    
    if (memory_gb) {
        if (ctx->active_backend == VSLA_BACKEND_CUDA) {
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
    if (!ctx || ctx->active_backend != VSLA_BACKEND_CUDA) return false;
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
    if (!ctx || !shape || rank == 0) return NULL;
    
    vsla_tensor_t* tensor = calloc(1, sizeof(vsla_tensor_t));
    if (!tensor) return NULL;
    
    // Initialize basic fields
    tensor->rank = rank;
    tensor->model = model;
    tensor->dtype = dtype;
    tensor->ctx = ctx;
    
    // Allocate shape arrays
    size_t shape_size = rank * sizeof(uint64_t);
    tensor->shape = malloc(shape_size);
    tensor->cap = malloc(shape_size);
    tensor->stride = malloc(shape_size);
    
    if (!tensor->shape || !tensor->cap || !tensor->stride) {
        free(tensor->shape);
        free(tensor->cap);
        free(tensor->stride);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and calculate strides
    memcpy(tensor->shape, shape, shape_size);
    memcpy(tensor->cap, shape, shape_size);  // Initially no padding
    
    // Calculate strides (row-major)
    size_t stride = vsla_dtype_size(dtype);
    for (int i = rank - 1; i >= 0; i--) {
        tensor->stride[i] = stride;
        stride *= tensor->cap[i];
    }
    
    // Calculate total size
    tensor->data_size = calculate_tensor_size(rank, shape, dtype);
    
    // Decide where to allocate
    bool use_gpu = should_use_gpu(ctx, tensor->data_size);
    
    if (use_gpu) {
#ifdef VSLA_ENABLE_CUDA
        // Allocate on GPU
        vsla_gpu_tensor_t gpu_temp = {0};
        gpu_temp.rank = rank;
        gpu_temp.dtype = dtype;
        gpu_temp.cap = tensor->cap;
        
        if (vsla_gpu_tensor_alloc(&gpu_temp, ctx->gpu_ctx) == VSLA_SUCCESS) {
            tensor->gpu_data = gpu_temp.gpu_data;
            tensor->gpu_valid = true;
            tensor->location = VSLA_BACKEND_CUDA;
            ctx->stats.gpu_operations++;
        } else {
            use_gpu = false;  // Fall back to CPU
        }
#else
        use_gpu = false;
#endif
    }
    
    if (!use_gpu) {
        // Allocate on CPU
        tensor->cpu_data = calloc(1, tensor->data_size);
        if (!tensor->cpu_data) {
            free(tensor->shape);
            free(tensor->cap);
            free(tensor->stride);
            free(tensor);
            return NULL;
        }
        tensor->cpu_valid = true;
        tensor->location = VSLA_BACKEND_CPU;
        ctx->stats.cpu_operations++;
    }
    
    ctx->stats.total_operations++;
    return tensor;
}

void vsla_tensor_free(vsla_tensor_t* tensor) {
    if (!tensor) return;
    
    free(tensor->cpu_data);
#ifdef VSLA_ENABLE_CUDA
    if (tensor->gpu_data) {
        cudaFree(tensor->gpu_data);
    }
#endif
    
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    free(tensor);
}

// === Data Access ===

static vsla_error_t ensure_cpu_valid(vsla_tensor_t* tensor) {
    if (!tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (tensor->cpu_valid) return VSLA_SUCCESS;
    
#ifdef VSLA_ENABLE_CUDA
    if (tensor->gpu_valid && tensor->gpu_data) {
        // Allocate CPU memory if needed
        if (!tensor->cpu_data) {
            tensor->cpu_data = malloc(tensor->data_size);
            if (!tensor->cpu_data) return VSLA_ERROR_MEMORY;
        }
        
        // Copy from GPU to CPU
        cudaError_t err = cudaMemcpy(tensor->cpu_data, tensor->gpu_data,
                                      tensor->data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
        
        tensor->cpu_valid = true;
        tensor->ctx->stats.transfer_time_ms += 0.1;  // TODO: Actual timing
    }
#endif
    
    return VSLA_SUCCESS;
}

static vsla_error_t ensure_gpu_valid(vsla_tensor_t* tensor) {
    if (!tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (tensor->gpu_valid) return VSLA_SUCCESS;
    
#ifdef VSLA_ENABLE_CUDA
    if (tensor->cpu_valid && tensor->cpu_data) {
        // Allocate GPU memory if needed
        if (!tensor->gpu_data) {
            vsla_gpu_tensor_t gpu_temp = {0};
            gpu_temp.rank = tensor->rank;
            gpu_temp.dtype = tensor->dtype;
            gpu_temp.cap = tensor->cap;
            
            if (vsla_gpu_tensor_alloc(&gpu_temp, tensor->ctx->gpu_ctx) != VSLA_SUCCESS) {
                return VSLA_ERROR_GPU_FAILURE;
            }
            tensor->gpu_data = gpu_temp.gpu_data;
        }
        
        // Copy from CPU to GPU
        cudaError_t err = cudaMemcpy(tensor->gpu_data, tensor->cpu_data,
                                      tensor->data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
        
        tensor->gpu_valid = true;
        tensor->ctx->stats.transfer_time_ms += 0.1;  // TODO: Actual timing
    }
#endif
    
    return VSLA_SUCCESS;
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
    
    // Determine where to execute
    bool use_gpu = ctx->active_backend == VSLA_BACKEND_CUDA &&
                   out->data_size >= ctx->memory_threshold;
    
    if (use_gpu) {
#ifdef VSLA_ENABLE_CUDA
        // Ensure all tensors are on GPU
        vsla_tensor_t* mut_a = (vsla_tensor_t*)a;
        vsla_tensor_t* mut_b = (vsla_tensor_t*)b;
        
        vsla_error_t err;
        err = ensure_gpu_valid(out);
        if (err != VSLA_SUCCESS) use_gpu = false;
        err = ensure_gpu_valid(mut_a);
        if (err != VSLA_SUCCESS) use_gpu = false;
        err = ensure_gpu_valid(mut_b);
        if (err != VSLA_SUCCESS) use_gpu = false;
        
        if (use_gpu) {
            // Create temporary GPU tensor wrappers
            vsla_gpu_tensor_t gpu_out = {
                .rank = out->rank, .dtype = out->dtype,
                .shape = out->shape, .cap = out->cap,
                .gpu_data = out->gpu_data
            };
            vsla_gpu_tensor_t gpu_a = {
                .rank = a->rank, .dtype = a->dtype,
                .shape = a->shape, .cap = a->cap,
                .gpu_data = mut_a->gpu_data
            };
            vsla_gpu_tensor_t gpu_b = {
                .rank = b->rank, .dtype = b->dtype,
                .shape = b->shape, .cap = b->cap,
                .gpu_data = mut_b->gpu_data
            };
            
            err = vsla_gpu_add(&gpu_out, &gpu_a, &gpu_b, ctx->gpu_ctx);
            if (err == VSLA_SUCCESS) {
                out->gpu_valid = true;
                out->cpu_valid = false;
                ctx->stats.gpu_operations++;
                ctx->stats.gpu_time_ms += 0.01;  // TODO: Actual timing
                return VSLA_SUCCESS;
            }
        }
#endif
    }
    
    // Fall back to CPU
    ensure_cpu_valid((vsla_tensor_t*)a);
    ensure_cpu_valid((vsla_tensor_t*)b);
    ensure_cpu_valid(out);
    
    // Create CPU tensor wrappers
    vsla_tensor_t cpu_out = {
        .rank = out->rank, .model = out->model, .dtype = out->dtype,
        .shape = out->shape, .cap = out->cap, .stride = out->stride,
        .data = out->cpu_data
    };
    vsla_tensor_t cpu_a = {
        .rank = a->rank, .model = a->model, .dtype = a->dtype,
        .shape = a->shape, .cap = a->cap, .stride = a->stride,
        .data = ((vsla_tensor_t*)a)->cpu_data
    };
    vsla_tensor_t cpu_b = {
        .rank = b->rank, .model = b->model, .dtype = b->dtype,
        .shape = b->shape, .cap = b->cap, .stride = b->stride,
        .data = ((vsla_tensor_t*)b)->cpu_data
    };
    
    vsla_error_t err = vsla_add_op(&cpu_out, &cpu_a, &cpu_b);
    if (err == VSLA_SUCCESS) {
        out->cpu_valid = true;
        out->gpu_valid = false;
        ctx->stats.cpu_operations++;
        ctx->stats.cpu_time_ms += 0.01;  // TODO: Actual timing
    }
    
    ctx->stats.total_operations++;
    return err;
}

vsla_error_t vsla_conv(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* signal,
                       const vsla_tensor_t* kernel) {
    if (!ctx || !out || !signal || !kernel) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // For large convolutions, prefer FFT on GPU if available
    size_t signal_size = signal->shape[0];
    size_t kernel_size = kernel->shape[0];
    bool use_fft = (signal_size * kernel_size) > 1024;
    bool use_gpu = use_fft && ctx->active_backend == VSLA_BACKEND_CUDA;
    
    if (use_gpu) {
#ifdef VSLA_ENABLE_CUDA
        // Ensure tensors are on GPU
        vsla_tensor_t* mut_signal = (vsla_tensor_t*)signal;
        vsla_tensor_t* mut_kernel = (vsla_tensor_t*)kernel;
        
        vsla_error_t err;
        err = ensure_gpu_valid(out);
        if (err == VSLA_SUCCESS) err = ensure_gpu_valid(mut_signal);
        if (err == VSLA_SUCCESS) err = ensure_gpu_valid(mut_kernel);
        
        if (err == VSLA_SUCCESS) {
            // Create GPU tensor wrappers and call GPU convolution
            vsla_gpu_tensor_t gpu_out = {
                .rank = out->rank, .dtype = out->dtype,
                .shape = out->shape, .cap = out->cap,
                .gpu_data = out->gpu_data
            };
            vsla_gpu_tensor_t gpu_signal = {
                .rank = signal->rank, .dtype = signal->dtype,
                .shape = signal->shape, .cap = signal->cap,
                .gpu_data = mut_signal->gpu_data
            };
            vsla_gpu_tensor_t gpu_kernel = {
                .rank = kernel->rank, .dtype = kernel->dtype,
                .shape = kernel->shape, .cap = kernel->cap,
                .gpu_data = mut_kernel->gpu_data
            };
            
            err = vsla_gpu_conv_fft(&gpu_out, &gpu_signal, &gpu_kernel, ctx->gpu_ctx);
            if (err == VSLA_SUCCESS) {
                out->gpu_valid = true;
                out->cpu_valid = false;
                ctx->stats.gpu_operations++;
                return VSLA_SUCCESS;
            }
        }
#endif
    }
    
    // Fall back to CPU
    ensure_cpu_valid((vsla_tensor_t*)signal);
    ensure_cpu_valid((vsla_tensor_t*)kernel);
    ensure_cpu_valid(out);
    
    // Create CPU tensor wrappers
    vsla_tensor_t cpu_out = {
        .rank = out->rank, .model = out->model, .dtype = out->dtype,
        .shape = out->shape, .cap = out->cap, .stride = out->stride,
        .data = out->cpu_data
    };
    vsla_tensor_t cpu_signal = {
        .rank = signal->rank, .model = signal->model, .dtype = signal->dtype,
        .shape = signal->shape, .cap = signal->cap, .stride = signal->stride,
        .data = ((vsla_tensor_t*)signal)->cpu_data
    };
    vsla_tensor_t cpu_kernel = {
        .rank = kernel->rank, .model = kernel->model, .dtype = kernel->dtype,
        .shape = kernel->shape, .cap = kernel->cap, .stride = kernel->stride,
        .data = ((vsla_tensor_t*)kernel)->cpu_data
    };
    
    vsla_error_t err;
    if (use_fft) {
        err = vsla_conv_fft(&cpu_out, &cpu_signal, &cpu_kernel);
    } else {
        err = vsla_conv_direct(&cpu_out, &cpu_signal, &cpu_kernel);
    }
    
    if (err == VSLA_SUCCESS) {
        out->cpu_valid = true;
        out->gpu_valid = false;
        ctx->stats.cpu_operations++;
    }
    
    ctx->stats.total_operations++;
    return err;
}

// === Performance and Statistics ===

// === Additional Operations (Stubs) ===

vsla_error_t vsla_fill(vsla_context_t* ctx, vsla_tensor_t* tensor, double value) {
    if (!ctx || !tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    ensure_cpu_valid(tensor);
    
    // Simple CPU implementation
    size_t elements = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        elements *= tensor->shape[i];
    }
    
    if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->cpu_data;
        for (size_t i = 0; i < elements; i++) {
            data[i] = (float)value;
        }
    } else {
        double* data = (double*)tensor->cpu_data;
        for (size_t i = 0; i < elements; i++) {
            data[i] = value;
        }
    }
    
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_scale(vsla_context_t* ctx,
                        vsla_tensor_t* out,
                        const vsla_tensor_t* in,
                        double scalar) {
    if (!ctx || !out || !in) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Simple CPU implementation for now
    ensure_cpu_valid((vsla_tensor_t*)in);
    ensure_cpu_valid(out);
    
    size_t elements = 1;
    for (uint8_t i = 0; i < in->rank; i++) {
        elements *= in->shape[i];
    }
    
    if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)in->cpu_data;
        float* out_data = (float*)out->cpu_data;
        float scale_f = (float)scalar;
        
        for (size_t i = 0; i < elements; i++) {
            out_data[i] = in_data[i] * scale_f;
        }
    } else {
        const double* in_data = (const double*)in->cpu_data;
        double* out_data = (double*)out->cpu_data;
        
        for (size_t i = 0; i < elements; i++) {
            out_data[i] = in_data[i] * scalar;
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    
    return VSLA_SUCCESS;
}

vsla_backend_t vsla_recommend_backend(vsla_context_t* ctx,
                                       const char* operation,
                                       const vsla_tensor_t** inputs,
                                       size_t input_count) {
    if (!ctx || !operation || !inputs) return VSLA_BACKEND_CPU;
    
    // Simple heuristic: use GPU for large tensors
    size_t total_elements = 0;
    for (size_t i = 0; i < input_count; i++) {
        if (inputs[i]) {
            size_t elements = 1;
            for (uint8_t j = 0; j < inputs[i]->rank; j++) {
                elements *= inputs[i]->shape[j];
            }
            total_elements += elements;
        }
    }
    
    // Use GPU for operations on large tensors
    if (total_elements > 1024 && ctx->active_backend == VSLA_BACKEND_CUDA) {
        return VSLA_BACKEND_CUDA;
    }
    
    return VSLA_BACKEND_CPU;
}

vsla_error_t vsla_tensor_get_info(const vsla_tensor_t* tensor,
                                   uint8_t* rank,
                                   const uint64_t** shape,
                                   vsla_model_t* model,
                                   vsla_dtype_t* dtype) {
    if (!tensor) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (rank) *rank = tensor->rank;
    if (shape) *shape = tensor->shape;
    if (model) *model = tensor->model;
    if (dtype) *dtype = tensor->dtype;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_get_stats(const vsla_context_t* ctx, vsla_stats_t* stats) {
    if (!ctx || !stats) return VSLA_ERROR_INVALID_ARGUMENT;
    
    *stats = ctx->stats;
    stats->total_time_ms = (double)(clock() - ctx->start_time) * 1000.0 / CLOCKS_PER_SEC;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_synchronize(vsla_context_t* ctx) {
    if (!ctx) return VSLA_ERROR_INVALID_ARGUMENT;
    
#ifdef VSLA_ENABLE_CUDA
    if (ctx->active_backend == VSLA_BACKEND_CUDA) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return VSLA_ERROR_GPU_FAILURE;
    }
#endif
    
    return VSLA_SUCCESS;
}