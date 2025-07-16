/**
 * @file vsla_gpu.h
 * @brief GPU acceleration support for VSLA using CUDA
 * 
 * This module provides GPU acceleration for VSLA operations using CUDA.
 * It extends the core tensor structure with GPU memory management and
 * provides CUDA kernels for high-performance tensor operations.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_GPU_H
#define VSLA_GPU_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VSLA_ENABLE_CUDA
#include <cuda_runtime.h>
// Note: We implement our own GPU kernels, not using cuBLAS/cuFFT
// This allows us to showcase VSLA's variable-shape advantages
#endif

/**
 * @brief GPU memory location flags
 */
typedef enum {
    VSLA_GPU_LOCATION_CPU = 0,      /**< Data is in CPU memory */
    VSLA_GPU_LOCATION_GPU = 1,      /**< Data is in GPU memory */
    VSLA_GPU_LOCATION_UNIFIED = 2   /**< Data is in unified memory */
} vsla_gpu_location_t;

/**
 * @brief GPU-extended tensor structure
 * 
 * This structure extends vsla_tensor_t with GPU-specific fields
 * for memory management and asynchronous operations.
 */
typedef struct {
    // Base tensor fields
    uint8_t    rank;      /**< Number of axes (dimensions) */
    uint8_t    model;     /**< Model: 0 = convolution, 1 = Kronecker */
    uint8_t    dtype;     /**< Data type: 0 = f64, 1 = f32 */
    uint8_t    flags;     /**< Reserved for future use */
    
    uint64_t  *shape;     /**< Logical extent per axis */
    uint64_t  *cap;       /**< Padded/allocated extent per axis */
    uint64_t  *stride;    /**< Byte strides for row-major traversal */
    void      *data;      /**< CPU data buffer */
    
    // GPU-specific fields
#ifdef VSLA_ENABLE_CUDA
    void      *gpu_data;        /**< GPU memory pointer */
    cudaStream_t stream;        /**< CUDA stream for async operations */
    uint8_t   location;         /**< Memory location (CPU/GPU/unified) */
    uint8_t   gpu_id;          /**< GPU device ID */
    size_t    gpu_capacity;     /**< GPU memory capacity in bytes */
#else
    void      *gpu_data;        /**< Placeholder when CUDA disabled */
    void      *stream;          /**< Placeholder when CUDA disabled */
    uint8_t   location;         /**< Always CPU when CUDA disabled */
    uint8_t   gpu_id;          /**< Always 0 when CUDA disabled */
    size_t    gpu_capacity;     /**< Always 0 when CUDA disabled */
#endif
} vsla_gpu_tensor_t;

/**
 * @brief GPU context for managing CUDA resources
 */
typedef struct {
#ifdef VSLA_ENABLE_CUDA
    cudaStream_t default_stream;    /**< Default CUDA stream */
    int device_id;                  /**< Current GPU device ID */
    size_t total_memory;            /**< Total GPU memory in bytes */
    size_t free_memory;             /**< Free GPU memory in bytes */
    
    // VSLA-specific GPU resources
    void *fft_workspace;            /**< Workspace for our custom FFT implementation */
    size_t fft_workspace_size;      /**< Size of FFT workspace */
    void *temp_buffer;              /**< Temporary buffer for variable-shape operations */
    size_t temp_buffer_size;        /**< Size of temporary buffer */
#else
    void *default_stream;           /**< Placeholder when CUDA disabled */
    int device_id;                  /**< Always -1 when CUDA disabled */
    size_t total_memory;            /**< Always 0 when CUDA disabled */
    size_t free_memory;             /**< Always 0 when CUDA disabled */
    void *fft_workspace;            /**< Placeholder when CUDA disabled */
    size_t fft_workspace_size;      /**< Always 0 when CUDA disabled */
    void *temp_buffer;              /**< Placeholder when CUDA disabled */
    size_t temp_buffer_size;        /**< Always 0 when CUDA disabled */
#endif
} vsla_gpu_context_t;

// GPU Initialization and Management
/**
 * @brief Initialize GPU context
 * 
 * @param device_id GPU device ID (-1 for auto-select)
 * @return GPU context or NULL on error
 */
vsla_gpu_context_t* vsla_gpu_init(int device_id);

/**
 * @brief Destroy GPU context and cleanup resources
 * 
 * @param ctx GPU context to destroy
 */
void vsla_gpu_destroy(vsla_gpu_context_t* ctx);

/**
 * @brief Check if GPU support is available
 * 
 * @return true if CUDA is available and functional
 */
bool vsla_gpu_is_available(void);

/**
 * @brief Get GPU device information
 * 
 * @param device_id GPU device ID
 * @param name Buffer for device name (minimum 256 chars)
 * @param memory_gb Total memory in GB
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_get_device_info(int device_id, char* name, double* memory_gb);

// GPU Memory Management
/**
 * @brief Create GPU tensor from CPU tensor
 * 
 * @param cpu_tensor Source CPU tensor
 * @param ctx GPU context
 * @return GPU tensor or NULL on error
 */
vsla_gpu_tensor_t* vsla_gpu_tensor_from_cpu(const vsla_tensor_t* cpu_tensor, 
                                             vsla_gpu_context_t* ctx);

/**
 * @brief Create CPU tensor from GPU tensor
 * 
 * @param gpu_tensor Source GPU tensor
 * @return CPU tensor or NULL on error
 */
vsla_tensor_t* vsla_gpu_tensor_to_cpu(const vsla_gpu_tensor_t* gpu_tensor);

/**
 * @brief Allocate GPU memory for tensor
 * 
 * @param tensor GPU tensor to allocate memory for
 * @param ctx GPU context
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_tensor_alloc(vsla_gpu_tensor_t* tensor, vsla_gpu_context_t* ctx);

/**
 * @brief Free GPU memory for tensor
 * 
 * @param tensor GPU tensor to free memory for
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_tensor_free(vsla_gpu_tensor_t* tensor);

/**
 * @brief Copy data from CPU to GPU
 * 
 * @param tensor GPU tensor
 * @param cpu_data Source CPU data
 * @param async Use asynchronous copy
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_tensor_copy_to_gpu(vsla_gpu_tensor_t* tensor, 
                                          const void* cpu_data, bool async);

/**
 * @brief Copy data from GPU to CPU
 * 
 * @param tensor GPU tensor
 * @param cpu_data Destination CPU data
 * @param async Use asynchronous copy
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_tensor_copy_to_cpu(const vsla_gpu_tensor_t* tensor, 
                                          void* cpu_data, bool async);

/**
 * @brief Synchronize GPU operations
 * 
 * @param tensor GPU tensor (NULL for device sync)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_tensor_sync(const vsla_gpu_tensor_t* tensor);

// GPU Operations
/**
 * @brief GPU tensor addition
 * 
 * @param result Result tensor (GPU)
 * @param a First operand tensor (GPU)
 * @param b Second operand tensor (GPU)
 * @param ctx GPU context
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_add(vsla_gpu_tensor_t* result, 
                          const vsla_gpu_tensor_t* a, 
                          const vsla_gpu_tensor_t* b, 
                          vsla_gpu_context_t* ctx);

/**
 * @brief GPU tensor scaling
 * 
 * @param result Result tensor (GPU)
 * @param tensor Input tensor (GPU)
 * @param scale Scale factor
 * @param ctx GPU context
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_scale(vsla_gpu_tensor_t* result, 
                            const vsla_gpu_tensor_t* tensor, 
                            double scale, 
                            vsla_gpu_context_t* ctx);

/**
 * @brief GPU matrix multiplication
 * 
 * @param result Result tensor (GPU)
 * @param a First matrix tensor (GPU)
 * @param b Second matrix tensor (GPU)
 * @param ctx GPU context
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_matmul(vsla_gpu_tensor_t* result, 
                             const vsla_gpu_tensor_t* a, 
                             const vsla_gpu_tensor_t* b, 
                             vsla_gpu_context_t* ctx);

/**
 * @brief GPU FFT-based convolution
 * 
 * @param result Result tensor (GPU)
 * @param signal Signal tensor (GPU)
 * @param kernel Kernel tensor (GPU)
 * @param ctx GPU context
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_conv_fft(vsla_gpu_tensor_t* result, 
                               const vsla_gpu_tensor_t* signal, 
                               const vsla_gpu_tensor_t* kernel, 
                               vsla_gpu_context_t* ctx);

// Utility Functions
/**
 * @brief Get GPU memory usage statistics
 * 
 * @param ctx GPU context
 * @param used_mb Used memory in MB
 * @param total_mb Total memory in MB
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_get_memory_usage(vsla_gpu_context_t* ctx, 
                                       size_t* used_mb, 
                                       size_t* total_mb);

/**
 * @brief Get optimal GPU grid/block dimensions
 * 
 * @param size Problem size
 * @param block_size Optimal block size
 * @param grid_size Optimal grid size
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_get_launch_config(size_t size, 
                                        size_t* block_size, 
                                        size_t* grid_size);

#ifdef __cplusplus
}
#endif

#endif // VSLA_GPU_H