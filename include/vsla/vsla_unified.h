/**
 * @file vsla_unified.h
 * @brief Hardware-agnostic unified interface for VSLA operations
 * 
 * This module provides a single, simple API that automatically uses the best
 * available hardware (CPU/GPU) and vendor libraries (cuFFT, rocFFT, MKL) to
 * achieve maximum performance without requiring users to manage hardware details.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_UNIFIED_H
#define VSLA_UNIFIED_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hardware backend type
 */
typedef enum {
    VSLA_BACKEND_CPU = 0,           /**< CPU-only execution */
    VSLA_BACKEND_CUDA = 1,          /**< NVIDIA GPU via CUDA */
    VSLA_BACKEND_ROCM = 2,          /**< AMD GPU via ROCm */
    VSLA_BACKEND_ONEAPI = 3,        /**< Intel GPU via oneAPI */
    VSLA_BACKEND_AUTO = 4           /**< Auto-select best available */
} vsla_backend_t;

/**
 * @brief Unified tensor structure that abstracts CPU/GPU memory
 */
typedef struct vsla_unified_tensor vsla_unified_tensor_t;

/**
 * @brief VSLA runtime context for managing hardware resources
 */
typedef struct vsla_unified_context vsla_context_t;

/**
 * @brief Performance hints for optimization
 */
typedef enum {
    VSLA_HINT_NONE = 0,             /**< No specific hints */
    VSLA_HINT_LATENCY = 1,          /**< Optimize for low latency */
    VSLA_HINT_THROUGHPUT = 2,       /**< Optimize for high throughput */
    VSLA_HINT_MEMORY = 3,           /**< Optimize for memory efficiency */
    VSLA_HINT_ENERGY = 4            /**< Optimize for energy efficiency */
} vsla_hint_t;

/**
 * @brief Runtime configuration
 */
typedef struct {
    vsla_backend_t backend;         /**< Preferred backend (AUTO recommended) */
    int device_id;                  /**< Device ID (-1 for auto-select) */
    size_t memory_limit;            /**< Memory limit in bytes (0 = no limit) */
    vsla_hint_t optimization_hint;  /**< Performance optimization hint */
    bool enable_profiling;          /**< Enable performance profiling */
    bool verbose;                   /**< Enable verbose logging */
} vsla_config_t;

// === Core Initialization ===

/**
 * @brief Initialize VSLA runtime with automatic hardware detection
 * 
 * This function automatically detects available hardware (GPUs, vendor libraries)
 * and initializes the runtime for optimal performance.
 * 
 * @param config Optional configuration (NULL for auto-configuration)
 * @return VSLA context or NULL on error
 * 
 * @code
 * // Simple initialization with auto-configuration
 * vsla_context_t* ctx = vsla_init(NULL);
 * 
 * // Custom configuration
 * vsla_config_t config = {
 *     .backend = VSLA_BACKEND_AUTO,
 *     .optimization_hint = VSLA_HINT_THROUGHPUT
 * };
 * vsla_context_t* ctx = vsla_init(&config);
 * @endcode
 */
vsla_context_t* vsla_init(const vsla_config_t* config);

/**
 * @brief Cleanup VSLA runtime and release all resources
 * 
 * @param ctx VSLA context
 */
void vsla_cleanup(vsla_context_t* ctx);

/**
 * @brief Get runtime information
 * 
 * @param ctx VSLA context
 * @param backend Current backend being used
 * @param device_name Device name (buffer must be at least 256 chars)
 * @param memory_gb Available memory in GB
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_get_runtime_info(const vsla_context_t* ctx,
                                    vsla_backend_t* backend,
                                    char* device_name,
                                    double* memory_gb);

// === Tensor Management ===

/**
 * @brief Create a new tensor with automatic memory management
 * 
 * The tensor is automatically allocated on the best available device
 * (GPU if available and beneficial, otherwise CPU).
 * 
 * @param ctx VSLA context
 * @param rank Number of dimensions
 * @param shape Shape array
 * @param model Tensor model (convolution or Kronecker)
 * @param dtype Data type
 * @return New tensor or NULL on error
 * 
 * @code
 * uint64_t shape[] = {1024, 1024};
 * vsla_tensor_t* tensor = vsla_tensor_create(ctx, 2, shape, 
 *                                             VSLA_MODEL_A, VSLA_DTYPE_F32);
 * @endcode
 */
vsla_tensor_t* vsla_tensor_create(vsla_context_t* ctx,
                                   uint8_t rank,
                                   const uint64_t* shape,
                                   vsla_model_t model,
                                   vsla_dtype_t dtype);

/**
 * @brief Create tensor from existing data
 * 
 * @param ctx VSLA context
 * @param rank Number of dimensions
 * @param shape Shape array
 * @param model Tensor model
 * @param dtype Data type
 * @param data Data pointer (will be copied)
 * @param copy If true, copy data; if false, take ownership
 * @return New tensor or NULL on error
 */
vsla_tensor_t* vsla_tensor_from_data(vsla_context_t* ctx,
                                      uint8_t rank,
                                      const uint64_t* shape,
                                      vsla_model_t model,
                                      vsla_dtype_t dtype,
                                      const void* data,
                                      bool copy);

/**
 * @brief Free tensor and associated memory
 * 
 * @param tensor Tensor to free
 */
void vsla_tensor_free(vsla_tensor_t* tensor);

/**
 * @brief Get tensor data for reading
 * 
 * This function ensures data is accessible from CPU, performing
 * GPU->CPU transfer if necessary. The returned pointer is valid
 * until the tensor is modified or freed.
 * 
 * @param tensor Tensor
 * @param size Optional output for data size in bytes
 * @return Data pointer or NULL on error
 */
const void* vsla_tensor_data(const vsla_tensor_t* tensor, size_t* size);

/**
 * @brief Get mutable tensor data
 * 
 * @param tensor Tensor
 * @param size Optional output for data size in bytes
 * @return Mutable data pointer or NULL on error
 */
void* vsla_tensor_data_mut(vsla_tensor_t* tensor, size_t* size);

/**
 * @brief Get tensor properties
 * 
 * @param tensor Tensor
 * @param rank Output for rank (can be NULL)
 * @param shape Output for shape array (can be NULL)
 * @param model Output for model (can be NULL)
 * @param dtype Output for data type (can be NULL)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_tensor_get_info(const vsla_tensor_t* tensor,
                                   uint8_t* rank,
                                   const uint64_t** shape,
                                   vsla_model_t* model,
                                   vsla_dtype_t* dtype);

// === Basic Operations (Hardware-Agnostic) ===

/**
 * @brief Add two tensors element-wise
 * 
 * Automatically uses GPU if available and beneficial.
 * 
 * @param ctx VSLA context
 * @param out Output tensor (can be same as input for in-place)
 * @param a First tensor
 * @param b Second tensor
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_add(vsla_context_t* ctx,
                      vsla_tensor_t* out,
                      const vsla_tensor_t* a,
                      const vsla_tensor_t* b);

/**
 * @brief Subtract two tensors element-wise
 */
vsla_error_t vsla_sub(vsla_context_t* ctx,
                      vsla_tensor_t* out,
                      const vsla_tensor_t* a,
                      const vsla_tensor_t* b);

/**
 * @brief Multiply tensor by scalar
 */
vsla_error_t vsla_scale(vsla_context_t* ctx,
                        vsla_tensor_t* out,
                        const vsla_tensor_t* in,
                        double scalar);

/**
 * @brief Fill tensor with value
 */
vsla_error_t vsla_fill(vsla_context_t* ctx,
                       vsla_tensor_t* tensor,
                       double value);

/**
 * @brief Copy tensor
 */
vsla_error_t vsla_copy(vsla_context_t* ctx,
                       vsla_tensor_t* dst,
                       const vsla_tensor_t* src);

// === Advanced Operations ===

/**
 * @brief Convolution with automatic algorithm selection
 * 
 * Automatically selects the best algorithm (direct, FFT, or vendor FFT)
 * and hardware (CPU or GPU) based on tensor sizes and available resources.
 * 
 * @param ctx VSLA context
 * @param out Output tensor
 * @param signal Signal tensor
 * @param kernel Kernel tensor
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_conv(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* signal,
                       const vsla_tensor_t* kernel);

/**
 * @brief Kronecker product
 * 
 * @param ctx VSLA context
 * @param out Output tensor
 * @param a First tensor
 * @param b Second tensor
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_kron(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* a,
                       const vsla_tensor_t* b);

/**
 * @brief Matrix multiplication
 * 
 * Automatically uses vendor BLAS libraries (cuBLAS, rocBLAS, MKL)
 * for optimal performance.
 * 
 * @param ctx VSLA context
 * @param out Output tensor
 * @param a First matrix
 * @param b Second matrix
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_matmul(vsla_context_t* ctx,
                         vsla_tensor_t* out,
                         const vsla_tensor_t* a,
                         const vsla_tensor_t* b);

/**
 * @brief Element-wise multiplication (Hadamard product)
 */
vsla_error_t vsla_hadamard(vsla_context_t* ctx,
                           vsla_tensor_t* out,
                           const vsla_tensor_t* a,
                           const vsla_tensor_t* b);

/**
 * @brief Transpose a 2D tensor (matrix)
 */
vsla_error_t vsla_transpose(vsla_context_t* ctx,
                            vsla_tensor_t* out,
                            const vsla_tensor_t* in);

/**
 * @brief Reshape tensor while preserving total elements
 */
vsla_error_t vsla_reshape(vsla_context_t* ctx,
                          vsla_tensor_t* tensor,
                          uint8_t new_rank,
                          const uint64_t* new_shape);

// === Reduction Operations ===

/**
 * @brief Compute sum of all elements
 */
vsla_error_t vsla_sum(vsla_context_t* ctx,
                      const vsla_tensor_t* tensor,
                      double* result);

/**
 * @brief Compute mean of all elements
 */
vsla_error_t vsla_mean(vsla_context_t* ctx,
                       const vsla_tensor_t* tensor,
                       double* result);

/**
 * @brief Find maximum element
 */
vsla_error_t vsla_max(vsla_context_t* ctx,
                      const vsla_tensor_t* tensor,
                      double* result);

/**
 * @brief Find minimum element
 */
vsla_error_t vsla_min(vsla_context_t* ctx,
                      const vsla_tensor_t* tensor,
                      double* result);

/**
 * @brief Find index of maximum element
 */
vsla_error_t vsla_argmax(vsla_context_t* ctx,
                         const vsla_tensor_t* tensor,
                         uint64_t* index);

/**
 * @brief Find index of minimum element
 */
vsla_error_t vsla_argmin(vsla_context_t* ctx,
                         const vsla_tensor_t* tensor,
                         uint64_t* index);

/**
 * @brief Compute variance
 */
vsla_error_t vsla_variance(vsla_context_t* ctx,
                           const vsla_tensor_t* tensor,
                           double* result);

/**
 * @brief Compute standard deviation
 */
vsla_error_t vsla_std(vsla_context_t* ctx,
                      const vsla_tensor_t* tensor,
                      double* result);

/**
 * @brief Compute Frobenius norm
 */
vsla_error_t vsla_norm(vsla_context_t* ctx,
                       const vsla_tensor_t* tensor,
                       double* result);

// === Activation Functions ===

/**
 * @brief ReLU activation (max(0, x))
 */
vsla_error_t vsla_relu(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* in);

/**
 * @brief Sigmoid activation (1 / (1 + exp(-x)))
 */
vsla_error_t vsla_sigmoid(vsla_context_t* ctx,
                          vsla_tensor_t* out,
                          const vsla_tensor_t* in);

/**
 * @brief Tanh activation
 */
vsla_error_t vsla_tanh(vsla_context_t* ctx,
                       vsla_tensor_t* out,
                       const vsla_tensor_t* in);

/**
 * @brief Softmax activation along specified axis
 */
vsla_error_t vsla_softmax(vsla_context_t* ctx,
                          vsla_tensor_t* out,
                          const vsla_tensor_t* in,
                          int axis);

// === Broadcasting and Shape Operations ===

/**
 * @brief Broadcast tensors to compatible shape
 */
vsla_error_t vsla_broadcast(vsla_context_t* ctx,
                            vsla_tensor_t* out_a,
                            vsla_tensor_t* out_b,
                            const vsla_tensor_t* a,
                            const vsla_tensor_t* b);

/**
 * @brief Squeeze (remove dimensions of size 1)
 */
vsla_error_t vsla_squeeze(vsla_context_t* ctx,
                          vsla_tensor_t* out,
                          const vsla_tensor_t* in,
                          int axis);

/**
 * @brief Unsqueeze (add dimension of size 1)
 */
vsla_error_t vsla_unsqueeze(vsla_context_t* ctx,
                            vsla_tensor_t* out,
                            const vsla_tensor_t* in,
                            int axis);

/**
 * @brief Concatenate tensors along specified axis
 */
vsla_error_t vsla_concat(vsla_context_t* ctx,
                         vsla_tensor_t* out,
                         const vsla_tensor_t** tensors,
                         size_t count,
                         int axis);

/**
 * @brief Split tensor along specified axis
 */
vsla_error_t vsla_split(vsla_context_t* ctx,
                        vsla_tensor_t** outputs,
                        const vsla_tensor_t* in,
                        size_t split_count,
                        int axis);

// === Stacking Operations ===

/**
 * @brief Stack tensors along new leading axis (Σ operator)
 * 
 * Implements the mathematical stacking operator Σ_k: (𝕋_r)^k → 𝕋_{r+1}
 * that creates a rank-(r+1) tensor by stacking k rank-r tensors along a 
 * new leading axis.
 * 
 * Mathematical properties:
 * - Associativity (nested levels)
 * - Neutral-zero absorption
 * - Distributivity over +, ⊙
 * - Forms strict monoidal category (𝕋_r, +, Σ)
 * 
 * @param ctx VSLA context
 * @param out Output tensor of rank r+1
 * @param tensors Array of k input tensors (all rank r)
 * @param count Number of tensors to stack
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_stack(vsla_context_t* ctx,
                        vsla_tensor_t* out,
                        const vsla_tensor_t** tensors,
                        size_t count);

/**
 * @brief Create stacked tensor (convenience function)
 * 
 * Automatically determines output shape and allocates result tensor.
 * 
 * @param ctx VSLA context
 * @param tensors Array of input tensors
 * @param count Number of tensors to stack
 * @return New stacked tensor or NULL on error
 */
vsla_tensor_t* vsla_stack_create(vsla_context_t* ctx,
                                 const vsla_tensor_t** tensors,
                                 size_t count);

/**
 * @brief Unstack tensor along leading axis
 * 
 * Inverse of stacking. Splits rank-(r+1) tensor into k rank-r tensors.
 * 
 * @param ctx VSLA context
 * @param tensor Input tensor to unstack
 * @param outputs Array to receive unstacked tensors
 * @param max_outputs Size of outputs array
 * @param num_outputs Actual number of tensors produced
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_unstack(vsla_context_t* ctx,
                          const vsla_tensor_t* tensor,
                          vsla_tensor_t** outputs,
                          size_t max_outputs,
                          size_t* num_outputs);

// === Automatic Differentiation ===

/**
 * @brief Create gradient tape for automatic differentiation
 */
typedef struct vsla_tape vsla_tape_t;

vsla_tape_t* vsla_tape_create(vsla_context_t* ctx);

/**
 * @brief Free gradient tape
 */
void vsla_tape_free(vsla_tape_t* tape);

/**
 * @brief Enable gradient recording for tensor
 */
vsla_error_t vsla_tensor_requires_grad(vsla_tensor_t* tensor, bool requires_grad);

/**
 * @brief Perform backward pass from loss tensor
 */
vsla_error_t vsla_backward(vsla_context_t* ctx,
                           vsla_tape_t* tape,
                           const vsla_tensor_t* loss);

/**
 * @brief Get gradient for tensor
 */
vsla_tensor_t* vsla_get_gradient(const vsla_tape_t* tape,
                                 const vsla_tensor_t* tensor);

/**
 * @brief Clear all gradients
 */
vsla_error_t vsla_zero_grad(vsla_tape_t* tape);

// === Matrix Operations ===

/**
 * @brief Matrix inverse (2D tensors only)
 */
vsla_error_t vsla_inverse(vsla_context_t* ctx,
                          vsla_tensor_t* out,
                          const vsla_tensor_t* in);

/**
 * @brief LU decomposition
 */
vsla_error_t vsla_lu(vsla_context_t* ctx,
                     vsla_tensor_t* L,
                     vsla_tensor_t* U,
                     vsla_tensor_t* P,
                     const vsla_tensor_t* A);

/**
 * @brief QR decomposition
 */
vsla_error_t vsla_qr(vsla_context_t* ctx,
                     vsla_tensor_t* Q,
                     vsla_tensor_t* R,
                     const vsla_tensor_t* A);

/**
 * @brief Singular Value Decomposition
 */
vsla_error_t vsla_svd(vsla_context_t* ctx,
                      vsla_tensor_t* U,
                      vsla_tensor_t* S,
                      vsla_tensor_t* V,
                      const vsla_tensor_t* A);

// === Batch Operations ===

/**
 * @brief Execute multiple operations as a batch for efficiency
 * 
 * @param ctx VSLA context
 * @param ops Array of operation descriptors
 * @param count Number of operations
 * @return VSLA_SUCCESS if all operations succeed
 */
typedef struct {
    enum {
        VSLA_OP_ADD,
        VSLA_OP_SUB,
        VSLA_OP_SCALE,
        VSLA_OP_HADAMARD,
        VSLA_OP_CONV,
        VSLA_OP_MATMUL,
        VSLA_OP_TRANSPOSE,
        VSLA_OP_RELU,
        VSLA_OP_SIGMOID,
        VSLA_OP_TANH
    } type;
    vsla_tensor_t* out;
    const vsla_tensor_t* in1;
    const vsla_tensor_t* in2;
    double scalar;
} vsla_operation_t;

vsla_error_t vsla_batch_execute(vsla_context_t* ctx,
                                const vsla_operation_t* ops,
                                size_t count);

// === Performance and Profiling ===

/**
 * @brief Get performance statistics
 */
typedef struct {
    uint64_t total_operations;      /**< Total operations executed */
    uint64_t gpu_operations;        /**< Operations executed on GPU */
    uint64_t cpu_operations;        /**< Operations executed on CPU */
    double total_time_ms;           /**< Total execution time */
    double gpu_time_ms;             /**< GPU execution time */
    double cpu_time_ms;             /**< CPU execution time */
    double transfer_time_ms;        /**< CPU<->GPU transfer time */
    size_t memory_used_mb;          /**< Current memory usage */
    size_t peak_memory_mb;          /**< Peak memory usage */
} vsla_stats_t;

vsla_error_t vsla_get_stats(const vsla_context_t* ctx, vsla_stats_t* stats);

/**
 * @brief Reset performance statistics
 */
vsla_error_t vsla_reset_stats(vsla_context_t* ctx);

/**
 * @brief Synchronize all pending operations
 * 
 * Ensures all asynchronous operations are complete.
 */
vsla_error_t vsla_synchronize(vsla_context_t* ctx);

// === Utility Functions ===

/**
 * @brief Set optimization hint for subsequent operations
 */
vsla_error_t vsla_set_hint(vsla_context_t* ctx, vsla_hint_t hint);

/**
 * @brief Enable/disable automatic tensor migration between CPU/GPU
 */
vsla_error_t vsla_set_auto_migration(vsla_context_t* ctx, bool enable);

/**
 * @brief Prefetch tensor to optimal device for upcoming operations
 */
vsla_error_t vsla_tensor_prefetch(vsla_context_t* ctx, vsla_tensor_t* tensor);

/**
 * @brief Get recommended backend for given operation
 */
vsla_backend_t vsla_recommend_backend(vsla_context_t* ctx,
                                       const char* operation,
                                       const vsla_tensor_t** inputs,
                                       size_t input_count);

#ifdef __cplusplus
}
#endif

#endif // VSLA_UNIFIED_H