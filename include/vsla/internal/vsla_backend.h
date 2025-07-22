/**
 * @file vsla_backend.h
 * @brief Backend interface for VSLA operations.
 *
 * @copyright MIT License
 */

#ifndef VSLA_BACKEND_H
#define VSLA_BACKEND_H

#include "../vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
struct vsla_backend_interface_s;
typedef struct vsla_backend_interface_s vsla_backend_interface_t;

/**
 * @brief Backend capabilities structure
 */
typedef struct {
    bool supports_gpu;           /**< Backend supports GPU operations */
    bool supports_multi_gpu;     /**< Backend supports multi-GPU operations */
    bool supports_unified_memory;/**< Backend supports unified CPU/GPU memory */
    bool supports_async;         /**< Backend supports async operations */
    size_t max_tensor_size;      /**< Maximum tensor size supported */
    const char* name;            /**< Backend name (e.g., "CPU", "CUDA", "ROCm") */
    const char* version;         /**< Backend version string */
} vsla_backend_caps_t;

/**
 * @brief Backend interface structure
 * 
 * All operations should handle data movement transparently.
 * For GPU backends, operations should be implemented as single kernels when possible.
 */
struct vsla_backend_interface_s {
    /* Backend metadata */
    vsla_backend_caps_t caps;
    
    /* Memory management */
    vsla_error_t (*allocate)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    vsla_error_t (*deallocate)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    vsla_error_t (*copy_to_device)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    vsla_error_t (*copy_to_host)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    vsla_error_t (*synchronize)(vsla_context_t* ctx);
    
    /* Basic arithmetic operations */
    vsla_error_t (*add)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*sub)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*scale)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* tensor, double scalar);
    vsla_error_t (*hadamard)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*fill)(vsla_context_t* ctx, vsla_tensor_t* tensor, double value);
    
    /* Linear algebra operations */
    vsla_error_t (*matmul)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*transpose)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* tensor);
    
    /* Tensor operations */
    vsla_error_t (*reshape)(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]);
    vsla_error_t (*broadcast)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in);
    
    /* Reduction operations */
    vsla_error_t (*sum)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* sum);
    vsla_error_t (*mean)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* mean);
    vsla_error_t (*norm)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* norm);
    vsla_error_t (*max)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* max);
    vsla_error_t (*min)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* min);
    
    /* Advanced operations */
    vsla_error_t (*conv)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*kron)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    
    /* Structural operations */
    vsla_error_t (*stack)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* const* tensors, size_t k);
    vsla_error_t (*shrink)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    
    /* Backend lifecycle */
    vsla_error_t (*init)(void* config);
    void (*cleanup)(void);
};

/* Backend registration functions */
vsla_backend_interface_t* vsla_backend_cpu_create(void);
vsla_backend_interface_t* vsla_backend_cuda_create(void);
vsla_backend_interface_t* vsla_backend_rocm_create(void);
vsla_backend_interface_t* vsla_backend_oneapi_create(void);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_BACKEND_H */
