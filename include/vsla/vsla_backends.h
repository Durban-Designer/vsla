/**
 * @file vsla_backends.h
 * @brief Backend interface definitions for VSLA
 * 
 * Defines the common interface that all compute backends must implement.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_BACKENDS_H
#define VSLA_BACKENDS_H

#include "vsla_core.h"
#include "vsla_unified.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backend interface structure
 * 
 * Each compute backend (CPU, CUDA, ROCm, oneAPI) implements this interface
 * to provide hardware-specific optimizations.
 */
typedef struct {
    const char* name;                   /**< Backend name */
    vsla_backend_t backend_type;        /**< Backend type enum */
    
    // Lifecycle management
    vsla_error_t (*init)(void);                     /**< Initialize backend */
    void (*cleanup)(void);                          /**< Cleanup backend */
    vsla_error_t (*get_info)(char*, size_t, size_t*, vsla_backend_t*); /**< Get device info */
    
    // Core tensor operations
    vsla_error_t (*add)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sub)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*scale)(vsla_tensor_t*, const vsla_tensor_t*, double);
    vsla_error_t (*hadamard)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*matmul)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*conv)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*transpose)(vsla_tensor_t*, const vsla_tensor_t*);
    
    // Activation functions
    vsla_error_t (*relu)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sigmoid)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*tanh_func)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*softmax)(vsla_tensor_t*, const vsla_tensor_t*, int);
    
    // Reduction operations
    vsla_error_t (*sum)(const vsla_tensor_t*, double*);
    vsla_error_t (*mean)(const vsla_tensor_t*, double*);
    vsla_error_t (*max)(const vsla_tensor_t*, double*);
    vsla_error_t (*min)(const vsla_tensor_t*, double*);
    vsla_error_t (*norm)(const vsla_tensor_t*, double*);
    vsla_error_t (*argmax)(const vsla_tensor_t*, uint64_t*);
    vsla_error_t (*argmin)(const vsla_tensor_t*, uint64_t*);
    
    // Matrix operations
    vsla_error_t (*inverse)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*lu)(vsla_tensor_t*, vsla_tensor_t*, vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*qr)(vsla_tensor_t*, vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*svd)(vsla_tensor_t*, vsla_tensor_t*, vsla_tensor_t*, const vsla_tensor_t*);
    
    // Shape operations
    vsla_error_t (*reshape)(vsla_tensor_t*, uint8_t, const uint64_t*);
    vsla_error_t (*squeeze)(vsla_tensor_t*, const vsla_tensor_t*, int);
    vsla_error_t (*unsqueeze)(vsla_tensor_t*, const vsla_tensor_t*, int);
    vsla_error_t (*concat)(vsla_tensor_t*, const vsla_tensor_t**, size_t, int);
    vsla_error_t (*split)(vsla_tensor_t**, const vsla_tensor_t*, size_t, int);
    
} vsla_backend_interface_t;

// === Backend Registry Functions ===

/**
 * @brief Initialize the backend registry
 * 
 * Discovers and initializes all available compute backends.
 * 
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_backend_registry_init(void);

/**
 * @brief Cleanup the backend registry
 * 
 * Releases resources for all registered backends.
 */
void vsla_backend_registry_cleanup(void);

/**
 * @brief Get the best available backend
 * 
 * @param preferred Preferred backend type (VSLA_BACKEND_AUTO for automatic)
 * @return Backend interface or NULL if none available
 */
const vsla_backend_interface_t* vsla_backend_get_best(vsla_backend_t preferred);

/**
 * @brief Get backend by specific type
 * 
 * @param backend_type Backend type to retrieve
 * @return Backend interface or NULL if not available
 */
const vsla_backend_interface_t* vsla_backend_get_by_type(vsla_backend_t backend_type);

/**
 * @brief List all available backends
 * 
 * @param backends Output array for backend types
 * @param count Output for number of backends
 * @param max_count Maximum number of backends to return
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_backend_list_available(vsla_backend_t* backends, 
                                         size_t* count, 
                                         size_t max_count);

/**
 * @brief Get information about a specific backend
 * 
 * @param backend_type Backend to query
 * @param name Output buffer for device name
 * @param name_size Size of name buffer
 * @param memory_gb Output for available memory in GB
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_backend_get_info(vsla_backend_t backend_type,
                                   char* name, size_t name_size,
                                   size_t* memory_gb);

/**
 * @brief Check if backend supports specific operation
 * 
 * @param backend_type Backend to check
 * @param operation Operation name (e.g., "matmul", "conv")
 * @return true if supported, false otherwise
 */
bool vsla_backend_supports_operation(vsla_backend_t backend_type, const char* operation);

/**
 * @brief Recommend best backend for operation
 * 
 * Uses heuristics based on tensor sizes and operation type to recommend
 * the optimal backend for performance.
 * 
 * @param operation Operation name
 * @param tensors Array of input tensors
 * @param tensor_count Number of tensors
 * @return Recommended backend type
 */
vsla_backend_t vsla_backend_recommend(const char* operation,
                                      const vsla_tensor_t** tensors,
                                      size_t tensor_count);

// === Individual Backend Accessors ===

/**
 * @brief Get CPU backend interface
 * @return CPU backend interface (always available)
 */
const vsla_backend_interface_t* vsla_get_cpu_backend(void);

/**
 * @brief Get CUDA backend interface
 * @return CUDA backend interface or NULL if not available
 */
const vsla_backend_interface_t* vsla_get_cuda_backend(void);

/**
 * @brief Get ROCm backend interface
 * @return ROCm backend interface or NULL if not available
 */
const vsla_backend_interface_t* vsla_get_rocm_backend(void);

/**
 * @brief Get oneAPI backend interface
 * @return oneAPI backend interface or NULL if not available
 */
const vsla_backend_interface_t* vsla_get_oneapi_backend(void);

#ifdef __cplusplus
}
#endif

#endif // VSLA_BACKENDS_H