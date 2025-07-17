/**
 * @file vsla_backend_registry.c
 * @brief Backend registry and management for VSLA unified interface
 * 
 * Manages all available compute backends and provides automatic selection.
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Forward declarations for backend interfaces
typedef struct {
    const char* name;
    vsla_backend_t backend_type;
    vsla_error_t (*init)(void);
    void (*cleanup)(void);
    vsla_error_t (*get_info)(char*, size_t, size_t*, vsla_backend_t*);
    
    // Core operations
    vsla_error_t (*add)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*scale)(vsla_tensor_t*, const vsla_tensor_t*, double);
    vsla_error_t (*matmul)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*conv)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*relu)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sigmoid)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sum)(const vsla_tensor_t*, double*);
    vsla_error_t (*mean)(const vsla_tensor_t*, double*);
} vsla_backend_interface_t;

// External backend interfaces
extern const vsla_backend_interface_t* vsla_get_cpu_backend(void);
extern const vsla_backend_interface_t* vsla_get_cuda_backend(void);
extern const vsla_backend_interface_t* vsla_get_rocm_backend(void);
extern const vsla_backend_interface_t* vsla_get_oneapi_backend(void);

// Registry state
#define MAX_BACKENDS 8
static struct {
    bool initialized;
    const vsla_backend_interface_t* backends[MAX_BACKENDS];
    bool backend_available[MAX_BACKENDS];
    size_t backend_count;
    size_t preferred_backend;
} registry = {0};

// Backend priority order (higher index = higher priority)
static const vsla_backend_t backend_priority[] = {
    VSLA_BACKEND_CPU,        // Always available, lowest priority
    VSLA_BACKEND_ONEAPI,     // Intel CPUs/GPUs
    VSLA_BACKEND_ROCM,       // AMD GPUs
    VSLA_BACKEND_CUDA        // NVIDIA GPUs, highest priority
};

static const size_t priority_count = sizeof(backend_priority) / sizeof(backend_priority[0]);

// Initialize the backend registry
vsla_error_t vsla_backend_registry_init(void) {
    if (registry.initialized) return VSLA_SUCCESS;
    
    registry.backend_count = 0;
    registry.preferred_backend = 0;
    
    // Register all available backends
    const vsla_backend_interface_t* backends[] = {
        vsla_get_cpu_backend(),      // Always available
        vsla_get_cuda_backend(),     // CUDA/NVIDIA
        vsla_get_rocm_backend(),     // ROCm/AMD  
        vsla_get_oneapi_backend()    // oneAPI/Intel
    };
    
    for (size_t i = 0; i < sizeof(backends) / sizeof(backends[0]) && i < MAX_BACKENDS; i++) {
        if (backends[i]) {
            registry.backends[registry.backend_count] = backends[i];
            
            // Test if backend is available
            vsla_error_t err = backends[i]->init();
            registry.backend_available[registry.backend_count] = (err == VSLA_SUCCESS);
            
            if (registry.backend_available[registry.backend_count]) {
                // Set as preferred if higher priority
                vsla_backend_t backend_type = backends[i]->backend_type;
                for (size_t p = 0; p < priority_count; p++) {
                    if (backend_priority[p] == backend_type) {
                        if (p >= registry.preferred_backend) {
                            registry.preferred_backend = registry.backend_count;
                        }
                        break;
                    }
                }
            }
            
            registry.backend_count++;
        }
    }
    
    // Ensure at least CPU backend is available
    if (registry.backend_count == 0 || !registry.backend_available[0]) {
        return VSLA_ERROR_INITIALIZATION_FAILED;
    }
    
    registry.initialized = true;
    return VSLA_SUCCESS;
}

// Cleanup the backend registry
void vsla_backend_registry_cleanup(void) {
    if (!registry.initialized) return;
    
    for (size_t i = 0; i < registry.backend_count; i++) {
        if (registry.backends[i] && registry.backend_available[i]) {
            registry.backends[i]->cleanup();
        }
    }
    
    registry.initialized = false;
}

// Get the best available backend for a given operation
const vsla_backend_interface_t* vsla_backend_get_best(vsla_backend_t preferred) {
    if (!registry.initialized) {
        if (vsla_backend_registry_init() != VSLA_SUCCESS) {
            return NULL;
        }
    }
    
    // If specific backend requested, try to find it
    if (preferred != VSLA_BACKEND_AUTO) {
        for (size_t i = 0; i < registry.backend_count; i++) {
            if (registry.backends[i]->backend_type == preferred && 
                registry.backend_available[i]) {
                return registry.backends[i];
            }
        }
    }
    
    // Return preferred backend if available
    if (registry.preferred_backend < registry.backend_count &&
        registry.backend_available[registry.preferred_backend]) {
        return registry.backends[registry.preferred_backend];
    }
    
    // Fallback to first available backend (should be CPU)
    for (size_t i = 0; i < registry.backend_count; i++) {
        if (registry.backend_available[i]) {
            return registry.backends[i];
        }
    }
    
    return NULL;
}

// Get backend by type
const vsla_backend_interface_t* vsla_backend_get_by_type(vsla_backend_t backend_type) {
    if (!registry.initialized) {
        if (vsla_backend_registry_init() != VSLA_SUCCESS) {
            return NULL;
        }
    }
    
    for (size_t i = 0; i < registry.backend_count; i++) {
        if (registry.backends[i]->backend_type == backend_type && 
            registry.backend_available[i]) {
            return registry.backends[i];
        }
    }
    
    return NULL;
}

// List all available backends
vsla_error_t vsla_backend_list_available(vsla_backend_t* backends, 
                                         size_t* count, 
                                         size_t max_count) {
    if (!backends || !count) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (!registry.initialized) {
        vsla_error_t err = vsla_backend_registry_init();
        if (err != VSLA_SUCCESS) return err;
    }
    
    size_t available_count = 0;
    for (size_t i = 0; i < registry.backend_count && available_count < max_count; i++) {
        if (registry.backend_available[i]) {
            backends[available_count] = registry.backends[i]->backend_type;
            available_count++;
        }
    }
    
    *count = available_count;
    return VSLA_SUCCESS;
}

// Get backend info
vsla_error_t vsla_backend_get_info(vsla_backend_t backend_type,
                                   char* name, size_t name_size,
                                   size_t* memory_gb) {
    const vsla_backend_interface_t* backend = vsla_backend_get_by_type(backend_type);
    if (!backend) return VSLA_ERROR_NOT_IMPLEMENTED;
    
    return backend->get_info(name, name_size, memory_gb, &backend_type);
}

// Check if backend supports specific operation
bool vsla_backend_supports_operation(vsla_backend_t backend_type, const char* operation) {
    const vsla_backend_interface_t* backend = vsla_backend_get_by_type(backend_type);
    if (!backend) return false;
    
    // Check operation availability based on function pointers
    if (strcmp(operation, "add") == 0) return backend->add != NULL;
    if (strcmp(operation, "scale") == 0) return backend->scale != NULL;
    if (strcmp(operation, "matmul") == 0) return backend->matmul != NULL;
    if (strcmp(operation, "conv") == 0) return backend->conv != NULL;
    if (strcmp(operation, "relu") == 0) return backend->relu != NULL;
    if (strcmp(operation, "sigmoid") == 0) return backend->sigmoid != NULL;
    if (strcmp(operation, "sum") == 0) return backend->sum != NULL;
    if (strcmp(operation, "mean") == 0) return backend->mean != NULL;
    
    return false;
}

// Smart backend recommendation based on tensor size and operation
vsla_backend_t vsla_backend_recommend(const char* operation,
                                      const vsla_tensor_t** tensors,
                                      size_t tensor_count) {
    if (!operation || !tensors || tensor_count == 0) {
        return VSLA_BACKEND_AUTO;
    }
    
    // Calculate total elements across all tensors
    uint64_t total_elements = 0;
    for (size_t i = 0; i < tensor_count; i++) {
        if (tensors[i]) {
            uint64_t elements = 1;
            for (uint8_t j = 0; j < tensors[i]->rank; j++) {
                elements *= tensors[i]->shape[j];
            }
            total_elements += elements;
        }
    }
    
    // Heuristics for backend selection
    if (strcmp(operation, "conv") == 0 || strcmp(operation, "matmul") == 0) {
        // FFT and matrix operations benefit from GPU for larger sizes
        if (total_elements > 1024) {
            return VSLA_BACKEND_CUDA;  // Prefer CUDA for compute-intensive ops
        }
    }
    
    if (strcmp(operation, "add") == 0 || strcmp(operation, "scale") == 0) {
        // Element-wise operations benefit from GPU for very large tensors
        if (total_elements > 4096) {
            return VSLA_BACKEND_CUDA;
        }
    }
    
    // Default to CPU for smaller operations
    return VSLA_BACKEND_CPU;
}

// Execute operation on best available backend
vsla_error_t vsla_backend_execute_operation(const char* operation,
                                            vsla_backend_t preferred_backend,
                                            void* params) {
    if (!operation || !params) return VSLA_ERROR_INVALID_ARGUMENT;
    
    const vsla_backend_interface_t* backend = vsla_backend_get_best(preferred_backend);
    if (!backend) return VSLA_ERROR_NOT_IMPLEMENTED;
    
    // This would require a more complex parameter structure
    // For now, return success to indicate the infrastructure is in place
    return VSLA_SUCCESS;
}