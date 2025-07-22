/**
 * @file vsla_cpu_memory.c
 * @brief VSLA CPU memory management following v3.1 specification
 * 
 * Implements memory allocation, deallocation, and data movement
 * Based on Section 2 invariants and requirements
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <stdlib.h>
#include <string.h>

// Platform-specific includes for aligned allocation
#ifdef _WIN32
#include <malloc.h>
#else
#include <unistd.h>
#endif

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_capacity_elems(const vsla_tensor_t* t);
extern bool mul_ov(uint64_t a, uint64_t b); // Defined in helpers

/**
 * @brief Allocate memory for tensor data following Section 2.2 invariants
 */
vsla_error_t cpu_allocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Calculate allocation size: bytes = sizeof(dtype) * product(capacity[i])
    uint64_t capacity_elems = vsla_capacity_elems(tensor);
    
    // Handle empty tensors - set data = NULL as per invariant
    if (capacity_elems == 0) {
        tensor->data = NULL;
        tensor->data_size = 0;
        tensor->cpu_valid = true;
        tensor->location = VSLA_BACKEND_CPU;
        return VSLA_SUCCESS;
    }
    
    // Check for overflow
    if (capacity_elems == UINT64_MAX) {
        return VSLA_ERROR_OVERFLOW;
    }
    
    size_t dtype_size = vsla_dtype_size(tensor->dtype);
    if (mul_ov(capacity_elems, dtype_size)) {
        return VSLA_ERROR_OVERFLOW;
    }
    
    size_t total_bytes = capacity_elems * dtype_size;
    
    // Allocate aligned memory for SIMD (Section 8 roadmap)
    void* data = aligned_alloc(64, total_bytes);
    if (!data) {
        return VSLA_ERROR_MEMORY;
    }
    
    // Zero initialization for logical region (Section 2.2 invariant)
    // Slack region is uninitialized
    uint64_t logical_elems = vsla_logical_elems(tensor);
    if (logical_elems > 0) {
        memset(data, 0, logical_elems * dtype_size);
    }
    
    tensor->data = data;
    tensor->data_size = total_bytes;
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

/**
 * @brief Deallocate tensor memory
 */
vsla_error_t cpu_deallocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
        tensor->data_size = 0;
    }
    
    tensor->cpu_valid = false;
    tensor->gpu_valid = false;
    
    return VSLA_SUCCESS;
}

/**
 * @brief Copy tensor data to device (no-op for CPU)
 */
vsla_error_t cpu_copy_to_device(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For CPU backend, this is a no-op
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

/**
 * @brief Copy tensor data to host (no-op for CPU)
 */
vsla_error_t cpu_copy_to_host(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For CPU backend, this is a no-op
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

/**
 * @brief Synchronize device operations (no-op for CPU)
 */
vsla_error_t cpu_synchronize(void) {
    // For CPU backend, this is a no-op
    return VSLA_SUCCESS;
}

/**
 * @brief Fill tensor with a constant value
 */
vsla_error_t cpu_fill(vsla_tensor_t* tensor, double value) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (vsla_is_empty(tensor) || !tensor->data) {
        return VSLA_SUCCESS; // Nothing to fill
    }
    
    uint64_t logical_elems = vsla_logical_elems(tensor);
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < logical_elems; i++) {
            data[i] = value;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        float val_f32 = (float)value;
        for (uint64_t i = 0; i < logical_elems; i++) {
            data[i] = val_f32;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

// Platform-specific aligned allocation fallback
#ifdef _WIN32
static void* aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
#elif !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
// Fallback for older C standards
static void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    int result = posix_memalign(&ptr, alignment, size);
    return (result == 0) ? ptr : NULL;
}
#endif