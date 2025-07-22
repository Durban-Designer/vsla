#include "vsla/vsla_core.h"
#include "vsla/vsla_tensor.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <string.h>
#include <stdatomic.h>
#include <stdlib.h>

// Note: vsla_get_f64 and vsla_set_f64 are implemented in vsla_unified.c

uint8_t vsla_get_rank(const vsla_tensor_t* tensor) {
    return tensor->rank;
}

vsla_error_t vsla_get_shape(const vsla_tensor_t* tensor, uint64_t* shape) {
    if (!tensor || !shape) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    memcpy(shape, tensor->shape, tensor->rank * sizeof(uint64_t));
    return VSLA_SUCCESS;
}

vsla_model_t vsla_get_model(const vsla_tensor_t* tensor) {
    return tensor->model;
}

vsla_dtype_t vsla_get_dtype(const vsla_tensor_t* tensor) {
    return tensor->dtype;
}

size_t vsla_numel(const vsla_tensor_t* tensor) {
    size_t numel = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        numel *= tensor->shape[i];
    }
    return numel;
}

int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (a->rank != b->rank) {
        return 0;
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

vsla_backend_t vsla_get_location(const vsla_tensor_t* tensor) {
    return tensor->location;
}

// vsla_dtype_size is defined in vsla_core.c

/**
 * @brief Increment tensor reference count (thread-safe)
 */
vsla_tensor_t* vsla_tensor_retain(vsla_tensor_t* tensor) {
    if (!tensor) {
        return NULL;
    }
    
    // Use atomic increment for thread safety
    atomic_fetch_add((_Atomic int32_t*)&tensor->ref_count, 1);
    
    return tensor;
}

/**
 * @brief Decrement tensor reference count and free if zero (thread-safe)
 */
vsla_error_t vsla_tensor_release(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_SUCCESS;
    }
    
    // Atomic decrement and check if we're the last reference
    int32_t old_count = atomic_fetch_sub((_Atomic int32_t*)&tensor->ref_count, 1);
    
    if (old_count == 1) {
        // We were the last reference, free the tensor
        
        // Free data using backend deallocation if we own it
        if (tensor->owns_data) {
            // Use the backend's deallocation function if available
            // This ensures proper cleanup for aligned_alloc, CUDA memory, etc.
            if (tensor->location == VSLA_BACKEND_CPU && tensor->data) {
                // For CPU backend, use the proper deallocator
                extern vsla_error_t cpu_deallocate(vsla_tensor_t* tensor);
                cpu_deallocate(tensor);
            }
            // TODO: Add GPU memory cleanup when backends support it
            // else if (tensor->location == VSLA_BACKEND_CUDA && tensor->gpu_data) {
            //     backend_free_gpu_memory(tensor->gpu_data);
            // }
            else if (tensor->cpu_data) {
                // Fallback for unknown backends
                free(tensor->cpu_data);
            }
        }
        
        // Free metadata arrays
        if (tensor->shape) free(tensor->shape);
        if (tensor->cap) free(tensor->cap);
        if (tensor->stride) free(tensor->stride);
        
        // Free the tensor structure itself
        free(tensor);
    }
    
    return VSLA_SUCCESS;
}