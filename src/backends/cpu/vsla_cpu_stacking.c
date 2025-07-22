/**
 * @file vsla_cpu_stacking.c
 * @brief VSLA CPU stacking operations following v3.1 specification
 * 
 * Implements Section 5: Structural Operators
 * - vsla_stack: Stack k tensors along new axis
 * - vsla_window_push: Window stacking with ring buffer
 * - Pyramid stacking functionality
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <stdlib.h>
#include <string.h>

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx);
extern void compute_strides(const vsla_tensor_t* t, uint64_t* s);
extern bool mul_ov(uint64_t a, uint64_t b);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);

/**
 * @brief Stack k tensors along a new axis following Section 5.1
 * 
 * Creates output with rank r+1, shape (k, A0, ..., A_{r-1}) where A[j] is the
 * ambient per-axis maximum. Each input tensor is copied into its slice.
 */
vsla_error_t cpu_stack(vsla_tensor_t* out, const vsla_tensor_t* const* tensors, size_t k) {
    if (!out || !tensors) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (k == 0) {
        // Empty stack - create empty tensor with rank 1
        out->rank = 1;
        out->shape[0] = 0;
        out->cap[0] = 0;
        out->data = NULL;
        out->data_size = 0;
        return VSLA_SUCCESS;
    }
    
    // Find the first non-null tensor to determine base properties
    const vsla_tensor_t* base = NULL;
    for (size_t i = 0; i < k; i++) {
        if (tensors[i]) {
            base = tensors[i];
            break;
        }
    }
    
    if (!base) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint8_t base_rank = base->rank;
    vsla_dtype_t dtype = base->dtype;
    
    // Validate all tensors have same rank and dtype
    for (size_t i = 0; i < k; i++) {
        if (tensors[i]) {
            if (tensors[i]->rank != base_rank) {
                return VSLA_ERROR_RANK;
            }
            if (tensors[i]->dtype != dtype) {
                return VSLA_ERROR_INVALID_DTYPE;
            }
        }
    }
    
    // Compute ambient shape (maximum along each axis)
    uint64_t ambient_shape[VSLA_MAX_RANK];
    for (int j = 0; j < base_rank; j++) {
        ambient_shape[j] = 0;
        for (size_t i = 0; i < k; i++) {
            if (tensors[i] && tensors[i]->shape[j] > ambient_shape[j]) {
                ambient_shape[j] = tensors[i]->shape[j];
            }
        }
    }
    
    // Check for overflow in stacking dimension
    if (k > UINT64_MAX) {
        return VSLA_ERROR_OVERFLOW;
    }
    
    // Set up output tensor: rank r+1, shape (k, A0, ..., A_{r-1})
    out->rank = base_rank + 1;
    out->dtype = dtype;
    out->shape[0] = (uint64_t)k;
    out->cap[0] = (uint64_t)k;
    
    for (int j = 0; j < base_rank; j++) {
        out->shape[j + 1] = ambient_shape[j];
        out->cap[j + 1] = ambient_shape[j];
    }
    
    // Calculate total capacity and check for overflow
    uint64_t total_capacity = 1;
    for (int j = 0; j < out->rank; j++) {
        if (mul_ov(total_capacity, out->cap[j])) {
            return VSLA_ERROR_OVERFLOW;
        }
        total_capacity *= out->cap[j];
    }
    
    // Allocate and zero-initialize output
    size_t dtype_size = vsla_dtype_size(dtype);
    if (mul_ov(total_capacity, dtype_size)) {
        return VSLA_ERROR_OVERFLOW;
    }
    
    size_t total_bytes = total_capacity * dtype_size;
    out->data = aligned_alloc(64, total_bytes);
    if (!out->data) {
        return VSLA_ERROR_MEMORY;
    }
    out->data_size = total_bytes;
    memset(out->data, 0, total_bytes);
    
    // Copy each input tensor into its corresponding slice
    uint64_t out_strides[VSLA_MAX_RANK];
    compute_strides(out, out_strides);
    
    for (size_t i = 0; i < k; i++) {
        const vsla_tensor_t* src = tensors[i];
        if (!src || vsla_is_empty(src) || !src->data) {
            continue; // Skip empty tensors (slice remains zero)
        }
        
        // Compute source strides
        uint64_t src_strides[VSLA_MAX_RANK];
        compute_strides(src, src_strides);
        
        // Copy logical region of source into slice i
        uint64_t src_elems = vsla_logical_elems(src);
        
        for (uint64_t linear_idx = 0; linear_idx < src_elems; linear_idx++) {
            // Unravel source linear index to multi-dimensional index
            uint64_t src_idx[VSLA_MAX_RANK];
            uint64_t temp = linear_idx;
            for (int j = src->rank - 1; j >= 0; j--) {
                src_idx[j] = temp % src->shape[j];
                temp /= src->shape[j];
            }
            
            // Map to output index: (i, src_idx[0], src_idx[1], ...)
            uint64_t out_idx[VSLA_MAX_RANK];
            out_idx[0] = i;
            for (int j = 0; j < src->rank; j++) {
                out_idx[j + 1] = src_idx[j];
            }
            
            // Copy the value
            uint64_t src_offset = 0;
            for (int j = 0; j < src->rank; j++) {
                src_offset += src_idx[j] * src_strides[j];
            }
            
            uint64_t out_offset = 0;
            for (int j = 0; j < out->rank; j++) {
                out_offset += out_idx[j] * out_strides[j];
            }
            
            if (dtype == VSLA_DTYPE_F64) {
                ((double*)out->data)[out_offset] = ((double*)src->data)[src_offset];
            } else if (dtype == VSLA_DTYPE_F32) {
                ((float*)out->data)[out_offset] = ((float*)src->data)[src_offset];
            }
        }
    }
    
    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

/**
 * @brief Create a new window for stacking
 */
vsla_window_t* cpu_window_create(size_t window_size, uint8_t rank, vsla_dtype_t dtype) {
    if (window_size == 0 || rank == 0) {
        return NULL;
    }
    
    vsla_window_t* window = malloc(sizeof(vsla_window_t));
    if (!window) {
        return NULL;
    }
    
    window->buf = calloc(window_size, sizeof(vsla_tensor_t*));
    if (!window->buf) {
        free(window);
        return NULL;
    }
    
    window->fill = 0;
    window->window_size = window_size;
    window->base_rank = rank;
    window->dtype = dtype;
    
    return window;
}

/**
 * @brief Destroy a window and release all tensors
 */
void cpu_window_destroy(vsla_window_t* window) {
    if (!window) return;
    
    // Release any remaining tensors
    for (size_t i = 0; i < window->fill; i++) {
        if (window->buf[i]) {
            // Note: This would call vsla_release if we had reference counting
            // For now, we assume caller manages tensor lifetime
        }
    }
    
    free(window->buf);
    free(window);
}

/**
 * @brief Push tensor to window, returning stacked result when window is full
 * 
 * Following Section 5.2: maintains ring buffer collecting w tensors then emitting S_w
 */
vsla_tensor_t* cpu_window_push(vsla_window_t* window, vsla_tensor_t* tensor) {
    if (!window || !tensor) {
        return NULL;
    }
    
    // Validate tensor compatibility
    if (tensor->rank != window->base_rank || tensor->dtype != window->dtype) {
        return NULL;
    }
    
    // Add tensor to buffer (would do vsla_retain if we had reference counting)
    window->buf[window->fill++] = tensor;
    
    // Check if window is full
    if (window->fill == window->window_size) {
        // Create output tensor for stacking
        vsla_tensor_t* result = malloc(sizeof(vsla_tensor_t));
        if (!result) {
            return NULL;
        }
        
        // Stack all tensors in the window
        vsla_error_t err = cpu_stack(result, (const vsla_tensor_t* const*)window->buf, window->window_size);
        if (err != VSLA_SUCCESS) {
            free(result);
            return NULL;
        }
        
        // Reset window for next batch (would do vsla_release if we had reference counting)
        window->fill = 0;
        
        return result;
    }
    
    return NULL; // Window not full yet
}

// Platform-specific aligned allocation fallback (if not already defined)
#ifdef _WIN32
#include <malloc.h>
static void* aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
#elif !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
#include <unistd.h>
static void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    int result = posix_memalign(&ptr, alignment, size);
    return (result == 0) ? ptr : NULL;
}
#endif