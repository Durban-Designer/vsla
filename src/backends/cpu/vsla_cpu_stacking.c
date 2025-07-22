/**
 * @file vsla_cpu_stacking.c
 * @brief VSLA CPU stacking operations following v3.1 specification
 * 
 * Implements Section 5: Structural Operators
 * - vsla_stack: Stack k tensors along new axis
 * - vsla_window_push: Window stacking with ring buffer
 * - Pyramid stacking functionality
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include "vsla/internal/vsla_window.h"
#include <stdlib.h>
#include <string.h>

// Ensure we have access to reference counting functions
extern vsla_tensor_t* vsla_tensor_retain(vsla_tensor_t* tensor);
extern vsla_error_t vsla_tensor_release(vsla_tensor_t* tensor);

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
                return VSLA_ERROR_INVALID_RANK;
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
    
    // Calculate required memory size
    size_t dtype_size = vsla_dtype_size(dtype);
    if (mul_ov(total_capacity, dtype_size)) {
        return VSLA_ERROR_OVERFLOW;
    }
    
    size_t total_bytes = total_capacity * dtype_size;
    
    // Check if we need to reallocate memory
    if (!out->data || out->data_size < total_bytes) {
        // Free existing memory if it exists
        if (out->data) {
            free(out->data);
        }
        
        // aligned_alloc requires size to be a multiple of alignment
        size_t aligned_size = ((total_bytes + 63) / 64) * 64; // Round up to multiple of 64
        out->data = aligned_alloc(64, aligned_size);
        if (!out->data) {
            return VSLA_ERROR_MEMORY;
        }
        out->data_size = total_bytes;
        out->cpu_data = out->data;  // Ensure cpu_data points to the same memory
    }
    
    // Zero-initialize the memory
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
 * @brief Create a new window for stacking - MODERN API WITH CONTEXT
 */
vsla_window_t* cpu_window_create(vsla_context_t* ctx, size_t window_size, uint8_t rank, vsla_dtype_t dtype) {
    if (!ctx || window_size == 0 || rank == 0) {
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
    window->ctx = ctx;  // Store context for unified tensor creation!
    
    return window;
}

/**
 * @brief Destroy a window and release all tensors
 */
void cpu_window_destroy(vsla_window_t* window) {
    if (!window) return;
    
    // Release any remaining tensors using reference counting
    for (size_t i = 0; i < window->fill; i++) {
        if (window->buf[i]) {
            vsla_tensor_release(window->buf[i]);
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
    
    // Add tensor to buffer and retain reference since we're storing a pointer to it
    window->buf[window->fill++] = vsla_tensor_retain(tensor);
    
    // Check if window is full
    if (window->fill == window->window_size) {
        // Determine output shape: (window_size, max_shape[0], max_shape[1], ...)
        uint64_t max_shape[VSLA_MAX_RANK];
        for (int j = 0; j < window->base_rank; j++) {
            max_shape[j] = 0;
            for (size_t i = 0; i < window->window_size; i++) {
                if (window->buf[i] && window->buf[i]->shape[j] > max_shape[j]) {
                    max_shape[j] = window->buf[i]->shape[j];
                }
            }
        }
        
        uint64_t out_shape[VSLA_MAX_RANK];
        out_shape[0] = window->window_size;
        for (int j = 0; j < window->base_rank; j++) {
            out_shape[j + 1] = max_shape[j];
        }
        
        // MODERN UNIFIED API - Use proper tensor creation!
        // This eliminates dual creation methods and ensures correctness
        vsla_tensor_t* result = vsla_tensor_create(window->ctx, window->base_rank + 1, out_shape, VSLA_MODEL_A, window->dtype);
        if (!result) {
            return NULL;
        }
        
        // Stack all tensors in the window
        vsla_error_t err = cpu_stack(result, (const vsla_tensor_t* const*)window->buf, window->window_size);
        if (err != VSLA_SUCCESS) {
            // Use reference counting system for cleanup
            vsla_tensor_release(result);
            return NULL;
        }
        
        // Release all tensors from the window buffer before resetting
        for (size_t i = 0; i < window->window_size; i++) {
            if (window->buf[i]) {
                vsla_tensor_release(window->buf[i]);
                window->buf[i] = NULL;
            }
        }
        window->fill = 0;
        
        return result;
    }
    
    return NULL; // Window not full yet
}

// === Pyramid Stacking Implementation (Section 5.2) ===

/**
 * @brief Create a pyramid with L levels of windows - MODERN API WITH CONTEXT
 * 
 * Following spec: "A pyramid is an array windows[L]; feed results recursively upward"
 */
vsla_pyramid_t* cpu_pyramid_create(vsla_context_t* ctx, size_t levels, size_t window_size, uint8_t rank, vsla_dtype_t dtype, bool discard_partials) {
    if (!ctx || levels == 0 || window_size == 0 || rank == 0) {
        return NULL;
    }
    
    vsla_pyramid_t* pyramid = malloc(sizeof(vsla_pyramid_t));
    if (!pyramid) {
        return NULL;
    }
    
    pyramid->windows = calloc(levels, sizeof(vsla_window_t*));
    if (!pyramid->windows) {
        free(pyramid);
        return NULL;
    }
    
    // Create windows for each level
    // Level 0: base rank, Level 1: rank+1, Level 2: rank+2, etc.
    for (size_t level = 0; level < levels; level++) {
        uint8_t level_rank = rank + level;  // Each level increases rank by 1
        pyramid->windows[level] = cpu_window_create(ctx, window_size, level_rank, dtype);
        if (!pyramid->windows[level]) {
            // Cleanup on failure
            for (size_t j = 0; j < level; j++) {
                cpu_window_destroy(pyramid->windows[j]);
            }
            free(pyramid->windows);
            free(pyramid);
            return NULL;
        }
    }
    
    pyramid->levels = levels;
    pyramid->window_size = window_size;
    pyramid->base_rank = rank;
    pyramid->dtype = dtype;
    pyramid->discard_partials = discard_partials;
    
    return pyramid;
}

/**
 * @brief Destroy pyramid and all its windows
 */
void cpu_pyramid_destroy(vsla_pyramid_t* pyramid) {
    if (!pyramid) return;
    
    if (pyramid->windows) {
        for (size_t i = 0; i < pyramid->levels; i++) {
            cpu_window_destroy(pyramid->windows[i]);
        }
        free(pyramid->windows);
    }
    
    free(pyramid);
}

/**
 * @brief Push tensor through pyramid levels recursively
 * 
 * Returns final output tensor if it emerges from top level, NULL otherwise
 */
vsla_tensor_t* cpu_pyramid_push(vsla_pyramid_t* pyramid, vsla_tensor_t* tensor) {
    if (!pyramid || !tensor) {
        return NULL;
    }
    
    // Validate input tensor
    if (tensor->rank != pyramid->base_rank || tensor->dtype != pyramid->dtype) {
        return NULL;
    }
    
    vsla_tensor_t* current = tensor;
    
    // Feed through each level of the pyramid
    for (size_t level = 0; level < pyramid->levels; level++) {
        if (!current) {
            break; // No tensor to feed to this level
        }
        
        // Push tensor to current level window
        vsla_tensor_t* result = cpu_window_push(pyramid->windows[level], current);
        
        // If we got a result from a previous level, release our reference to it
        // (except for the original input tensor which we don't own)
        if (current != tensor) {
            // Now that we have reference counting, properly release intermediate tensors
            vsla_tensor_release(current);
        }
        
        current = result; // Result becomes input for next level
    }
    
    // Return final result (may be NULL if no tensor emerged from top)
    return current;
}

/**
 * @brief Flush pyramid by forcing partial windows to emit
 * 
 * Returns array of tensors from all levels with partial data
 * Caller must free the returned array and manage tensor lifetimes
 */
vsla_tensor_t** cpu_pyramid_flush(vsla_pyramid_t* pyramid, size_t* count) {
    if (!pyramid || !count) {
        return NULL;
    }
    
    *count = 0;
    
    if (pyramid->discard_partials) {
        // Discard policy: just reset all windows without emitting
        for (size_t level = 0; level < pyramid->levels; level++) {
            pyramid->windows[level]->fill = 0;
        }
        return NULL;
    }
    
    // Pad policy: create tensors from partial windows
    vsla_tensor_t** results = malloc(pyramid->levels * sizeof(vsla_tensor_t*));
    if (!results) {
        return NULL;
    }
    
    size_t result_count = 0;
    
    for (size_t level = 0; level < pyramid->levels; level++) {
        vsla_window_t* window = pyramid->windows[level];
        
        if (window->fill > 0) {
            // Calculate output shape for the partial window stack
            uint64_t max_shape[VSLA_MAX_RANK];
            for (int j = 0; j < window->base_rank; j++) {
                max_shape[j] = 0;
                for (size_t i = 0; i < window->fill; i++) {
                    if (window->buf[i] && window->buf[i]->shape[j] > max_shape[j]) {
                        max_shape[j] = window->buf[i]->shape[j];
                    }
                }
            }
            
            uint64_t out_shape[VSLA_MAX_RANK];
            out_shape[0] = window->fill;
            for (int j = 0; j < window->base_rank; j++) {
                out_shape[j + 1] = max_shape[j];
            }
            
            // MODERN UNIFIED API - Use proper tensor creation with context!
            vsla_tensor_t* result = vsla_tensor_create(window->ctx, window->base_rank + 1, out_shape, VSLA_MODEL_A, window->dtype);
            if (result) {
                // Stack the partial window (remaining slots are effectively zero)
                vsla_error_t err = cpu_stack(result, (const vsla_tensor_t* const*)window->buf, window->fill);
                if (err == VSLA_SUCCESS) {
                    results[result_count++] = result;
                } else {
                    vsla_tensor_free(result);
                }
            }
            
            // Reset window
            window->fill = 0;
        }
    }
    
    *count = result_count;
    
    if (result_count == 0) {
        free(results);
        return NULL;
    }
    
    return results;
}

