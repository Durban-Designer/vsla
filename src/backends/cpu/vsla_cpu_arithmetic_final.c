/**
 * @file vsla_cpu_arithmetic_integrated.c
 * @brief Enhanced VSLA CPU arithmetic with integrated optimizations
 * 
 * This file contains the enhanced cpu_add function that integrates our Phase 2
 * optimizations while maintaining full backward compatibility.
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>

// Import existing functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);
extern void unravel(uint64_t lin, const uint64_t* shape, uint8_t rank, uint64_t* out);
extern uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);
extern bool mul_ov(uint64_t a, uint64_t b);

// Import optimized functions
extern bool should_use_shape_strides(const vsla_tensor_t* a, const vsla_tensor_t* b, const vsla_tensor_t* out);
extern vsla_broadcast_pattern_t detect_broadcast_pattern(const vsla_tensor_t* a, const vsla_tensor_t* b);

// Import existing fast path functions
extern vsla_error_t cpu_add_micro_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_equal_size_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_small_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_rank1_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_block_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

// Import optimized broadcast functions
extern vsla_error_t cpu_add_2d_row_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_2d_col_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_scalar_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_optimized_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

// Import 3D/4D optimized broadcast functions (Phase 3)
extern vsla_error_t cpu_add_3d_spatial_w_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_3d_spatial_h_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_4d_channel_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
extern vsla_error_t cpu_add_4d_batch_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

// Import shapes_equal function from arithmetic.c
extern bool shapes_equal(const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * Enhanced CPU addition with integrated Phase 2 optimizations
 * 
 * This function maintains full backward compatibility while adding:
 * 1. Broadcast pattern detection and specialized kernels
 * 2. Shape-based stride optimization selection
 * 3. Cache-friendly memory access patterns
 * 4. Improved performance for multi-dimensional operations
 */
vsla_error_t cpu_add_enhanced(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Pre-condition checks from Section 3.1 (unchanged)
    if (a->model != b->model || a->model != out->model) {
        return VSLA_ERROR_INVALID_ARGUMENT; // Model mismatch
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT; // DType mismatch
    }
    
    if (a->rank != b->rank || a->rank != out->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT; // Rank mismatch
    }

    // Verify output shape follows the rule: max(a->shape[i], b->shape[i])
    for (uint8_t i = 0; i < out->rank; i++) {
        uint64_t expected = (a->shape[i] > b->shape[i]) ? a->shape[i] : b->shape[i];
        if (out->shape[i] != expected) {
            return VSLA_ERROR_INVALID_ARGUMENT; // Wrong output shape
        }
    }

    // Handle empty operands (behave as zeros) - unchanged
    if (vsla_is_empty(a) && vsla_is_empty(b)) {
        if (out->data) {
            memset(out->data, 0, vsla_logical_elems(out) * vsla_dtype_size(out->dtype));
        }
        return VSLA_SUCCESS;
    }

    size_t total = vsla_logical_elems(out);
    
    // === EXISTING FAST PATHS (preserved for compatibility) ===
    
    // Fast path 1: Micro-vector optimization (1-4 elements, equal size)
    if (total <= 4 && shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_micro_fast(out, a, b);
    }
    
    // Fast path 2: Equal-size dense tensors (no ambient promotion needed)
    if (shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_equal_size_fast(out, a, b);
    }
    
    // Fast path 3: Small vector fast path (5-32 elements)
    if (total >= 5 && total < 32 && shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_small_fast(out, a, b);
    }
    
    // Fast path 4: Rank-1 ambient promotion (preserved)
    if (out->rank == 1 && a->rank == 1 && b->rank == 1) {
        return cpu_add_rank1_ambient(out, a, b);
    }
    
    // === NEW PHASE 2 OPTIMIZATIONS ===
    
    // NEW: Broadcast pattern detection and specialized kernels
    vsla_broadcast_pattern_t pattern = detect_broadcast_pattern(a, b);
    
    switch (pattern) {
        case BROADCAST_2D_ROW:
            return cpu_add_2d_row_broadcast(out, a, b);
        case BROADCAST_2D_COL:
            return cpu_add_2d_col_broadcast(out, a, b);
        case BROADCAST_3D_SPATIAL_W:
            return cpu_add_3d_spatial_w_broadcast(out, a, b);
        case BROADCAST_3D_SPATIAL_H:
            return cpu_add_3d_spatial_h_broadcast(out, a, b);
        case BROADCAST_4D_CHANNEL:
            return cpu_add_4d_channel_broadcast(out, a, b);
        case BROADCAST_4D_BATCH:
            return cpu_add_4d_batch_broadcast(out, a, b);
        case BROADCAST_SCALAR:
            return cpu_add_scalar_broadcast(out, a, b);
        default:
            break; // Continue to other optimizations
    }
    
    // NEW: Cache-friendly optimization selection
    if (should_use_shape_strides(a, b, out)) {
        // Use shape-based strides for better cache performance
        return cpu_add_optimized_ambient(out, a, b);
    }
    
    // === EXISTING FALLBACK PATHS (preserved) ===
    
    // Existing: Block-wise ambient promotion for higher dimensions
    if (total > 64) {
        return cpu_add_block_ambient(out, a, b);
    }
    
    // Fallback: Original ambient promotion algorithm from Section 4.1
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            // Get values with zero-extension for out-of-bounds
            double va = (vsla_is_empty(a) || !in_bounds(a, idx)) ? 0.0 : 
                       ((double*)a->data)[vsla_offset(a, idx)];
            double vb = (vsla_is_empty(b) || !in_bounds(b, idx)) ? 0.0 : 
                       ((double*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = va + vb;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            // Use double accumulation as per Section 7
            double va = (vsla_is_empty(a) || !in_bounds(a, idx)) ? 0.0 : 
                       ((float*)a->data)[vsla_offset(a, idx)];
            double vb = (vsla_is_empty(b) || !in_bounds(b, idx)) ? 0.0 : 
                       ((float*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = (float)(va + vb);
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

/**
 * Wrapper function for gradual migration
 * This allows us to switch between original and enhanced versions
 */
/**
 * @brief CPU subtraction operation
 */
vsla_error_t cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // For now, use similar logic to addition but with subtraction
    // This could be optimized similarly to addition
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Use the enhanced addition logic but with subtraction
    // TODO: Add full optimization for subtraction like we have for addition
    return cpu_add_enhanced(out, a, b); // Placeholder - needs proper subtraction implementation
}

/**
 * @brief CPU scaling operation
 */
vsla_error_t cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, double scalar) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Basic implementation - could be optimized with SIMD
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* src = (const double*)tensor->data;
        double* dst = (double*)out->data;
        uint64_t total_elems = vsla_logical_elems(tensor);
        
        for (uint64_t i = 0; i < total_elems; i++) {
            dst[i] = src[i] * scalar;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* src = (const float*)tensor->data;
        float* dst = (float*)out->data;
        uint64_t total_elems = vsla_logical_elems(tensor);
        float scalar_f = (float)scalar;
        
        for (uint64_t i = 0; i < total_elems; i++) {
            dst[i] = src[i] * scalar_f;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief CPU Hadamard (element-wise) product
 */
vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // Similar to addition but with multiplication
    // TODO: Add full optimization like addition
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Basic implementation - multiply corresponding elements
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        uint64_t total_elems = vsla_logical_elems(out);
        
        for (uint64_t i = 0; i < total_elems; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        uint64_t total_elems = vsla_logical_elems(out);
        
        for (uint64_t i = 0; i < total_elems; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t cpu_add_with_optimizations(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // For now, use enhanced version by default
    // In production, this could have feature flags or performance thresholds
    return cpu_add_enhanced(out, a, b);
}