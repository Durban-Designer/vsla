/**
 * @file vsla_cpu_arithmetic_optimized_v2.c
 * @brief Cache-optimized VSLA CPU arithmetic operations
 * 
 * This file implements optimized versions of arithmetic operations that address
 * the 20-40% performance gap in multi-dimensional operations by:
 * 
 * 1. Using shape-based strides for better cache locality
 * 2. Specialized kernels for common broadcast patterns  
 * 3. Reduced coordinate transformation overhead
 * 4. SIMD-friendly memory access patterns
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>

// Import optimized helper functions
extern void compute_shape_strides(const vsla_tensor_t* t, uint64_t* strides);
extern void compute_capacity_strides(const vsla_tensor_t* t, uint64_t* strides);
extern bool should_use_shape_strides(const vsla_tensor_t* a, const vsla_tensor_t* b, const vsla_tensor_t* out);
extern uint64_t vsla_offset_shape_based(const vsla_tensor_t* t, const uint64_t* idx);

// Import from existing helpers
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);

// Broadcast pattern detection
typedef enum {
    BROADCAST_UNKNOWN = 0,
    BROADCAST_2D_ROW,      
    BROADCAST_2D_COL,      
    BROADCAST_3D_SPATIAL,  
    BROADCAST_4D_BATCH,    
    BROADCAST_4D_CHANNEL,  
    BROADCAST_SCALAR       
} vsla_broadcast_pattern_t;

extern vsla_broadcast_pattern_t detect_broadcast_pattern(const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * Optimized 2D row broadcasting: [N,M] + [1,M] → [N,M]
 * Eliminates coordinate transformation by direct stride arithmetic
 */
static vsla_error_t cpu_add_2d_row_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t rows = a->shape[0];
    uint64_t cols = a->shape[1];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    // Use shape-based strides for optimal cache access
    uint64_t a_strides[2], out_strides[2];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    // Vectorizable row-wise operation
    for (uint64_t row = 0; row < rows; row++) {
        uint64_t a_row_offset = row * a_strides[0];
        uint64_t out_row_offset = row * out_strides[0];
        
        // Inner loop is sequential and SIMD-friendly
        for (uint64_t col = 0; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_data[col];
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized 2D column broadcasting: [N,M] + [N,1] → [N,M]
 */
static vsla_error_t cpu_add_2d_col_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t rows = a->shape[0];
    uint64_t cols = a->shape[1];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    uint64_t a_strides[2], out_strides[2];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    for (uint64_t row = 0; row < rows; row++) {
        uint64_t a_row_offset = row * a_strides[0];
        uint64_t out_row_offset = row * out_strides[0];
        double b_val = b_data[row]; // Broadcast value
        
        // Vectorizable: add same value to entire row
        for (uint64_t col = 0; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_val;
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized scalar broadcasting: [shape] + [1] → [shape]
 */
static vsla_error_t cpu_add_scalar_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t total = vsla_logical_elems(a);
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    double b_val = b_data[0]; // Scalar value
    
    // Perfect sequential access, highly vectorizable
    for (uint64_t i = 0; i < total; i++) {
        out_data[i] = a_data[i] + b_val;
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized ambient promotion using shape-based strides
 * Reduces coordinate transformation overhead
 */
static vsla_error_t cpu_add_optimized_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t total = vsla_logical_elems(out);
    
    // Use shape-based strides for better cache locality
    uint64_t out_strides[VSLA_MAX_RANK];
    uint64_t a_strides[VSLA_MAX_RANK];
    uint64_t b_strides[VSLA_MAX_RANK];
    
    compute_shape_strides(out, out_strides);
    compute_shape_strides(a, a_strides);
    compute_shape_strides(b, b_strides);
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    // Larger block size for better cache utilization
    const uint64_t BLOCK_SIZE = 256;
    uint64_t idx[VSLA_MAX_RANK];
    
    for (uint64_t block_start = 0; block_start < total; block_start += BLOCK_SIZE) {
        uint64_t block_end = (block_start + BLOCK_SIZE < total) ? block_start + BLOCK_SIZE : total;
        
        for (uint64_t lin = block_start; lin < block_end; ++lin) {
            // Optimized coordinate computation
            uint64_t temp_lin = lin;
            for (int j = out->rank - 1; j >= 0; --j) {
                uint64_t s = out->shape[j];
                idx[j] = temp_lin % s;
                temp_lin /= s;
            }
            
            // Fast bounds checking with early exit
            double va = 0.0, vb = 0.0;
            
            // Optimized bounds check for a
            bool a_valid = (a->rank <= out->rank);
            uint64_t a_off = 0;
            for (int d = 0; d < a->rank && a_valid; ++d) {
                if (idx[d] >= a->shape[d]) {
                    a_valid = false;
                } else {
                    a_off += idx[d] * a_strides[d];
                }
            }
            if (a_valid && !vsla_is_empty(a)) {
                va = a_data[a_off];
            }
            
            // Optimized bounds check for b  
            bool b_valid = (b->rank <= out->rank);
            uint64_t b_off = 0;
            for (int d = 0; d < b->rank && b_valid; ++d) {
                if (idx[d] >= b->shape[d]) {
                    b_valid = false;
                } else {
                    b_off += idx[d] * b_strides[d];
                }
            }
            if (b_valid && !vsla_is_empty(b)) {
                vb = b_data[b_off];
            }
            
            // Sequential output access using shape-based stride
            uint64_t out_off = 0;
            for (int j = 0; j < out->rank; ++j) {
                out_off += idx[j] * out_strides[j];
            }
            out_data[out_off] = va + vb;
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Main optimized addition function with fast path selection
 */
vsla_error_t cpu_add_optimized(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) return VSLA_ERROR_NULL_POINTER;
    if (vsla_is_empty(out)) return VSLA_SUCCESS; // Empty result
    
    // Fast path 1: Equal shapes (existing optimization)
    if (a->rank == b->rank && a->rank == out->rank) {
        bool shapes_equal = true;
        for (uint8_t i = 0; i < a->rank; i++) {
            if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i]) {
                shapes_equal = false;
                break;
            }
        }
        if (shapes_equal) {
            // Use existing fast path
            uint64_t total = vsla_logical_elems(out);
            if (out->dtype == VSLA_DTYPE_F64) {
                double* out_data = (double*)out->data;
                const double* a_data = (const double*)a->data;
                const double* b_data = (const double*)b->data;
                
                for (uint64_t i = 0; i < total; i++) {
                    out_data[i] = a_data[i] + b_data[i];
                }
                return VSLA_SUCCESS;
            }
        }
    }
    
    // Fast path 2: Detect and handle common broadcast patterns
    vsla_broadcast_pattern_t pattern = detect_broadcast_pattern(a, b);
    
    switch (pattern) {
        case BROADCAST_2D_ROW:
            return cpu_add_2d_row_broadcast(out, a, b);
        case BROADCAST_2D_COL:
            return cpu_add_2d_col_broadcast(out, a, b);
        case BROADCAST_SCALAR:
            return cpu_add_scalar_broadcast(out, a, b);
        default:
            break;
    }
    
    // Fallback: Optimized ambient promotion with shape-based strides
    return cpu_add_optimized_ambient(out, a, b);
}

/**
 * Wrapper function to replace existing cpu_add
 * This allows gradual migration to optimized implementation
 */
vsla_error_t cpu_add_v2(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // For now, use optimized version if shapes suggest cache benefits
    if (should_use_shape_strides(a, b, out)) {
        return cpu_add_optimized(out, a, b);
    }
    
    // Fall back to original implementation for edge cases
    // This would call the original cpu_add function
    // For this demo, we'll use the optimized version
    return cpu_add_optimized(out, a, b);
}