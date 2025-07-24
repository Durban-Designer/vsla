/**
 * @file vsla_cpu_arithmetic_optimized.c
 * @brief Optimized VSLA CPU arithmetic operations
 * 
 * High-performance implementations with multiple optimization paths:
 * 1. Equal-size fast path (no ambient promotion needed)
 * 2. Small vector fast path (< 32 elements)
 * 3. Vectorized ambient promotion for larger tensors
 * 4. Cache-optimized memory access patterns
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);

// Check if two tensors have identical shapes
static bool shapes_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (a->rank != b->rank) return false;
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

// Check if tensor is dense and contiguous (no gaps)
static bool is_dense_contiguous(const vsla_tensor_t* t) {
    // For now, assume all our tensors are dense and contiguous
    // This could be extended to check actual memory layout
    return !vsla_is_empty(t);
}

// Fast path for equal-size dense tensors
static vsla_error_t cpu_add_equal_size_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Vectorized addition - compiler will auto-vectorize this
        for (size_t i = 0; i < total; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        
        for (size_t i = 0; i < total; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

// Small vector fast path (< 32 elements)
static vsla_error_t cpu_add_small_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // For small vectors, unroll manually to avoid function call overhead
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Manual unrolling for small sizes
        switch (total) {
            case 1:
                out_data[0] = a_data[0] + b_data[0];
                break;
            case 2:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                break;
            case 4:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                out_data[3] = a_data[3] + b_data[3];
                break;
            case 8:
                for (int i = 0; i < 8; i++) {
                    out_data[i] = a_data[i] + b_data[i];
                }
                break;
            default:
                // General case for other small sizes
                for (size_t i = 0; i < total; i++) {
                    out_data[i] = a_data[i] + b_data[i];
                }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        
        for (size_t i = 0; i < total; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

// Optimized ambient promotion for different-sized tensors
static vsla_error_t cpu_add_ambient_optimized(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // This is more complex - we need to handle the case where tensors have different sizes
    // but we can still optimize by avoiding the expensive unravel/offset calls
    
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        
        // First, zero the output
        memset(out_data, 0, total * sizeof(double));
        
        // Add tensor 'a' data
        if (!vsla_is_empty(a)) {
            const double* a_data = (const double*)a->data;
            size_t a_elems = vsla_logical_elems(a);
            
            // For 1D tensors, this is straightforward
            if (out->rank == 1 && a->rank == 1) {
                size_t copy_elems = (a_elems < total) ? a_elems : total;
                for (size_t i = 0; i < copy_elems; i++) {
                    out_data[i] += a_data[i];
                }
            } else {
                // Fall back to general case (could be optimized further)
                // This would need the full unravel/offset logic
                return VSLA_ERROR_NOT_IMPLEMENTED;
            }
        }
        
        // Add tensor 'b' data
        if (!vsla_is_empty(b)) {
            const double* b_data = (const double*)b->data;
            size_t b_elems = vsla_logical_elems(b);
            
            if (out->rank == 1 && b->rank == 1) {
                size_t copy_elems = (b_elems < total) ? b_elems : total;
                for (size_t i = 0; i < copy_elems; i++) {
                    out_data[i] += b_data[i];
                }
            } else {
                return VSLA_ERROR_NOT_IMPLEMENTED;
            }
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief Optimized elementwise addition with multiple fast paths
 */
vsla_error_t cpu_add_optimized(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Pre-condition checks
    if (a->model != b->model || a->model != out->model) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->rank != b->rank || a->rank != out->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    // Verify output shape
    for (uint8_t i = 0; i < out->rank; i++) {
        uint64_t expected = (a->shape[i] > b->shape[i]) ? a->shape[i] : b->shape[i];
        if (out->shape[i] != expected) {
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
    }

    // Handle empty operands
    if (vsla_is_empty(a) && vsla_is_empty(b)) {
        if (out->data) {
            memset(out->data, 0, vsla_logical_elems(out) * vsla_dtype_size(out->dtype));
        }
        return VSLA_SUCCESS;
    }

    size_t total_elems = vsla_logical_elems(out);
    
    // **OPTIMIZATION PATH 1: Equal-size dense tensors**
    if (shapes_equal(a, b) && shapes_equal(a, out) && 
        is_dense_contiguous(a) && is_dense_contiguous(b)) {
        
        // Small vector optimization
        if (total_elems < 32) {
            return cpu_add_small_fast(out, a, b);
        } else {
            return cpu_add_equal_size_fast(out, a, b);
        }
    }
    
    // **OPTIMIZATION PATH 2: Ambient promotion optimization (1D only for now)**
    if (out->rank == 1 && a->rank == 1 && b->rank == 1) {
        return cpu_add_ambient_optimized(out, a, b);
    }
    
    // **FALLBACK: Use original implementation for complex cases**
    // This would call the original cpu_add function
    return VSLA_ERROR_NOT_IMPLEMENTED; // For now, indicate we need the original
}

// Similar optimizations for other operations
vsla_error_t cpu_sub_optimized(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // Similar structure to cpu_add_optimized but with subtraction
    // Implementation would follow the same pattern
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t cpu_mul_optimized(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // Hadamard product optimization
    return VSLA_ERROR_NOT_IMPLEMENTED;
}