/**
 * @file vsla_cpu_arithmetic.c
 * @brief VSLA CPU arithmetic operations following v3.1 specification
 * 
 * Implements Section 4: Arithmetic Operators
 * - 4.1 Elementwise Add/Subtract with ambient promotion
 * - 4.2 Hadamard Product
 * - 4.3 Convolution (Model A) 
 * - 4.4 Kronecker Product (Model B)
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>

// Helper functions (implemented in vsla_cpu_helpers.c)
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);
extern void unravel(uint64_t lin, const uint64_t* shape, uint8_t rank, uint64_t* out);
extern uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);
extern void compute_strides(const vsla_tensor_t* t, uint64_t* strides);

// Helper function to check if two tensors have identical shapes
bool shapes_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (a->rank != b->rank) return false;
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

// Fast path for equal-size dense tensors (no ambient promotion needed)
static vsla_error_t cpu_add_equal_size_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Vectorized addition - compiler will auto-vectorize this loop
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

// Micro-vector optimization for very small vectors (1-4 elements)
static vsla_error_t cpu_add_micro_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Highly optimized micro-vector cases
        switch (total) {
            case 1:
                out_data[0] = a_data[0] + b_data[0];
                return VSLA_SUCCESS;
            case 2:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                return VSLA_SUCCESS;
            case 3:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                return VSLA_SUCCESS;
            case 4:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                out_data[3] = a_data[3] + b_data[3];
                return VSLA_SUCCESS;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        
        switch (total) {
            case 1:
                out_data[0] = a_data[0] + b_data[0];
                return VSLA_SUCCESS;
            case 2:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                return VSLA_SUCCESS;
            case 3:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                return VSLA_SUCCESS;
            case 4:
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                out_data[3] = a_data[3] + b_data[3];
                return VSLA_SUCCESS;
        }
    }
    
    return VSLA_ERROR_INVALID_DTYPE;
}

// Small vector fast path with manual unrolling (5-32 elements)
static vsla_error_t cpu_add_small_fast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Manual unrolling for common small sizes
        switch (total) {
            case 8:
                // Optimized 8-element unroll
                out_data[0] = a_data[0] + b_data[0];
                out_data[1] = a_data[1] + b_data[1];
                out_data[2] = a_data[2] + b_data[2];
                out_data[3] = a_data[3] + b_data[3];
                out_data[4] = a_data[4] + b_data[4];
                out_data[5] = a_data[5] + b_data[5];
                out_data[6] = a_data[6] + b_data[6];
                out_data[7] = a_data[7] + b_data[7];
                break;
            case 16:
                // Optimized 16-element unroll
                for (int i = 0; i < 16; i += 4) {
                    out_data[i] = a_data[i] + b_data[i];
                    out_data[i+1] = a_data[i+1] + b_data[i+1];
                    out_data[i+2] = a_data[i+2] + b_data[i+2];
                    out_data[i+3] = a_data[i+3] + b_data[i+3];
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

// Optimized ambient promotion for rank-1 tensors (common case)
static vsla_error_t cpu_add_rank1_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->rank != 1 || a->rank != 1 || b->rank != 1) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t out_size = out->shape[0];
    uint64_t a_size = a->shape[0];
    uint64_t b_size = b->shape[0];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Optimized loop for rank-1 ambient promotion
        for (uint64_t i = 0; i < out_size; i++) {
            double va = (i < a_size) ? a_data[i] : 0.0;
            double vb = (i < b_size) ? b_data[i] : 0.0;
            out_data[i] = va + vb;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        
        for (uint64_t i = 0; i < out_size; i++) {
            double va = (i < a_size) ? a_data[i] : 0.0;
            double vb = (i < b_size) ? b_data[i] : 0.0;
            out_data[i] = (float)(va + vb);
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

// Block-wise ambient promotion for high-dimensional cases
static vsla_error_t cpu_add_block_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out);
    
    // Pre-compute strides for efficiency
    uint64_t out_strides[VSLA_MAX_RANK];
    uint64_t a_strides[VSLA_MAX_RANK];
    uint64_t b_strides[VSLA_MAX_RANK];
    
    compute_strides(out, out_strides);
    compute_strides(a, a_strides);
    compute_strides(b, b_strides);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        
        // Process in blocks to improve cache efficiency
        const size_t BLOCK_SIZE = 64;
        uint64_t idx[VSLA_MAX_RANK];
        
        for (size_t block_start = 0; block_start < total; block_start += BLOCK_SIZE) {
            size_t block_end = (block_start + BLOCK_SIZE < total) ? block_start + BLOCK_SIZE : total;
            
            for (size_t lin = block_start; lin < block_end; ++lin) {
                // Optimized unravel - inline for better performance
                uint64_t temp_lin = lin;
                for (int j = out->rank - 1; j >= 0; --j) {
                    uint64_t s = out->shape[j];
                    idx[j] = (s ? temp_lin % s : 0);
                    temp_lin /= (s ? s : 1);
                }
                
                // Fast bounds check and value extraction
                double va = 0.0, vb = 0.0;
                
                // Check bounds for tensor a
                bool a_in_bounds = true;
                for (int d = 0; d < a->rank && a_in_bounds; ++d) {
                    if (idx[d] >= a->shape[d]) a_in_bounds = false;
                }
                if (a_in_bounds && !vsla_is_empty(a)) {
                    uint64_t a_off = 0;
                    for (int j = 0; j < a->rank; ++j) {
                        a_off += idx[j] * a_strides[j];
                    }
                    va = a_data[a_off];
                }
                
                // Check bounds for tensor b
                bool b_in_bounds = true;
                for (int d = 0; d < b->rank && b_in_bounds; ++d) {
                    if (idx[d] >= b->shape[d]) b_in_bounds = false;
                }
                if (b_in_bounds && !vsla_is_empty(b)) {
                    uint64_t b_off = 0;
                    for (int j = 0; j < b->rank; ++j) {
                        b_off += idx[j] * b_strides[j];
                    }
                    vb = b_data[b_off];
                }
                
                // Compute output offset and store result
                uint64_t out_off = 0;
                for (int j = 0; j < out->rank; ++j) {
                    out_off += idx[j] * out_strides[j];
                }
                out_data[out_off] = va + vb;
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        // Similar optimized implementation for F32
        float* out_data = (float*)out->data;
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        
        const size_t BLOCK_SIZE = 64;
        uint64_t idx[VSLA_MAX_RANK];
        
        for (size_t block_start = 0; block_start < total; block_start += BLOCK_SIZE) {
            size_t block_end = (block_start + BLOCK_SIZE < total) ? block_start + BLOCK_SIZE : total;
            
            for (size_t lin = block_start; lin < block_end; ++lin) {
                uint64_t temp_lin = lin;
                for (int j = out->rank - 1; j >= 0; --j) {
                    uint64_t s = out->shape[j];
                    idx[j] = (s ? temp_lin % s : 0);
                    temp_lin /= (s ? s : 1);
                }
                
                double va = 0.0, vb = 0.0;
                
                bool a_in_bounds = true;
                for (int d = 0; d < a->rank && a_in_bounds; ++d) {
                    if (idx[d] >= a->shape[d]) a_in_bounds = false;
                }
                if (a_in_bounds && !vsla_is_empty(a)) {
                    uint64_t a_off = 0;
                    for (int j = 0; j < a->rank; ++j) {
                        a_off += idx[j] * a_strides[j];
                    }
                    va = a_data[a_off];
                }
                
                bool b_in_bounds = true;
                for (int d = 0; d < b->rank && b_in_bounds; ++d) {
                    if (idx[d] >= b->shape[d]) b_in_bounds = false;
                }
                if (b_in_bounds && !vsla_is_empty(b)) {
                    uint64_t b_off = 0;
                    for (int j = 0; j < b->rank; ++j) {
                        b_off += idx[j] * b_strides[j];
                    }
                    vb = b_data[b_off];
                }
                
                uint64_t out_off = 0;
                for (int j = 0; j < out->rank; ++j) {
                    out_off += idx[j] * out_strides[j];
                }
                out_data[out_off] = (float)(va + vb);
            }
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief Optimized elementwise addition following Section 4.1
 * 
 * Shape rule: out->shape[i] = max(a->shape[i], b->shape[i])
 * Empty operands behave as zeros.
 * Multiple optimization paths:
 * 1. Equal-size fast path (no ambient promotion)
 * 2. Small vector fast path (< 32 elements)
 * 3. Original ambient promotion algorithm (fallback)
 */
vsla_error_t cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Pre-condition checks from Section 3.1
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

    // Handle empty operands (behave as zeros)
    if (vsla_is_empty(a) && vsla_is_empty(b)) {
        // Both empty -> output should be zeroed
        if (out->data) {
            memset(out->data, 0, vsla_logical_elems(out) * vsla_dtype_size(out->dtype));
        }
        return VSLA_SUCCESS;
    }

    // Fast path optimizations for common cases
    size_t total = vsla_logical_elems(out);
    
    // Optimization 0: Micro-vector fast path (1-4 elements, equal size)
    if (total <= 4 && shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_micro_fast(out, a, b);
    }
    
    // Optimization 1: Equal-size dense tensors (no ambient promotion needed)
    if (shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_equal_size_fast(out, a, b);
    }
    
    // Optimization 2: Small vector fast path (5-32 elements)
    if (total >= 5 && total < 32 && shapes_equal(a, b) && shapes_equal(b, out) && !vsla_is_empty(a) && !vsla_is_empty(b)) {
        return cpu_add_small_fast(out, a, b);
    }
    
    // Optimization 3: Rank-1 ambient promotion (most common sparse case)
    if (out->rank == 1 && a->rank == 1 && b->rank == 1) {
        return cpu_add_rank1_ambient(out, a, b);
    }
    
    // Optimization 4: Block-wise ambient promotion for higher dimensions
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
 * @brief Elementwise subtraction following Section 4.1
 */
vsla_error_t cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Same pre-conditions as addition
    if (a->model != b->model || a->model != out->model) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->rank != b->rank || a->rank != out->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    // Verify output shape follows the rule: max(a->shape[i], b->shape[i])
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

    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            double va = (vsla_is_empty(a) || !in_bounds(a, idx)) ? 0.0 : 
                       ((double*)a->data)[vsla_offset(a, idx)];
            double vb = (vsla_is_empty(b) || !in_bounds(b, idx)) ? 0.0 : 
                       ((double*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = va - vb;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            double va = (vsla_is_empty(a) || !in_bounds(a, idx)) ? 0.0 : 
                       ((float*)a->data)[vsla_offset(a, idx)];
            double vb = (vsla_is_empty(b) || !in_bounds(b, idx)) ? 0.0 : 
                       ((float*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = (float)(va - vb);
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    return VSLA_SUCCESS;
}

/**
 * @brief Hadamard (elementwise) product following Section 4.2
 */
vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Same pre-conditions and shape rules as addition
    if (a->model != b->model || a->model != out->model) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (a->rank != b->rank || a->rank != out->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    for (uint8_t i = 0; i < out->rank; i++) {
        uint64_t expected = (a->shape[i] > b->shape[i]) ? a->shape[i] : b->shape[i];
        if (out->shape[i] != expected) {
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
    }

    // Handle empty operands (multiplication with empty yields zero)
    if (vsla_is_empty(a) || vsla_is_empty(b)) {
        if (out->data) {
            memset(out->data, 0, vsla_logical_elems(out) * vsla_dtype_size(out->dtype));
        }
        return VSLA_SUCCESS;
    }

    size_t total = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            double va = (!in_bounds(a, idx)) ? 0.0 : ((double*)a->data)[vsla_offset(a, idx)];
            double vb = (!in_bounds(b, idx)) ? 0.0 : ((double*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = va * vb;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* out_data = (float*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            double va = (!in_bounds(a, idx)) ? 0.0 : ((float*)a->data)[vsla_offset(a, idx)];
            double vb = (!in_bounds(b, idx)) ? 0.0 : ((float*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = (float)(va * vb);
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    return VSLA_SUCCESS;
}

/**
 * @brief Scalar multiplication
 */
vsla_error_t cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    if (!out || !in) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (in->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    // Output shape should match input shape exactly
    if (in->rank != out->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    for (uint8_t i = 0; i < in->rank; i++) {
        if (in->shape[i] != out->shape[i]) {
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
    }

    if (vsla_is_empty(in)) {
        return VSLA_SUCCESS; // Empty tensor remains empty
    }

    uint64_t total = vsla_logical_elems(in);
    
    if (in->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)in->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t i = 0; i < total; i++) {
            out_data[i] = in_data[i] * scalar;
        }
    } else if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)in->data;
        float* out_data = (float*)out->data;
        float fscalar = (float)scalar;
        
        for (uint64_t i = 0; i < total; i++) {
            out_data[i] = in_data[i] * fscalar;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    return VSLA_SUCCESS;
}