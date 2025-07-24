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

/**
 * @brief Elementwise addition following Section 4.1
 * 
 * Shape rule: out->shape[i] = max(a->shape[i], b->shape[i])
 * Empty operands behave as zeros.
 * Complexity: ∏_i max(d^a_i, d^b_i)
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

    // Reference algorithm from Section 4.1
    size_t total = vsla_logical_elems(out); // ambient size
    
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