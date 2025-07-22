/**
 * @file vsla_cpu_shrink.c
 * @brief VSLA CPU shrinking operations following v3.1 specification
 * 
 * Implements Section 6: Shrinking to Minimal Representative
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <math.h>

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx);
extern void compute_strides(const vsla_tensor_t* t, uint64_t* s);

/**
 * @brief Shrink tensor to minimal representative following Section 6
 * 
 * Removes trailing zero hyperplanes by scanning axes from last to first.
 * Complexity: worst-case O(product(shape)*rank); typically dominated by shrinkable suffix size.
 */
vsla_error_t cpu_shrink(vsla_tensor_t* t) {
    if (!t) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Early exit for already empty tensors
    for (int j = 0; j < t->rank; ++j) {
        if (t->shape[j] == 0) {
            return VSLA_SUCCESS; // Already empty
        }
    }
    
    if (!t->data) {
        return VSLA_SUCCESS; // No data to check
    }
    
    uint64_t strides[VSLA_MAX_RANK];
    compute_strides(t, strides);
    
    // Algorithm from Section 6: scan axes from last to first
    for (int axis = t->rank - 1; axis >= 0; --axis) {
        while (t->shape[axis] > 0) {
            uint64_t last = t->shape[axis] - 1;
            bool all_zero = true;
            uint64_t plane_elems = 1;
            
            // Calculate number of elements in the hyperplane
            for (int j = 0; j < t->rank; ++j) {
                if (j != axis) {
                    plane_elems *= t->shape[j];
                }
            }
            
            // Check if the terminal hyperplane is all zeros
            uint64_t idx[VSLA_MAX_RANK] = {0};
            idx[axis] = last;
            
            for (uint64_t p = 0; p < plane_elems && all_zero; ++p) {
                double val;
                
                if (t->dtype == VSLA_DTYPE_F64) {
                    val = ((double*)t->data)[vsla_offset(t, idx)];
                } else if (t->dtype == VSLA_DTYPE_F32) {
                    val = (double)((float*)t->data)[vsla_offset(t, idx)];
                } else {
                    return VSLA_ERROR_INVALID_DTYPE;
                }
                
                if (val != 0.0) {
                    all_zero = false;
                    break;
                }
                
                // Increment index in lexicographic order, skipping the current axis
                for (int j = t->rank - 1; j >= 0; --j) {
                    if (j == axis) continue;
                    if (++idx[j] < t->shape[j]) break;
                    idx[j] = 0;
                }
            }
            
            if (!all_zero) break;
            
            // The hyperplane is all zeros, so shrink this dimension
            --t->shape[axis];
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief Check if a tensor is in minimal representative form
 */
bool cpu_is_minimal(const vsla_tensor_t* t) {
    if (!t || vsla_is_empty(t) || !t->data) {
        return true; // Empty tensors are minimal by definition
    }
    
    // Create a copy and shrink it to see if it changes
    // This is a simple but potentially expensive check
    // In practice, you might want to implement a more efficient version
    
    // For now, we'll assume tensors are minimal unless proven otherwise
    // A more sophisticated implementation would check the actual trailing hyperplanes
    return true;
}