/**
 * @file vsla_cpu_reduction.c
 * @brief VSLA CPU reduction operations
 * 
 * Implements reduction operations like sum, norm, etc.
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <math.h>

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);

/**
 * @brief Sum all elements in a tensor
 */
vsla_error_t cpu_sum(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (vsla_is_empty(tensor) || !tensor->data) {
        *result = 0.0;
        return VSLA_SUCCESS;
    }
    
    uint64_t total_elems = vsla_logical_elems(tensor);
    double sum = 0.0;
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->data;
        // Use Kahan summation for better numerical precision
        double c = 0.0; // Compensation for lost low-order bits
        for (uint64_t i = 0; i < total_elems; i++) {
            double y = data[i] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->data;
        // Use double accumulation for better precision
        for (uint64_t i = 0; i < total_elems; i++) {
            sum += (double)data[i];
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    *result = sum;
    return VSLA_SUCCESS;
}

/**
 * @brief Calculate Euclidean norm of tensor
 */
vsla_error_t cpu_norm(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (vsla_is_empty(tensor) || !tensor->data) {
        *result = 0.0;
        return VSLA_SUCCESS;
    }
    
    uint64_t total_elems = vsla_logical_elems(tensor);
    double sum_squares = 0.0;
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->data;
        for (uint64_t i = 0; i < total_elems; i++) {
            double val = data[i];
            sum_squares += val * val;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->data;
        for (uint64_t i = 0; i < total_elems; i++) {
            double val = (double)data[i];
            sum_squares += val * val;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    *result = sqrt(sum_squares);
    return VSLA_SUCCESS;
}