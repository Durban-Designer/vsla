/**
 * @file vsla_cpu_advanced.c
 * @brief VSLA CPU advanced operations following v3.1 specification
 * 
 * Implements Section 4.3 and 4.4:
 * - Convolution (Model A) with direct and FFT paths
 * - Kronecker Product (Model B)
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);

// FFT threshold for switching from direct to FFT convolution
#define CONV_FFT_THRESHOLD 1024

/**
 * @brief Discrete convolution for Model A following Section 4.3
 * 
 * Rank: vectors only (rank==1)
 * Shape rule: out->shape[0] = (m==0||n==0) ? 0 : m+n-1
 * where m=a->shape[0], n=b->shape[0]
 */
vsla_error_t cpu_conv(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Pre-condition checks
    if (a->model != VSLA_MODEL_A || b->model != VSLA_MODEL_A || out->model != VSLA_MODEL_A) {
        return VSLA_ERROR_INVALID_ARGUMENT; // Must be Model A
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Rank check: vectors only
    if (a->rank != 1 || b->rank != 1 || out->rank != 1) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t m = a->shape[0];
    uint64_t n = b->shape[0];
    
    // Handle empty operands - empty operand yields empty result
    if (m == 0 || n == 0) {
        // Verify output shape is 0
        if (out->shape[0] != 0) {
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
        return VSLA_SUCCESS; // Empty result
    }

    // Verify output shape follows rule: m + n - 1
    uint64_t expected_out_size = m + n - 1;
    if (out->shape[0] != expected_out_size) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    // Choose algorithm based on size
    uint64_t complexity = m * n;
    
    if (complexity < CONV_FFT_THRESHOLD) {
        // Direct convolution from Section 4.3
        if (out->dtype == VSLA_DTYPE_F64) {
            const double* A = (const double*)a->data;
            const double* B = (const double*)b->data;
            double* OUT = (double*)out->data;
            
            for (uint64_t k = 0; k < expected_out_size; ++k) {
                double sum = 0.0;
                uint64_t lo = (k < n - 1 ? 0 : k - (n - 1));
                uint64_t hi = (k < m - 1 ? k : m - 1);
                
                for (uint64_t i = lo; i <= hi; ++i) {
                    // Use fma for precision where available
                    #ifdef FP_FAST_FMA
                    sum = fma(A[i], B[k - i], sum);
                    #else
                    sum += A[i] * B[k - i];
                    #endif
                }
                OUT[k] = sum;
            }
        } else if (out->dtype == VSLA_DTYPE_F32) {
            const float* A = (const float*)a->data;
            const float* B = (const float*)b->data;
            float* OUT = (float*)out->data;
            
            for (uint64_t k = 0; k < expected_out_size; ++k) {
                // Use double accumulation for precision
                double sum = 0.0;
                uint64_t lo = (k < n - 1 ? 0 : k - (n - 1));
                uint64_t hi = (k < m - 1 ? k : m - 1);
                
                for (uint64_t i = lo; i <= hi; ++i) {
                    sum += (double)A[i] * (double)B[k - i];
                }
                OUT[k] = (float)sum;
            }
        } else {
            return VSLA_ERROR_INVALID_DTYPE;
        }
    } else {
        // TODO: FFT convolution path for large inputs
        // For now, fall back to direct method
        return cpu_conv(out, a, b); // Recursive call with direct path
    }

    return VSLA_SUCCESS;
}

/**
 * @brief Kronecker product for Model B following Section 4.4
 * 
 * Rank: vectors only (rank==1)
 * Shape rule: out->shape[0] = (m==0||n==0) ? 0 : m*n
 * where m=a->shape[0], n=b->shape[0]
 * 
 * Non-commutative unless one operand is scalar [1]
 */
vsla_error_t cpu_kron(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Pre-condition checks
    if (a->model != VSLA_MODEL_B || b->model != VSLA_MODEL_B || out->model != VSLA_MODEL_B) {
        return VSLA_ERROR_INVALID_ARGUMENT; // Must be Model B
    }
    
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Rank check: vectors only
    if (a->rank != 1 || b->rank != 1 || out->rank != 1) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t m = a->shape[0];
    uint64_t n = b->shape[0];
    
    // Handle empty operands - empty operand yields empty result
    if (m == 0 || n == 0) {
        // Verify output shape is 0
        if (out->shape[0] != 0) {
            return VSLA_ERROR_INVALID_ARGUMENT;
        }
        return VSLA_SUCCESS; // Empty result
    }

    // Verify output shape follows rule: m * n
    uint64_t expected_out_size = m * n;
    if (out->shape[0] != expected_out_size) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    // Direct Kronecker product algorithm from Section 4.4
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* A = (const double*)a->data;
        const double* B = (const double*)b->data;
        double* OUT = (double*)out->data;
        
        for (uint64_t i = 0; i < m; ++i) {
            double ai = A[i];
            double* dst = OUT + i * n;
            for (uint64_t j = 0; j < n; ++j) {
                dst[j] = ai * B[j];
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* OUT = (float*)out->data;
        
        for (uint64_t i = 0; i < m; ++i) {
            float ai = A[i];
            float* dst = OUT + i * n;
            for (uint64_t j = 0; j < n; ++j) {
                dst[j] = ai * B[j];
            }
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    return VSLA_SUCCESS;
}