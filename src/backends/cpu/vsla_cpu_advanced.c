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

// FFT convolution function from vsla_cpu_fft.c
extern int vsla_conv_fft(double* out, const double* A, size_t m, const double* B, size_t n);

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
        // FFT convolution path for large inputs
        if (out->dtype == VSLA_DTYPE_F64) {
            const double* A = (const double*)a->data;
            const double* B = (const double*)b->data;
            double* OUT = (double*)out->data;
            
            int fft_result = vsla_conv_fft(OUT, A, m, B, n);
            if (fft_result != 0) {
                return VSLA_ERROR_INVALID_ARGUMENT; // FFT failed, could be allocation
            }
        } else if (out->dtype == VSLA_DTYPE_F32) {
            // For float32, we need to convert to double for FFT, then back
            // Allocate temporary double buffers
            double* A_d = (double*)malloc(m * sizeof(double));
            double* B_d = (double*)malloc(n * sizeof(double));
            double* OUT_d = (double*)malloc(expected_out_size * sizeof(double));
            
            if (!A_d || !B_d || !OUT_d) {
                free(A_d);
                free(B_d);
                free(OUT_d);
                return VSLA_ERROR_INVALID_ARGUMENT;
            }
            
            // Convert inputs to double
            const float* A_f = (const float*)a->data;
            const float* B_f = (const float*)b->data;
            for (uint64_t i = 0; i < m; i++) A_d[i] = (double)A_f[i];
            for (uint64_t i = 0; i < n; i++) B_d[i] = (double)B_f[i];
            
            // Run FFT convolution
            int fft_result = vsla_conv_fft(OUT_d, A_d, m, B_d, n);
            
            if (fft_result == 0) {
                // Convert result back to float
                float* OUT_f = (float*)out->data;
                for (uint64_t i = 0; i < expected_out_size; i++) {
                    OUT_f[i] = (float)OUT_d[i];
                }
            }
            
            free(A_d);
            free(B_d);
            free(OUT_d);
            
            if (fft_result != 0) {
                return VSLA_ERROR_INVALID_ARGUMENT;
            }
        } else {
            return VSLA_ERROR_INVALID_DTYPE;
        }
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