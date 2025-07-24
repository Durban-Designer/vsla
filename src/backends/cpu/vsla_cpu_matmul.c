/**
 * @file vsla_cpu_matmul.c
 * @brief VSLA CPU matrix multiplication implementation
 * 
 * Implements matrix multiplication with VSLA variable-shape semantics.
 * Supports broadcasting for bias addition and efficient computation patterns.
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

// SIMD includes for optimized matrix multiplication
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX2__)
        #include <immintrin.h>
        #ifndef VSLA_HAS_AVX2
        #define VSLA_HAS_AVX2
        #endif
    #elif defined(__SSE2__)
        #include <emmintrin.h>
        #ifndef VSLA_HAS_SSE2
        #define VSLA_HAS_SSE2
        #endif
    #endif
#elif defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #ifndef VSLA_HAS_NEON
    #define VSLA_HAS_NEON
    #endif
#endif

// Helper functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);

// Block size for cache-efficient computation
#define MATMUL_BLOCK_SIZE 64

/**
 * @brief Validate matrix multiplication shapes
 * For VSLA semantics: A[m,k] @ B[k,n] = C[m,n]
 * With variable-shape support for broadcasting scenarios
 */
static vsla_error_t validate_matmul_shapes(const vsla_tensor_t* a, const vsla_tensor_t* b, 
                                          const vsla_tensor_t* out, 
                                          uint64_t* m, uint64_t* k, uint64_t* n) {
    // Both inputs must be at least 2D
    if (a->rank < 2 || b->rank < 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Extract dimensions
    *m = a->shape[a->rank - 2];  // rows of A
    *k = a->shape[a->rank - 1];  // cols of A / rows of B
    uint64_t k_b = b->shape[b->rank - 2];  // rows of B
    *n = b->shape[b->rank - 1];  // cols of B
    
    // Inner dimensions must match (standard matrix multiplication rule)
    if (*k != k_b) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Output shape validation
    if (out->rank < 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Output should be [m, n] for the last two dimensions
    if (out->shape[out->rank - 2] != *m || out->shape[out->rank - 1] != *n) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Handle batch dimensions (broadcasting support)
    // For now, we support simple 2D matrix multiplication
    // TODO: Add batch matrix multiplication support
    if (a->rank > 2 || b->rank > 2 || out->rank > 2) {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief Naive matrix multiplication kernel
 * C[i,j] = sum(A[i,k] * B[k,j]) for k in [0, K)
 */
static void matmul_naive_f64(const double* A, const double* B, double* C,
                            uint64_t m, uint64_t k, uint64_t n) {
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint64_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Cache-blocked matrix multiplication kernel
 * Uses blocking to improve cache locality
 */
static void matmul_blocked_f64(const double* A, const double* B, double* C,
                              uint64_t m, uint64_t k, uint64_t n) {
    // Initialize output to zero
    memset(C, 0, m * n * sizeof(double));
    
    // Block over all three dimensions
    for (uint64_t i0 = 0; i0 < m; i0 += MATMUL_BLOCK_SIZE) {
        uint64_t i_max = (i0 + MATMUL_BLOCK_SIZE < m) ? i0 + MATMUL_BLOCK_SIZE : m;
        
        for (uint64_t j0 = 0; j0 < n; j0 += MATMUL_BLOCK_SIZE) {
            uint64_t j_max = (j0 + MATMUL_BLOCK_SIZE < n) ? j0 + MATMUL_BLOCK_SIZE : n;
            
            for (uint64_t k0 = 0; k0 < k; k0 += MATMUL_BLOCK_SIZE) {
                uint64_t k_max = (k0 + MATMUL_BLOCK_SIZE < k) ? k0 + MATMUL_BLOCK_SIZE : k;
                
                // Compute block
                for (uint64_t i = i0; i < i_max; i++) {
                    for (uint64_t j = j0; j < j_max; j++) {
                        double sum = C[i * n + j];
                        for (uint64_t l = k0; l < k_max; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

#ifdef VSLA_HAS_AVX2
/**
 * @brief AVX2-optimized matrix multiplication kernel
 * Processes 4 doubles at a time
 */
static void matmul_avx2_f64(const double* A, const double* B, double* C,
                           uint64_t m, uint64_t k, uint64_t n) {
    // Initialize output to zero
    memset(C, 0, m * n * sizeof(double));
    
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < n; j += 4) {
            if (j + 4 <= n) {
                // Process 4 elements at once
                __m256d sum = _mm256_setzero_pd();
                
                for (uint64_t l = 0; l < k; l++) {
                    __m256d a_broadcast = _mm256_broadcast_sd(&A[i * k + l]);
                    __m256d b_vec = _mm256_loadu_pd(&B[l * n + j]);
                    sum = _mm256_fmadd_pd(a_broadcast, b_vec, sum);
                }
                
                _mm256_storeu_pd(&C[i * n + j], sum);
            } else {
                // Handle remaining elements
                for (uint64_t jj = j; jj < n; jj++) {
                    double sum = 0.0;
                    for (uint64_t l = 0; l < k; l++) {
                        sum += A[i * k + l] * B[l * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
            }
        }
    }
}
#endif

#ifdef VSLA_HAS_SSE2
/**
 * @brief SSE2-optimized matrix multiplication kernel
 * Processes 2 doubles at a time
 */
static void matmul_sse2_f64(const double* A, const double* B, double* C,
                           uint64_t m, uint64_t k, uint64_t n) {
    // Initialize output to zero
    memset(C, 0, m * n * sizeof(double));
    
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < n; j += 2) {
            if (j + 2 <= n) {
                // Process 2 elements at once
                __m128d sum = _mm_setzero_pd();
                
                for (uint64_t l = 0; l < k; l++) {
                    __m128d a_broadcast = _mm_set1_pd(A[i * k + l]);
                    __m128d b_vec = _mm_loadu_pd(&B[l * n + j]);
                    sum = _mm_add_pd(sum, _mm_mul_pd(a_broadcast, b_vec));
                }
                
                _mm_storeu_pd(&C[i * n + j], sum);
            } else {
                // Handle remaining element
                double sum = 0.0;
                for (uint64_t l = 0; l < k; l++) {
                    sum += A[i * k + l] * B[l * n + n - 1];
                }
                C[i * n + n - 1] = sum;
            }
        }
    }
}
#endif

/**
 * @brief Float32 matrix multiplication kernel
 */
static void matmul_f32(const float* A, const float* B, float* C,
                      uint64_t m, uint64_t k, uint64_t n) {
    // Use blocked algorithm for better cache performance
    memset(C, 0, m * n * sizeof(float));
    
    const uint64_t block_size = 32; // Smaller block for float32
    
    for (uint64_t i0 = 0; i0 < m; i0 += block_size) {
        uint64_t i_max = (i0 + block_size < m) ? i0 + block_size : m;
        
        for (uint64_t j0 = 0; j0 < n; j0 += block_size) {
            uint64_t j_max = (j0 + block_size < n) ? j0 + block_size : n;
            
            for (uint64_t k0 = 0; k0 < k; k0 += block_size) {
                uint64_t k_max = (k0 + block_size < k) ? k0 + block_size : k;
                
                for (uint64_t i = i0; i < i_max; i++) {
                    for (uint64_t j = j0; j < j_max; j++) {
                        float sum = C[i * n + j];
                        for (uint64_t l = k0; l < k_max; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * @brief Matrix multiplication for VSLA tensors
 * 
 * Computes C = A @ B following VSLA semantics
 * Supports 2D matrices with plans for batch operations
 */
vsla_error_t cpu_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // All tensors must have the same model
    if (a->model != b->model || a->model != out->model) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Type checking
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    // Validate shapes and extract dimensions
    uint64_t m, k, n;
    vsla_error_t err = validate_matmul_shapes(a, b, out, &m, &k, &n);
    if (err != VSLA_SUCCESS) {
        return err;
    }
    
    // Handle empty matrices
    if (m == 0 || k == 0 || n == 0) {
        // Output should already be empty, nothing to compute
        return VSLA_SUCCESS;
    }
    
    // Perform matrix multiplication based on data type
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* A = (const double*)a->data;
        const double* B = (const double*)b->data;
        double* C = (double*)out->data;
        
        // Choose optimal kernel based on size and architecture
        #ifdef VSLA_HAS_AVX2
        if (n >= 4) {
            matmul_avx2_f64(A, B, C, m, k, n);
        } else
        #elif defined(VSLA_HAS_SSE2)
        if (n >= 2) {
            matmul_sse2_f64(A, B, C, m, k, n);
        } else
        #endif
        if (m * n * k > 1000) {
            // Use blocked algorithm for larger matrices
            matmul_blocked_f64(A, B, C, m, k, n);
        } else {
            // Use naive algorithm for small matrices
            matmul_naive_f64(A, B, C, m, k, n);
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* C = (float*)out->data;
        
        matmul_f32(A, B, C, m, k, n);
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}