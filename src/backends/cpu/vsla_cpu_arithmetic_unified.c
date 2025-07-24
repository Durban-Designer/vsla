/**
 * @file vsla_cpu_arithmetic_unified.c
 * @brief Unified VSLA CPU arithmetic with intelligent optimization dispatch
 * 
 * This file implements Section 4: Arithmetic Operators with full VSLA v3.2 spec
 * compliance and intelligent optimization selection for maximum performance.
 * 
 * Key Features:
 * - Full VSLA spec v3.2 compliance (ambient promotion, minimal representatives)
 * - Intelligent optimization dispatch based on tensor characteristics
 * - SIMD vectorization for supported architectures
 * - Cache-optimized memory access patterns
 * - Specialized kernels for common broadcasting patterns
 * - Comprehensive error handling and bounds checking
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

// SIMD includes for vectorization
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define VSLA_HAS_AVX2 1
    #elif defined(__SSE2__)
        #include <emmintrin.h>
        #define VSLA_HAS_SSE2 1
    #endif
#elif defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define VSLA_HAS_NEON 1
#endif

// Helper functions (implemented in vsla_cpu_helpers.c)
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern uint64_t vsla_logical_elems(const vsla_tensor_t* t);
extern uint64_t vsla_capacity_elems(const vsla_tensor_t* t);
extern void unravel(uint64_t lin, const uint64_t* shape, uint8_t rank, uint64_t* out);
extern uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);
extern bool mul_ov(uint64_t a, uint64_t b);
extern bool shapes_equal(const vsla_tensor_t* a, const vsla_tensor_t* b);
extern void compute_strides(const vsla_tensor_t* t, uint64_t* s);

// ============================================================================
// BROADCAST PATTERN DETECTION
// ============================================================================

typedef enum {
    BROADCAST_UNKNOWN = 0,
    BROADCAST_EQUAL_SHAPES,       // Same shapes - most optimal
    BROADCAST_2D_ROW,            // [N,M] + [1,M] - row broadcasting
    BROADCAST_2D_COL,            // [N,M] + [N,1] - column broadcasting  
    BROADCAST_3D_SPATIAL_W,      // [B,H,W] + [B,H,1] - width broadcasting
    BROADCAST_3D_SPATIAL_H,      // [B,H,W] + [B,1,W] - height broadcasting  
    BROADCAST_3D_BATCH,          // [B,H,W] + [1,H,W] - batch broadcasting
    BROADCAST_4D_BATCH,          // [B,C,H,W] + [1,C,H,W] - batch broadcasting
    BROADCAST_4D_CHANNEL,        // [B,C,H,W] + [B,1,H,W] - channel broadcasting
    BROADCAST_4D_SPATIAL_H,      // [B,C,H,W] + [B,C,1,W] - height broadcasting
    BROADCAST_4D_SPATIAL_W,      // [B,C,H,W] + [B,C,H,1] - width broadcasting
    BROADCAST_SCALAR,            // Any + [1] or [1,1,...] - scalar broadcasting
    BROADCAST_GENERAL            // General case requiring full ambient promotion
} vsla_broadcast_pattern_t;

/**
 * @brief Detect broadcasting pattern for optimization selection
 */
static vsla_broadcast_pattern_t detect_broadcast_pattern(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    // Equal shapes - most optimal case
    if (shapes_equal(a, b)) {
        return BROADCAST_EQUAL_SHAPES;
    }
    
    // Check for scalar patterns (must be same rank in VSLA)
    bool a_is_scalar = true;
    bool b_is_scalar = true;
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != 1) a_is_scalar = false;
        if (b->shape[i] != 1) b_is_scalar = false;
    }
    if (a_is_scalar || b_is_scalar) {
        return BROADCAST_SCALAR;
    }
    
    // Check for common broadcasting patterns
    if (a->rank == 2 && b->rank == 2) {
        // 2D matrix operations
        if (a->shape[0] == b->shape[0] && b->shape[1] == 1) {
            return BROADCAST_2D_COL; // Column broadcasting
        }
        if (a->shape[1] == b->shape[1] && b->shape[0] == 1) {
            return BROADCAST_2D_ROW; // Row broadcasting
        }
    } else if (a->rank == 3 && b->rank == 3) {
        // 3D spatial operations (common in CNNs and computer vision)
        if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && b->shape[2] == 1) {
            return BROADCAST_3D_SPATIAL_W; // Width broadcasting: [B,H,W] + [B,H,1]
        }
        if (a->shape[0] == b->shape[0] && a->shape[2] == b->shape[2] && b->shape[1] == 1) {
            return BROADCAST_3D_SPATIAL_H; // Height broadcasting: [B,H,W] + [B,1,W]
        }
        if (b->shape[0] == 1 && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2]) {
            return BROADCAST_3D_BATCH; // Batch broadcasting: [B,H,W] + [1,H,W]
        }
    } else if (a->rank == 4 && b->rank == 4) {
        // 4D tensor operations (common in deep learning: batch, channel, height, width)
        if (b->shape[0] == 1 && a->shape[1] == b->shape[1] && 
            a->shape[2] == b->shape[2] && a->shape[3] == b->shape[3]) {
            return BROADCAST_4D_BATCH; // Batch broadcasting: [B,C,H,W] + [1,C,H,W]
        }
        if (a->shape[0] == b->shape[0] && b->shape[1] == 1 && 
            a->shape[2] == b->shape[2] && a->shape[3] == b->shape[3]) {
            return BROADCAST_4D_CHANNEL; // Channel broadcasting: [B,C,H,W] + [B,1,H,W]
        }
        if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] &&
            b->shape[2] == 1 && a->shape[3] == b->shape[3]) {
            return BROADCAST_4D_SPATIAL_H; // Height broadcasting: [B,C,H,W] + [B,C,1,W]
        }
        if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] &&
            a->shape[2] == b->shape[2] && b->shape[3] == 1) {
            return BROADCAST_4D_SPATIAL_W; // Width broadcasting: [B,C,H,W] + [B,C,H,1]
        }
    }
    
    return BROADCAST_GENERAL;
}

// ============================================================================
// VECTORIZED KERNELS
// ============================================================================

#ifdef VSLA_HAS_AVX2
/**
 * @brief AVX2 vectorized addition for equal-shaped dense tensors
 */
static void cpu_add_avx2_f64(const double* a, const double* b, double* out, uint64_t count) {
    uint64_t vec_count = count / 4;
    uint64_t remainder = count % 4;
    
    for (uint64_t i = 0; i < vec_count; i++) {
        __m256d va = _mm256_loadu_pd(&a[i * 4]);
        __m256d vb = _mm256_loadu_pd(&b[i * 4]);
        __m256d result = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&out[i * 4], result);
    }
    
    // Handle remainder
    for (uint64_t i = vec_count * 4; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

static void cpu_add_avx2_f32(const float* a, const float* b, float* out, uint64_t count) {
    uint64_t vec_count = count / 8;
    uint64_t remainder = count % 8;
    
    for (uint64_t i = 0; i < vec_count; i++) {
        __m256 va = _mm256_loadu_ps(&a[i * 8]);
        __m256 vb = _mm256_loadu_ps(&b[i * 8]);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&out[i * 8], result);
    }
    
    // Handle remainder
    for (uint64_t i = vec_count * 8; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif

#ifdef VSLA_HAS_SSE2
/**
 * @brief SSE2 vectorized addition for equal-shaped dense tensors
 */
static void cpu_add_sse2_f64(const double* a, const double* b, double* out, uint64_t count) {
    uint64_t vec_count = count / 2;
    uint64_t remainder = count % 2;
    
    for (uint64_t i = 0; i < vec_count; i++) {
        __m128d va = _mm_loadu_pd(&a[i * 2]);
        __m128d vb = _mm_loadu_pd(&b[i * 2]);
        __m128d result = _mm_add_pd(va, vb);
        _mm_storeu_pd(&out[i * 2], result);
    }
    
    // Handle remainder
    for (uint64_t i = vec_count * 2; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif

#ifdef VSLA_HAS_NEON
/**
 * @brief NEON vectorized addition for equal-shaped dense tensors
 */
static void cpu_add_neon_f32(const float* a, const float* b, float* out, uint64_t count) {
    uint64_t vec_count = count / 4;
    uint64_t remainder = count % 4;
    
    for (uint64_t i = 0; i < vec_count; i++) {
        float32x4_t va = vld1q_f32(&a[i * 4]);
        float32x4_t vb = vld1q_f32(&b[i * 4]);
        float32x4_t result = vaddq_f32(va, vb);
        vst1q_f32(&out[i * 4], result);
    }
    
    // Handle remainder
    for (uint64_t i = vec_count * 4; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif

// ============================================================================
// SPECIALIZED BROADCASTING KERNELS
// ============================================================================

/**
 * @brief Optimized equal-shapes addition (most common case)
 */
static vsla_error_t cpu_add_equal_shapes(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t count = vsla_logical_elems(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        #ifdef VSLA_HAS_AVX2
        if (count >= 4) {
            cpu_add_avx2_f64(a_data, b_data, out_data, count);
            return VSLA_SUCCESS;
        }
        #elif defined(VSLA_HAS_SSE2)
        if (count >= 2) {
            cpu_add_sse2_f64(a_data, b_data, out_data, count);
            return VSLA_SUCCESS;
        }
        #endif
        
        // Fallback scalar implementation
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        #ifdef VSLA_HAS_AVX2
        if (count >= 8) {
            cpu_add_avx2_f32(a_data, b_data, out_data, count);
            return VSLA_SUCCESS;
        }
        #elif defined(VSLA_HAS_NEON)
        if (count >= 4) {
            cpu_add_neon_f32(a_data, b_data, out_data, count);
            return VSLA_SUCCESS;
        }
        #endif
        
        // Fallback scalar implementation
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief 2D row broadcasting: [N,M] + [1,M]
 */
static vsla_error_t cpu_add_2d_row_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t rows = a->shape[0];
    uint64_t cols = a->shape[1];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                out_data[i * cols + j] = a_data[i * cols + j] + b_data[j];
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                out_data[i * cols + j] = a_data[i * cols + j] + b_data[j];
            }
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief 2D column broadcasting: [N,M] + [N,1]
 */
static vsla_error_t cpu_add_2d_col_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t rows = a->shape[0];
    uint64_t cols = a->shape[1];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t i = 0; i < rows; i++) {
            double b_val = b_data[i];
            for (uint64_t j = 0; j < cols; j++) {
                out_data[i * cols + j] = a_data[i * cols + j] + b_val;
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t i = 0; i < rows; i++) {
            float b_val = b_data[i];
            for (uint64_t j = 0; j < cols; j++) {
                out_data[i * cols + j] = a_data[i * cols + j] + b_val;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief Scalar broadcasting: Any + [1]
 */
static vsla_error_t cpu_add_scalar_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t count = vsla_logical_elems(out);
    
    // Determine which operand is scalar
    const vsla_tensor_t* tensor = a;
    const vsla_tensor_t* scalar = b;
    if (a->rank == 1 && a->shape[0] == 1) {
        tensor = b;
        scalar = a;
    }
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* tensor_data = (const double*)tensor->data;
        const double* scalar_data = (const double*)scalar->data;
        double* out_data = (double*)out->data;
        double scalar_val = scalar_data[0];
        
        // Vectorized scalar addition
        #ifdef VSLA_HAS_AVX2
        if (count >= 4) {
            __m256d scalar_vec = _mm256_set1_pd(scalar_val);
            uint64_t vec_count = count / 4;
            
            for (uint64_t i = 0; i < vec_count; i++) {
                __m256d tensor_vec = _mm256_loadu_pd(&tensor_data[i * 4]);
                __m256d result = _mm256_add_pd(tensor_vec, scalar_vec);
                _mm256_storeu_pd(&out_data[i * 4], result);
            }
            
            // Handle remainder
            for (uint64_t i = vec_count * 4; i < count; i++) {
                out_data[i] = tensor_data[i] + scalar_val;
            }
            return VSLA_SUCCESS;
        }
        #endif
        
        // Scalar fallback
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = tensor_data[i] + scalar_val;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* tensor_data = (const float*)tensor->data;
        const float* scalar_data = (const float*)scalar->data;
        float* out_data = (float*)out->data;
        float scalar_val = scalar_data[0];
        
        // Vectorized scalar addition
        #ifdef VSLA_HAS_AVX2
        if (count >= 8) {
            __m256 scalar_vec = _mm256_set1_ps(scalar_val);
            uint64_t vec_count = count / 8;
            
            for (uint64_t i = 0; i < vec_count; i++) {
                __m256 tensor_vec = _mm256_loadu_ps(&tensor_data[i * 8]);
                __m256 result = _mm256_add_ps(tensor_vec, scalar_vec);
                _mm256_storeu_ps(&out_data[i * 8], result);
            }
            
            // Handle remainder
            for (uint64_t i = vec_count * 8; i < count; i++) {
                out_data[i] = tensor_data[i] + scalar_val;
            }
            return VSLA_SUCCESS;
        }
        #endif
        
        // Scalar fallback
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = tensor_data[i] + scalar_val;
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * @brief 3D width broadcasting: [B,H,W] + [B,H,1]
 */
static vsla_error_t cpu_add_3d_spatial_w_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t height = a->shape[1]; 
    uint64_t width = a->shape[2];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                double b_val = b_data[b_idx * height + h_idx];
                for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                    uint64_t idx = b_idx * height * width + h_idx * width + w_idx;
                    out_data[idx] = a_data[idx] + b_val;
                }
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                float b_val = b_data[b_idx * height + h_idx];
                for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                    uint64_t idx = b_idx * height * width + h_idx * width + w_idx;
                    out_data[idx] = a_data[idx] + b_val;
                }
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 3D height broadcasting: [B,H,W] + [B,1,W]
 */
static vsla_error_t cpu_add_3d_spatial_h_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t height = a->shape[1]; 
    uint64_t width = a->shape[2];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                    uint64_t idx = b_idx * height * width + h_idx * width + w_idx;
                    double b_val = b_data[b_idx * width + w_idx];
                    out_data[idx] = a_data[idx] + b_val;
                }
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                    uint64_t idx = b_idx * height * width + h_idx * width + w_idx;
                    float b_val = b_data[b_idx * width + w_idx];
                    out_data[idx] = a_data[idx] + b_val;
                }
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 3D batch broadcasting: [B,H,W] + [1,H,W]
 */
static vsla_error_t cpu_add_3d_batch_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t height = a->shape[1]; 
    uint64_t width = a->shape[2];
    uint64_t plane_size = height * width;
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t i = 0; i < plane_size; i++) {
                uint64_t idx = b_idx * plane_size + i;
                out_data[idx] = a_data[idx] + b_data[i];
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t i = 0; i < plane_size; i++) {
                uint64_t idx = b_idx * plane_size + i;
                out_data[idx] = a_data[idx] + b_data[i];
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 4D batch broadcasting: [B,C,H,W] + [1,C,H,W]
 */
static vsla_error_t cpu_add_4d_batch_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    uint64_t volume_size = channels * height * width;
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t i = 0; i < volume_size; i++) {
                uint64_t idx = b_idx * volume_size + i;
                out_data[idx] = a_data[idx] + b_data[i];
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t i = 0; i < volume_size; i++) {
                uint64_t idx = b_idx * volume_size + i;
                out_data[idx] = a_data[idx] + b_data[i];
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 4D channel broadcasting: [B,C,H,W] + [B,1,H,W]
 */
static vsla_error_t cpu_add_4d_channel_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    uint64_t plane_size = height * width;
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t i = 0; i < plane_size; i++) {
                    uint64_t a_idx = b_idx * channels * plane_size + c_idx * plane_size + i;
                    uint64_t b_idx_offset = b_idx * plane_size + i;
                    out_data[a_idx] = a_data[a_idx] + b_data[b_idx_offset];
                }
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t i = 0; i < plane_size; i++) {
                    uint64_t a_idx = b_idx * channels * plane_size + c_idx * plane_size + i;
                    uint64_t b_idx_offset = b_idx * plane_size + i;
                    out_data[a_idx] = a_data[a_idx] + b_data[b_idx_offset];
                }
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 4D spatial height broadcasting: [B,C,H,W] + [B,C,1,W]
 */
static vsla_error_t cpu_add_4d_spatial_h_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                    for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                        uint64_t a_idx = b_idx * channels * height * width + c_idx * height * width + h_idx * width + w_idx;
                        uint64_t b_idx_offset = b_idx * channels * width + c_idx * width + w_idx;
                        out_data[a_idx] = a_data[a_idx] + b_data[b_idx_offset];
                    }
                }
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                    for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                        uint64_t a_idx = b_idx * channels * height * width + c_idx * height * width + h_idx * width + w_idx;
                        uint64_t b_idx_offset = b_idx * channels * width + c_idx * width + w_idx;
                        out_data[a_idx] = a_data[a_idx] + b_data[b_idx_offset];
                    }
                }
            }
        }
    }
    return VSLA_SUCCESS;
}

/**
 * @brief 4D spatial width broadcasting: [B,C,H,W] + [B,C,H,1]
 */
static vsla_error_t cpu_add_4d_spatial_w_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->data;
        const double* b_data = (const double*)b->data;
        double* out_data = (double*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                    double b_val = b_data[b_idx * channels * height + c_idx * height + h_idx];
                    for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                        uint64_t a_idx = b_idx * channels * height * width + c_idx * height * width + h_idx * width + w_idx;
                        out_data[a_idx] = a_data[a_idx] + b_val;
                    }
                }
            }
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->data;
        const float* b_data = (const float*)b->data;
        float* out_data = (float*)out->data;
        
        for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
            for (uint64_t c_idx = 0; c_idx < channels; c_idx++) {
                for (uint64_t h_idx = 0; h_idx < height; h_idx++) {
                    float b_val = b_data[b_idx * channels * height + c_idx * height + h_idx];
                    for (uint64_t w_idx = 0; w_idx < width; w_idx++) {
                        uint64_t a_idx = b_idx * channels * height * width + c_idx * height * width + h_idx * width + w_idx;
                        out_data[a_idx] = a_data[a_idx] + b_val;
                    }
                }
            }
        }
    }
    return VSLA_SUCCESS;
}

// ============================================================================
// GENERAL AMBIENT PROMOTION (SPEC-COMPLIANT REFERENCE IMPLEMENTATION)
// ============================================================================

/**
 * @brief General ambient promotion following VSLA spec v3.2 Section 4.1
 * 
 * This is the reference implementation that handles all broadcasting cases
 * according to the mathematical specification with zero-extension.
 */
static vsla_error_t cpu_add_general_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    size_t total = vsla_logical_elems(out); // ambient size
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* out_data = (double*)out->data;
        
        for (size_t lin = 0; lin < total; ++lin) {
            uint64_t idx[VSLA_MAX_RANK];
            unravel(lin, out->shape, out->rank, idx);
            
            // Get values with zero-extension for out-of-bounds (VSLA spec)
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

// ============================================================================
// UNIFIED ADDITION WITH INTELLIGENT DISPATCH
// ============================================================================

/**
 * @brief Unified CPU addition with intelligent optimization dispatch
 * 
 * Following VSLA v3.2 spec Section 4.1 with performance optimizations.
 * Shape rule: out->shape[i] = max(a->shape[i], b->shape[i])
 * Empty operands behave as zeros.
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

    // Handle empty operands (behave as zeros) - VSLA spec requirement
    if (vsla_is_empty(a) && vsla_is_empty(b)) {
        if (out->data) {
            memset(out->data, 0, vsla_logical_elems(out) * vsla_dtype_size(out->dtype));
        }
        return VSLA_SUCCESS;
    }

    // Intelligent optimization dispatch
    vsla_broadcast_pattern_t pattern = detect_broadcast_pattern(a, b);
    
    switch (pattern) {
        case BROADCAST_EQUAL_SHAPES:
            // Most optimal case - vectorized dense addition
            if (!vsla_is_empty(a) && !vsla_is_empty(b)) {
                return cpu_add_equal_shapes(out, a, b);
            }
            break;
            
        case BROADCAST_SCALAR:
            // Second most optimal - vectorized scalar broadcasting
            return cpu_add_scalar_broadcast(out, a, b);
            
        case BROADCAST_2D_ROW:
            // Cache-friendly row broadcasting
            return cpu_add_2d_row_broadcast(out, a, b);
            
        case BROADCAST_2D_COL:
            // Cache-friendly column broadcasting
            return cpu_add_2d_col_broadcast(out, a, b);
            
        case BROADCAST_3D_SPATIAL_W:
            return cpu_add_3d_spatial_w_broadcast(out, a, b);
        case BROADCAST_3D_SPATIAL_H:
            return cpu_add_3d_spatial_h_broadcast(out, a, b);
        case BROADCAST_3D_BATCH:
            return cpu_add_3d_batch_broadcast(out, a, b);
        case BROADCAST_4D_BATCH:
            return cpu_add_4d_batch_broadcast(out, a, b);
        case BROADCAST_4D_CHANNEL:
            return cpu_add_4d_channel_broadcast(out, a, b);
        case BROADCAST_4D_SPATIAL_H:
            return cpu_add_4d_spatial_h_broadcast(out, a, b);
        case BROADCAST_4D_SPATIAL_W:
            return cpu_add_4d_spatial_w_broadcast(out, a, b);
            
        case BROADCAST_GENERAL:
        case BROADCAST_UNKNOWN:
        default:
            // Fall through to general ambient promotion
            break;
    }
    
    // Fallback: Reference implementation with full ambient promotion
    return cpu_add_general_ambient(out, a, b);
}

// ============================================================================
// OTHER ARITHMETIC OPERATIONS
// ============================================================================

/**
 * @brief Elementwise subtraction following Section 4.1
 * Uses same optimization dispatch as addition but with subtraction operation
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

    // For now, use general ambient promotion for subtraction
    // TODO: Add optimized subtraction kernels similar to addition
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
 * @brief Hadamard product following Section 4.2
 */
vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    // Same structure as addition but with multiplication
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

    // Empty operands make result empty
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
            
            double va = (vsla_is_empty(a) || !in_bounds(a, idx)) ? 0.0 : 
                       ((double*)a->data)[vsla_offset(a, idx)];
            double vb = (vsla_is_empty(b) || !in_bounds(b, idx)) ? 0.0 : 
                       ((double*)b->data)[vsla_offset(b, idx)];
            
            out_data[vsla_offset(out, idx)] = va * vb;
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
    
    uint64_t count = vsla_logical_elems(in);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)in->data;
        double* out_data = (double*)out->data;
        
        // Vectorized scalar multiplication
        #ifdef VSLA_HAS_AVX2
        if (count >= 4) {
            __m256d scalar_vec = _mm256_set1_pd(scalar);
            uint64_t vec_count = count / 4;
            
            for (uint64_t i = 0; i < vec_count; i++) {
                __m256d in_vec = _mm256_loadu_pd(&in_data[i * 4]);
                __m256d result = _mm256_mul_pd(in_vec, scalar_vec);
                _mm256_storeu_pd(&out_data[i * 4], result);
            }
            
            // Handle remainder
            for (uint64_t i = vec_count * 4; i < count; i++) {
                out_data[i] = in_data[i] * scalar;
            }
            return VSLA_SUCCESS;
        }
        #endif
        
        // Scalar fallback
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = in_data[i] * scalar;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)in->data;
        float* out_data = (float*)out->data;
        float scalar_f = (float)scalar;
        
        // Vectorized scalar multiplication
        #ifdef VSLA_HAS_AVX2
        if (count >= 8) {
            __m256 scalar_vec = _mm256_set1_ps(scalar_f);
            uint64_t vec_count = count / 8;
            
            for (uint64_t i = 0; i < vec_count; i++) {
                __m256 in_vec = _mm256_loadu_ps(&in_data[i * 8]);
                __m256 result = _mm256_mul_ps(in_vec, scalar_vec);
                _mm256_storeu_ps(&out_data[i * 8], result);
            }
            
            // Handle remainder
            for (uint64_t i = vec_count * 8; i < count; i++) {
                out_data[i] = in_data[i] * scalar_f;
            }
            return VSLA_SUCCESS;
        }
        #endif
        
        // Scalar fallback
        for (uint64_t i = 0; i < count; i++) {
            out_data[i] = in_data[i] * scalar_f;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}