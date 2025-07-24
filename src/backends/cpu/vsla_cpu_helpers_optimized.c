/**
 * @file vsla_cpu_helpers_optimized.c
 * @brief Optimized VSLA CPU backend helper functions for cache performance
 * 
 * This file implements optimized versions of stride computation and memory access
 * patterns to address the 20-40% performance gap in multi-dimensional operations.
 * 
 * Key optimizations:
 * 1. Shape-based strides for dense operations (cache-friendly)
 * 2. Stride caching to avoid recomputation 
 * 3. Specialized fast paths for common broadcast patterns
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// SIMD intrinsics for vectorization
#ifdef __AVX2__
#include <immintrin.h>
#define VSLA_HAS_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define VSLA_HAS_SSE2 1
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#define VSLA_HAS_NEON 1
#endif

// Cache for stride computations to avoid recomputation
typedef struct {
    uint64_t shape_strides[VSLA_MAX_RANK];    // Dense, cache-friendly strides
    uint64_t capacity_strides[VSLA_MAX_RANK]; // Current capacity-based strides
    uint64_t shape_hash;                       // Hash of current shape
    uint64_t capacity_hash;                    // Hash of current capacity
    bool shape_strides_valid;
    bool capacity_strides_valid;
} vsla_stride_cache_t;

// Simple hash function for stride validation
static uint64_t compute_shape_hash(const uint64_t* dims, uint8_t rank) {
    uint64_t hash = 0x811c9dc5; // FNV-1a basis
    for (uint8_t i = 0; i < rank; i++) {
        hash ^= dims[i];
        hash *= 0x01000193; // FNV-1a prime
    }
    return hash;
}

/**
 * Compute shape-based strides (cache-friendly, no gaps)
 * This eliminates memory gaps between logical elements for better cache locality
 */
void compute_shape_strides(const vsla_tensor_t* t, uint64_t* strides) {
    uint64_t acc = 1;
    for (int j = t->rank - 1; j >= 0; --j) {
        strides[j] = acc;
        acc *= t->shape[j];  // Use shape, not capacity!
    }
}

/**
 * Compute capacity-based strides (current method, for growth operations)
 * Preserves current behavior for operations that need capacity-based layout
 */
void compute_capacity_strides(const vsla_tensor_t* t, uint64_t* strides) {
    uint64_t acc = 1;
    for (int j = t->rank - 1; j >= 0; --j) {
        strides[j] = acc;
        acc *= t->cap[j];
    }
}

/**
 * Get cached shape-based strides with automatic invalidation
 */
void get_cached_shape_strides(const vsla_tensor_t* t, uint64_t* strides, vsla_stride_cache_t* cache) {
    uint64_t current_hash = compute_shape_hash(t->shape, t->rank);
    
    if (!cache->shape_strides_valid || cache->shape_hash != current_hash) {
        compute_shape_strides(t, cache->shape_strides);
        cache->shape_hash = current_hash;
        cache->shape_strides_valid = true;
    }
    
    memcpy(strides, cache->shape_strides, t->rank * sizeof(uint64_t));
}

/**
 * Get cached capacity-based strides with automatic invalidation
 */
void get_cached_capacity_strides(const vsla_tensor_t* t, uint64_t* strides, vsla_stride_cache_t* cache) {
    uint64_t current_hash = compute_shape_hash(t->cap, t->rank);
    
    if (!cache->capacity_strides_valid || cache->capacity_hash != current_hash) {
        compute_capacity_strides(t, cache->capacity_strides);
        cache->capacity_hash = current_hash;
        cache->capacity_strides_valid = true;
    }
    
    memcpy(strides, cache->capacity_strides, t->rank * sizeof(uint64_t));
}

/**
 * Determine optimal stride type for given operation
 * Returns true if shape-based strides should be used, false for capacity-based
 */
bool should_use_shape_strides(const vsla_tensor_t* a, const vsla_tensor_t* b, const vsla_tensor_t* out) {
    // Use shape-based strides for:
    // 1. Dense operations (all tensors at capacity)
    // 2. Broadcasting scenarios (shape mismatches common)
    // 3. Large tensors (cache efficiency matters more)
    
    // Check if tensors are at capacity (no growth expected)
    bool a_at_capacity = true, b_at_capacity = true, out_at_capacity = true;
    
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] < a->cap[i]) a_at_capacity = false;
    }
    for (uint8_t i = 0; i < b->rank; i++) {
        if (b->shape[i] < b->cap[i]) b_at_capacity = false;
    }
    for (uint8_t i = 0; i < out->rank; i++) {
        if (out->shape[i] < out->cap[i]) out_at_capacity = false;
    }
    
    // Use shape strides if most tensors are at capacity or if large enough
    uint64_t total_elements = 1;
    for (uint8_t i = 0; i < out->rank; i++) {
        total_elements *= out->shape[i];
    }
    
    return (a_at_capacity && b_at_capacity && out_at_capacity) || 
           (total_elements > 1000); // Threshold for cache efficiency
}

/**
 * Optimized offset computation using shape-based strides
 * Eliminates memory gaps for better cache performance
 */
uint64_t vsla_offset_shape_based(const vsla_tensor_t* t, const uint64_t* idx) {
    uint64_t strides[VSLA_MAX_RANK];
    compute_shape_strides(t, strides);
    
    uint64_t off = 0;
    for (int j = 0; j < t->rank; ++j) {
        off += idx[j] * strides[j];
    }
    return off;
}

/**
 * Check if broadcast pattern is common and can use specialized kernel
 */
typedef enum {
    BROADCAST_UNKNOWN = 0,
    BROADCAST_2D_ROW,         // [N,M] + [1,M]
    BROADCAST_2D_COL,         // [N,M] + [N,1]  
    BROADCAST_3D_SPATIAL_W,   // [B,H,W] + [B,H,1] - width broadcasting
    BROADCAST_3D_SPATIAL_H,   // [B,H,W] + [B,1,W] - height broadcasting  
    BROADCAST_3D_BATCH,       // [B,H,W] + [1,H,W] - batch broadcasting
    BROADCAST_4D_BATCH,       // [B,C,H,W] + [1,C,H,W] - batch broadcasting
    BROADCAST_4D_CHANNEL,     // [B,C,H,W] + [B,1,H,W] - channel broadcasting
    BROADCAST_4D_SPATIAL_H,   // [B,C,H,W] + [B,C,1,W] - height broadcasting
    BROADCAST_4D_SPATIAL_W,   // [B,C,H,W] + [B,C,H,1] - width broadcasting
    BROADCAST_SCALAR          // Any + [1] or [1,1,...]
} vsla_broadcast_pattern_t;

vsla_broadcast_pattern_t detect_broadcast_pattern(const vsla_tensor_t* a, const vsla_tensor_t* b) {
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
    
    // Check for scalar broadcasting (one tensor has all dimensions = 1)
    bool b_is_scalar = true;
    for (uint8_t i = 0; i < b->rank; i++) {
        if (b->shape[i] != 1) {
            b_is_scalar = false;
            break;
        }
    }
    if (b_is_scalar) return BROADCAST_SCALAR;
    
    return BROADCAST_UNKNOWN;
}

/**
 * Calculate memory access efficiency ratio
 * Higher values indicate better cache performance
 */
double calculate_cache_efficiency(const vsla_tensor_t* t) {
    uint64_t shape_size = 1, capacity_size = 1;
    for (uint8_t i = 0; i < t->rank; i++) {
        shape_size *= t->shape[i];
        capacity_size *= t->cap[i];
    }
    return (double)shape_size / capacity_size;
}

/**
 * Initialize stride cache
 */
void init_stride_cache(vsla_stride_cache_t* cache) {
    memset(cache, 0, sizeof(vsla_stride_cache_t));
}

/**
 * Invalidate stride cache (call when tensor shapes change)
 */
void invalidate_stride_cache(vsla_stride_cache_t* cache) {
    cache->shape_strides_valid = false;
    cache->capacity_strides_valid = false;
}

// Import vsla_dtype_size function
extern size_t vsla_dtype_size(vsla_dtype_t dtype);

/**
 * Optimized 2D row broadcasting: [N,M] + [1,M] → [N,M]
 * Eliminates coordinate transformation by direct stride arithmetic
 */
vsla_error_t cpu_add_2d_row_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
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
        
        // SIMD-optimized inner loop for better performance
#ifdef VSLA_HAS_AVX2
        // AVX2: Process 4 doubles at a time
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 4) * 4;
        
        for (; col < simd_end; col += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_data[a_row_offset + col]);
            __m256d b_vec = _mm256_loadu_pd(&b_data[col]);
            __m256d result = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_data[col];
        }
#elif defined(VSLA_HAS_SSE2)
        // SSE2: Process 2 doubles at a time
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 2) * 2;
        
        for (; col < simd_end; col += 2) {
            __m128d a_vec = _mm_loadu_pd(&a_data[a_row_offset + col]);
            __m128d b_vec = _mm_loadu_pd(&b_data[col]);
            __m128d result = _mm_add_pd(a_vec, b_vec);
            _mm_storeu_pd(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_data[col];
        }
#elif defined(VSLA_HAS_NEON)
        // ARM NEON: Process 2 doubles at a time
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 2) * 2;
        
        for (; col < simd_end; col += 2) {
            float64x2_t a_vec = vld1q_f64(&a_data[a_row_offset + col]);
            float64x2_t b_vec = vld1q_f64(&b_data[col]);
            float64x2_t result = vaddq_f64(a_vec, b_vec);
            vst1q_f64(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_data[col];
        }
#else
        // Fallback: scalar code
        for (uint64_t col = 0; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_data[col];
        }
#endif
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized 2D column broadcasting: [N,M] + [N,1] → [N,M]
 */
vsla_error_t cpu_add_2d_col_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
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
        
        // SIMD-optimized: add same value to entire row
#ifdef VSLA_HAS_AVX2
        // AVX2: Process 4 doubles at a time with broadcast
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 4) * 4;
        __m256d b_vec = _mm256_set1_pd(b_val); // Broadcast scalar to vector
        
        for (; col < simd_end; col += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_data[a_row_offset + col]);
            __m256d result = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_val;
        }
#elif defined(VSLA_HAS_SSE2)
        // SSE2: Process 2 doubles at a time with broadcast
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 2) * 2;
        __m128d b_vec = _mm_set1_pd(b_val); // Broadcast scalar to vector
        
        for (; col < simd_end; col += 2) {
            __m128d a_vec = _mm_loadu_pd(&a_data[a_row_offset + col]);
            __m128d result = _mm_add_pd(a_vec, b_vec);
            _mm_storeu_pd(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_val;
        }
#elif defined(VSLA_HAS_NEON)
        // ARM NEON: Process 2 doubles at a time with broadcast
        uint64_t col = 0;
        const uint64_t simd_end = (cols / 2) * 2;
        float64x2_t b_vec = vdupq_n_f64(b_val); // Broadcast scalar to vector
        
        for (; col < simd_end; col += 2) {
            float64x2_t a_vec = vld1q_f64(&a_data[a_row_offset + col]);
            float64x2_t result = vaddq_f64(a_vec, b_vec);
            vst1q_f64(&out_data[out_row_offset + col], result);
        }
        
        // Handle remaining elements
        for (; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_val;
        }
#else
        // Fallback: scalar code
        for (uint64_t col = 0; col < cols; col++) {
            out_data[out_row_offset + col] = a_data[a_row_offset + col] + b_val;
        }
#endif
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized scalar broadcasting: [shape] + [1] → [shape]
 */
vsla_error_t cpu_add_scalar_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t total = 1;
    for (uint8_t i = 0; i < a->rank; i++) {
        total *= a->shape[i];
    }
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    double b_val = b_data[0]; // Scalar value
    
    // Perfect sequential access, highly vectorizable
#ifdef VSLA_HAS_AVX2
    // AVX2: Process 4 doubles at a time
    uint64_t i = 0;
    const uint64_t simd_end = (total / 4) * 4;
    __m256d b_vec = _mm256_set1_pd(b_val); // Broadcast scalar to vector
    
    for (; i < simd_end; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a_data[i]);
        __m256d result = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(&out_data[i], result);
    }
    
    // Handle remaining elements
    for (; i < total; i++) {
        out_data[i] = a_data[i] + b_val;
    }
#elif defined(VSLA_HAS_SSE2)
    // SSE2: Process 2 doubles at a time
    uint64_t i = 0;
    const uint64_t simd_end = (total / 2) * 2;
    __m128d b_vec = _mm_set1_pd(b_val); // Broadcast scalar to vector
    
    for (; i < simd_end; i += 2) {
        __m128d a_vec = _mm_loadu_pd(&a_data[i]);
        __m128d result = _mm_add_pd(a_vec, b_vec);
        _mm_storeu_pd(&out_data[i], result);
    }
    
    // Handle remaining elements
    for (; i < total; i++) {
        out_data[i] = a_data[i] + b_val;
    }
#elif defined(VSLA_HAS_NEON)
    // ARM NEON: Process 2 doubles at a time
    uint64_t i = 0;
    const uint64_t simd_end = (total / 2) * 2;
    float64x2_t b_vec = vdupq_n_f64(b_val); // Broadcast scalar to vector
    
    for (; i < simd_end; i += 2) {
        float64x2_t a_vec = vld1q_f64(&a_data[i]);
        float64x2_t result = vaddq_f64(a_vec, b_vec);
        vst1q_f64(&out_data[i], result);
    }
    
    // Handle remaining elements
    for (; i < total; i++) {
        out_data[i] = a_data[i] + b_val;
    }
#else
    // Fallback: scalar code
    for (uint64_t i = 0; i < total; i++) {
        out_data[i] = a_data[i] + b_val;
    }
#endif
    
    return VSLA_SUCCESS;
}

// Import existing functions
extern bool vsla_is_empty(const vsla_tensor_t* t);
extern bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx);

/**
 * Optimized ambient promotion using shape-based strides
 * Reduces coordinate transformation overhead
 */
vsla_error_t cpu_add_optimized_ambient(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t total = 1;
    for (uint8_t i = 0; i < out->rank; i++) {
        total *= out->shape[i];
    }
    
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
 * Optimized 3D spatial width broadcasting: [B,H,W] + [B,H,1] → [B,H,W]
 * Common in CNN feature map operations
 */
vsla_error_t cpu_add_3d_spatial_w_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t batch = a->shape[0];
    uint64_t height = a->shape[1];
    uint64_t width = a->shape[2];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    // Use shape-based strides for optimal cache access
    uint64_t a_strides[3], out_strides[3];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    // Broadcast across width dimension
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        for (uint64_t h = 0; h < height; h++) {
            uint64_t a_plane_offset = b_idx * a_strides[0] + h * a_strides[1];
            uint64_t out_plane_offset = b_idx * out_strides[0] + h * out_strides[1];
            uint64_t b_offset = b_idx * height + h; // [B,H,1] layout
            
            double b_val = b_data[b_offset]; // Broadcast value
            
            // SIMD-optimized: add same value to entire row (width dimension)
#ifdef VSLA_HAS_AVX2
            // AVX2: Process 4 doubles at a time with broadcast
            uint64_t w = 0;
            const uint64_t simd_end = (width / 4) * 4;
            __m256d b_vec = _mm256_set1_pd(b_val); // Broadcast scalar to vector
            
            for (; w < simd_end; w += 4) {
                __m256d a_vec = _mm256_loadu_pd(&a_data[a_plane_offset + w]);
                __m256d result = _mm256_add_pd(a_vec, b_vec);
                _mm256_storeu_pd(&out_data[out_plane_offset + w], result);
            }
            
            // Handle remaining elements
            for (; w < width; w++) {
                out_data[out_plane_offset + w] = a_data[a_plane_offset + w] + b_val;
            }
#elif defined(VSLA_HAS_SSE2)
            // SSE2: Process 2 doubles at a time with broadcast
            uint64_t w = 0;
            const uint64_t simd_end = (width / 2) * 2;
            __m128d b_vec = _mm_set1_pd(b_val); // Broadcast scalar to vector
            
            for (; w < simd_end; w += 2) {
                __m128d a_vec = _mm_loadu_pd(&a_data[a_plane_offset + w]);
                __m128d result = _mm_add_pd(a_vec, b_vec);
                _mm_storeu_pd(&out_data[out_plane_offset + w], result);
            }
            
            // Handle remaining elements
            for (; w < width; w++) {
                out_data[out_plane_offset + w] = a_data[a_plane_offset + w] + b_val;
            }
#elif defined(VSLA_HAS_NEON)
            // ARM NEON: Process 2 doubles at a time with broadcast
            uint64_t w = 0;
            const uint64_t simd_end = (width / 2) * 2;
            float64x2_t b_vec = vdupq_n_f64(b_val); // Broadcast scalar to vector
            
            for (; w < simd_end; w += 2) {
                float64x2_t a_vec = vld1q_f64(&a_data[a_plane_offset + w]);
                float64x2_t result = vaddq_f64(a_vec, b_vec);
                vst1q_f64(&out_data[out_plane_offset + w], result);
            }
            
            // Handle remaining elements
            for (; w < width; w++) {
                out_data[out_plane_offset + w] = a_data[a_plane_offset + w] + b_val;
            }
#else
            // Fallback: scalar code
            for (uint64_t w = 0; w < width; w++) {
                out_data[out_plane_offset + w] = a_data[a_plane_offset + w] + b_val;
            }
#endif
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized 3D spatial height broadcasting: [B,H,W] + [B,1,W] → [B,H,W]
 * Common in CNN spatial operations
 */
vsla_error_t cpu_add_3d_spatial_h_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t batch = a->shape[0];
    uint64_t height = a->shape[1];
    uint64_t width = a->shape[2];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    uint64_t a_strides[3], out_strides[3];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    // Broadcast across height dimension
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t b_batch_offset = b_idx * width; // [B,1,W] layout
        
        for (uint64_t h = 0; h < height; h++) {
            uint64_t a_plane_offset = b_idx * a_strides[0] + h * a_strides[1];
            uint64_t out_plane_offset = b_idx * out_strides[0] + h * out_strides[1];
            
            // Vectorizable: element-wise addition with broadcasted row
            for (uint64_t w = 0; w < width; w++) {
                out_data[out_plane_offset + w] = a_data[a_plane_offset + w] + b_data[b_batch_offset + w];
            }
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized 4D channel broadcasting: [B,C,H,W] + [B,1,H,W] → [B,C,H,W]
 * Common in deep learning: bias addition, normalization
 */
vsla_error_t cpu_add_4d_channel_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    uint64_t a_strides[4], out_strides[4];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    uint64_t spatial_size = height * width;
    
    // Broadcast across channel dimension
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t b_batch_offset = b_idx * spatial_size; // [B,1,H,W] layout
        
        for (uint64_t c = 0; c < channels; c++) {
            uint64_t a_channel_offset = b_idx * a_strides[0] + c * a_strides[1];
            uint64_t out_channel_offset = b_idx * out_strides[0] + c * out_strides[1];
            
            // SIMD-optimized: add broadcasted spatial map to each channel
#ifdef VSLA_HAS_AVX2
            // AVX2: Process 4 doubles at a time
            uint64_t spatial = 0;
            const uint64_t simd_end = (spatial_size / 4) * 4;
            
            for (; spatial < simd_end; spatial += 4) {
                __m256d a_vec = _mm256_loadu_pd(&a_data[a_channel_offset + spatial]);
                __m256d b_vec = _mm256_loadu_pd(&b_data[b_batch_offset + spatial]);
                __m256d result = _mm256_add_pd(a_vec, b_vec);
                _mm256_storeu_pd(&out_data[out_channel_offset + spatial], result);
            }
            
            // Handle remaining elements
            for (; spatial < spatial_size; spatial++) {
                out_data[out_channel_offset + spatial] = a_data[a_channel_offset + spatial] + b_data[b_batch_offset + spatial];
            }
#elif defined(VSLA_HAS_SSE2)
            // SSE2: Process 2 doubles at a time
            uint64_t spatial = 0;
            const uint64_t simd_end = (spatial_size / 2) * 2;
            
            for (; spatial < simd_end; spatial += 2) {
                __m128d a_vec = _mm_loadu_pd(&a_data[a_channel_offset + spatial]);
                __m128d b_vec = _mm_loadu_pd(&b_data[b_batch_offset + spatial]);
                __m128d result = _mm_add_pd(a_vec, b_vec);
                _mm_storeu_pd(&out_data[out_channel_offset + spatial], result);
            }
            
            // Handle remaining elements
            for (; spatial < spatial_size; spatial++) {
                out_data[out_channel_offset + spatial] = a_data[a_channel_offset + spatial] + b_data[b_batch_offset + spatial];
            }
#elif defined(VSLA_HAS_NEON)
            // ARM NEON: Process 2 doubles at a time
            uint64_t spatial = 0;
            const uint64_t simd_end = (spatial_size / 2) * 2;
            
            for (; spatial < simd_end; spatial += 2) {
                float64x2_t a_vec = vld1q_f64(&a_data[a_channel_offset + spatial]);
                float64x2_t b_vec = vld1q_f64(&b_data[b_batch_offset + spatial]);
                float64x2_t result = vaddq_f64(a_vec, b_vec);
                vst1q_f64(&out_data[out_channel_offset + spatial], result);
            }
            
            // Handle remaining elements
            for (; spatial < spatial_size; spatial++) {
                out_data[out_channel_offset + spatial] = a_data[a_channel_offset + spatial] + b_data[b_batch_offset + spatial];
            }
#else
            // Fallback: scalar code
            for (uint64_t spatial = 0; spatial < spatial_size; spatial++) {
                out_data[out_channel_offset + spatial] = a_data[a_channel_offset + spatial] + b_data[b_batch_offset + spatial];
            }
#endif
        }
    }
    
    return VSLA_SUCCESS;
}

/**
 * Optimized 4D batch broadcasting: [B,C,H,W] + [1,C,H,W] → [B,C,H,W]
 * Common in deep learning: applying same transformation across batch
 */
vsla_error_t cpu_add_4d_batch_broadcast(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64) return VSLA_ERROR_INVALID_DTYPE;
    
    uint64_t batch = a->shape[0];
    uint64_t channels = a->shape[1];
    uint64_t height = a->shape[2];
    uint64_t width = a->shape[3];
    
    double* out_data = (double*)out->data;
    const double* a_data = (const double*)a->data;
    const double* b_data = (const double*)b->data;
    
    uint64_t a_strides[4], out_strides[4];
    compute_shape_strides(a, a_strides);
    compute_shape_strides(out, out_strides);
    
    uint64_t channel_size = height * width;
    uint64_t batch_size = channels * channel_size;
    
    // Broadcast across batch dimension
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t a_batch_offset = b_idx * a_strides[0];
        uint64_t out_batch_offset = b_idx * out_strides[0];
        
        // Vectorizable: add same [C,H,W] tensor to each batch element
        for (uint64_t elem = 0; elem < batch_size; elem++) {
            out_data[out_batch_offset + elem] = a_data[a_batch_offset + elem] + b_data[elem];
        }
    }
    
    return VSLA_SUCCESS;
}