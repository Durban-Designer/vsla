/**
 * @file vsla_gpu_types.h
 * @brief GPU-compatible floating-point types for VSLA
 * 
 * This header provides a compatibility layer for floating-point types
 * that works with current CUDA limitations while preparing for future
 * C23 support.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_GPU_TYPES_H
#define VSLA_GPU_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Disable C23 features that cause CUDA compilation issues
 * 
 * These defines prevent system headers from using C23 floating-point
 * types that CUDA doesn't support yet.
 */
#ifdef __CUDACC__
#define __STDC_WANT_IEC_60559_TYPES_EXT__ 0
#define __STDC_WANT_IEC_60559_FUNCS_EXT__ 0
#define __STDC_WANT_IEC_60559_ATTRIBS_EXT__ 0
#define __STDC_WANT_IEC_60559_BFP_EXT__ 0
#define __STDC_WANT_IEC_60559_DFP_EXT__ 0
#define __STDC_WANT_IEC_60559_EXT__ 0
#endif

/**
 * @brief GPU-compatible floating-point types
 * 
 * These types provide a compatibility layer that uses traditional
 * floating-point types now but can be easily migrated to C23
 * exact-width types when CUDA supports them.
 */

#ifdef VSLA_ENABLE_C23_TYPES
    // Future: Use C23 exact-width types (when CUDA supports them)
    typedef _Float32 vsla_gpu_f32_t;
    typedef _Float64 vsla_gpu_f64_t;
    
    #ifdef __STDC_IEC_60559_TYPES__
        typedef _Float128 vsla_gpu_f128_t;
        typedef _Float32x vsla_gpu_f32x_t;
        typedef _Float64x vsla_gpu_f64x_t;
        #define VSLA_GPU_HAS_EXTENDED_PRECISION 1
    #else
        typedef long double vsla_gpu_f128_t;
        typedef double vsla_gpu_f32x_t;
        typedef long double vsla_gpu_f64x_t;
        #define VSLA_GPU_HAS_EXTENDED_PRECISION 0
    #endif
#else
    // Current: Use traditional types (CUDA-compatible)
    typedef float vsla_gpu_f32_t;
    typedef double vsla_gpu_f64_t;
    typedef long double vsla_gpu_f128_t;
    typedef double vsla_gpu_f32x_t;
    typedef long double vsla_gpu_f64x_t;
    #define VSLA_GPU_HAS_EXTENDED_PRECISION 0
#endif

/**
 * @brief Type size constants
 */
#define VSLA_GPU_F32_SIZE sizeof(vsla_gpu_f32_t)
#define VSLA_GPU_F64_SIZE sizeof(vsla_gpu_f64_t)
#define VSLA_GPU_F128_SIZE sizeof(vsla_gpu_f128_t)

/**
 * @brief Precision constants
 */
#define VSLA_GPU_F32_EPSILON 1.19209290e-07f
#define VSLA_GPU_F64_EPSILON 2.2204460492503131e-16
#define VSLA_GPU_F128_EPSILON 1.08420217248550443401e-19L

/**
 * @brief GPU-compatible complex types
 */
typedef struct {
    vsla_gpu_f32_t real;
    vsla_gpu_f32_t imag;
} vsla_gpu_complex32_t;

typedef struct {
    vsla_gpu_f64_t real;
    vsla_gpu_f64_t imag;
} vsla_gpu_complex64_t;

typedef struct {
    vsla_gpu_f128_t real;
    vsla_gpu_f128_t imag;
} vsla_gpu_complex128_t;

/**
 * @brief GPU kernel launch configuration
 */
typedef struct {
    uint32_t block_size_x;
    uint32_t block_size_y;
    uint32_t block_size_z;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    size_t shared_memory_size;
} vsla_gpu_launch_config_t;

/**
 * @brief GPU memory information
 */
typedef struct {
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    int device_id;
    char device_name[256];
} vsla_gpu_memory_info_t;

/**
 * @brief Utility functions for GPU type checking
 */

/**
 * @brief Check if GPU supports extended precision
 * 
 * @return true if GPU supports extended precision types
 */
static inline bool vsla_gpu_has_extended_precision(void) {
    return VSLA_GPU_HAS_EXTENDED_PRECISION;
}

/**
 * @brief Get the size of a GPU floating-point type
 * 
 * @param dtype Data type (VSLA_DTYPE_F32 or VSLA_DTYPE_F64)
 * @return Size in bytes
 */
static inline size_t vsla_gpu_dtype_size(int dtype) {
    switch (dtype) {
        case 1: return VSLA_GPU_F32_SIZE;  // VSLA_DTYPE_F32
        case 0: return VSLA_GPU_F64_SIZE;  // VSLA_DTYPE_F64
        default: return 0;
    }
}

/**
 * @brief Get the epsilon value for a GPU floating-point type
 * 
 * @param dtype Data type (VSLA_DTYPE_F32 or VSLA_DTYPE_F64)
 * @return Epsilon value
 */
static inline double vsla_gpu_dtype_epsilon(int dtype) {
    switch (dtype) {
        case 1: return VSLA_GPU_F32_EPSILON;  // VSLA_DTYPE_F32
        case 0: return VSLA_GPU_F64_EPSILON;  // VSLA_DTYPE_F64
        default: return 0.0;
    }
}

/**
 * @brief Convert between GPU and CPU floating-point types
 */
#ifdef __CUDACC__
__host__ __device__ static inline vsla_gpu_f32_t vsla_gpu_f32_from_double(double value) {
    return (vsla_gpu_f32_t)value;
}

__host__ __device__ static inline vsla_gpu_f64_t vsla_gpu_f64_from_double(double value) {
    return (vsla_gpu_f64_t)value;
}

__host__ __device__ static inline double vsla_gpu_f32_to_double(vsla_gpu_f32_t value) {
    return (double)value;
}

__host__ __device__ static inline double vsla_gpu_f64_to_double(vsla_gpu_f64_t value) {
    return (double)value;
}
#else
static inline vsla_gpu_f32_t vsla_gpu_f32_from_double(double value) {
    return (vsla_gpu_f32_t)value;
}

static inline vsla_gpu_f64_t vsla_gpu_f64_from_double(double value) {
    return (vsla_gpu_f64_t)value;
}

static inline double vsla_gpu_f32_to_double(vsla_gpu_f32_t value) {
    return (double)value;
}

static inline double vsla_gpu_f64_to_double(vsla_gpu_f64_t value) {
    return (double)value;
}
#endif

/**
 * @brief Optimal GPU launch configuration calculation
 * 
 * @param problem_size Total number of elements to process
 * @param config Output launch configuration
 * @return 0 on success, -1 on error
 */
int vsla_gpu_calculate_launch_config(size_t problem_size, 
                                     vsla_gpu_launch_config_t* config);

/**
 * @brief C23 migration utilities (for future use)
 */
#ifdef VSLA_ENABLE_C23_TYPES
    #define VSLA_GPU_C23_AVAILABLE 1
    #define VSLA_GPU_MIGRATION_COMPLETE 1
#else
    #define VSLA_GPU_C23_AVAILABLE 0
    #define VSLA_GPU_MIGRATION_COMPLETE 0
#endif

/**
 * @brief Version information for migration tracking
 */
#define VSLA_GPU_TYPES_VERSION_MAJOR 1
#define VSLA_GPU_TYPES_VERSION_MINOR 0
#define VSLA_GPU_TYPES_VERSION_PATCH 0

/**
 * @brief Migration status string
 */
static inline const char* vsla_gpu_migration_status(void) {
    if (VSLA_GPU_C23_AVAILABLE) {
        return "C23 types enabled";
    } else {
        return "Traditional types (C23 migration pending)";
    }
}

#ifdef __cplusplus
}
#endif

#endif // VSLA_GPU_TYPES_H