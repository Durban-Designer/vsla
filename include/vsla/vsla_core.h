/**
 * @file vsla_core.h
 * @brief Core definitions and error codes for VSLA library
 * 
 * @copyright MIT License
 */

#ifndef VSLA_CORE_H
#define VSLA_CORE_H

#include <stdint.h>
#include <stddef.h>

#define VSLA_VERSION_MAJOR 1
#define VSLA_VERSION_MINOR 0
#define VSLA_VERSION_PATCH 0
#define VSLA_VERSION_STRING "1.0.0"

// Maximum number of tensor dimensions (as per v3.1 spec)
#define VSLA_MAX_RANK 16

/**
 * @brief Error codes returned by VSLA functions
 */
typedef enum {
    VSLA_SUCCESS = 0,
    VSLA_ERROR_NULL_POINTER,
    VSLA_ERROR_INVALID_ARGUMENT,
    VSLA_ERROR_MEMORY,
    VSLA_ERROR_DIMENSION_MISMATCH,
    VSLA_ERROR_INVALID_MODEL,
    VSLA_ERROR_INVALID_DTYPE,
    VSLA_ERROR_IO,
    VSLA_ERROR_NOT_IMPLEMENTED,
    VSLA_ERROR_INVALID_RANK,
    VSLA_ERROR_OVERFLOW,
    VSLA_ERROR_FFT,
    VSLA_ERROR_INVALID_FILE,
    VSLA_ERROR_INCOMPATIBLE_MODELS,
    VSLA_ERROR_INCOMPATIBLE_SHAPES,
    VSLA_ERROR_GPU_FAILURE,
    VSLA_ERROR_INVALID_STATE
} vsla_error_t;

/**
 * @brief Model types for VSLA operations
 */
typedef enum {
    VSLA_MODEL_A = 0,
    VSLA_MODEL_B = 1
} vsla_model_t;

/**
 * @brief Data types supported by tensors
 */
typedef enum {
    VSLA_DTYPE_F64 = 0,
    VSLA_DTYPE_F32 = 1
} vsla_dtype_t;

/**
 * @brief Hardware backend type
 */
typedef enum {
    VSLA_BACKEND_CPU = 0,
    VSLA_BACKEND_CUDA = 1,
    VSLA_BACKEND_ROCM = 2,
    VSLA_BACKEND_ONEAPI = 3,
    VSLA_BACKEND_AUTO = 4
} vsla_backend_t;

const char* vsla_error_string(vsla_error_t error);
size_t vsla_dtype_size(vsla_dtype_t dtype);
uint64_t vsla_next_pow2(uint64_t n);
int vsla_is_pow2(uint64_t n);

#endif /* VSLA_CORE_H */