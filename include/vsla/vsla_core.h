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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error codes returned by VSLA functions
 */
typedef enum {
    VSLA_SUCCESS = 0,              /**< Operation completed successfully */
    VSLA_ERROR_NULL_POINTER,       /**< Null pointer passed where not allowed */
    VSLA_ERROR_INVALID_ARGUMENT,   /**< Invalid argument provided */
    VSLA_ERROR_MEMORY,             /**< Memory allocation failed */
    VSLA_ERROR_DIMENSION_MISMATCH, /**< Dimension mismatch in operation */
    VSLA_ERROR_INVALID_MODEL,      /**< Invalid model specified */
    VSLA_ERROR_INVALID_DTYPE,      /**< Invalid data type specified */
    VSLA_ERROR_IO,                 /**< I/O operation failed */
    VSLA_ERROR_NOT_IMPLEMENTED,    /**< Feature not yet implemented */
    VSLA_ERROR_INVALID_RANK,       /**< Invalid rank (must be 0-255) */
    VSLA_ERROR_OVERFLOW,           /**< Numeric overflow detected */
    VSLA_ERROR_FFT,                /**< FFT operation failed */
    VSLA_ERROR_INVALID_FILE,       /**< Invalid file format */
    VSLA_ERROR_INCOMPATIBLE_MODELS /**< Incompatible models in operation */
} vsla_error_t;

/**
 * @brief Model types for VSLA operations
 */
typedef enum {
    VSLA_MODEL_A = 0,  /**< Model A: Convolution-based (commutative) */
    VSLA_MODEL_B = 1   /**< Model B: Kronecker product-based (non-commutative) */
} vsla_model_t;

/**
 * @brief Data types supported by tensors
 */
typedef enum {
    VSLA_DTYPE_F64 = 0,  /**< 64-bit floating point (double) */
    VSLA_DTYPE_F32 = 1   /**< 32-bit floating point (float) */
} vsla_dtype_t;

/**
 * @brief Get human-readable error message
 * 
 * @param error Error code
 * @return String describing the error
 */
const char* vsla_error_string(vsla_error_t error);

/**
 * @brief Get the size in bytes of a data type
 * 
 * @param dtype Data type
 * @return Size in bytes, or 0 if invalid dtype
 */
size_t vsla_dtype_size(vsla_dtype_t dtype);

/**
 * @brief Compute the next power of 2 greater than or equal to n
 * 
 * @param n Input value
 * @return Next power of 2 >= n
 */
uint64_t vsla_next_pow2(uint64_t n);

/**
 * @brief Check if a number is a power of 2
 * 
 * @param n Number to check
 * @return 1 if n is a power of 2, 0 otherwise
 */
int vsla_is_pow2(uint64_t n);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_CORE_H */