/**
 * @file vsla_tensor_internal.h
 * @brief Internal tensor structure definition and helper functions
 * 
 * This header defines the actual tensor structure and internal helper functions.
 * It should only be included by implementation files, not exposed to users.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_TENSOR_INTERNAL_H
#define VSLA_TENSOR_INTERNAL_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Internal tensor structure
 * 
 * This structure contains all the internal state of a VSLA tensor.
 * It supports both CPU and GPU data with validity tracking.
 */
struct vsla_tensor {
    /* Tensor metadata */
    uint8_t rank;                    /**< Number of dimensions */
    uint8_t model;                   /**< Model type (A or B) - stored as uint8_t */
    uint8_t dtype;                   /**< Data type - stored as uint8_t */
    uint8_t flags;                   /**< Various flags */
    
    /* Shape information */
    uint64_t* shape;                 /**< Actual shape per dimension */
    uint64_t* cap;                   /**< Allocated capacity per dimension (power of 2) */
    uint64_t* stride;                /**< Stride in bytes per dimension */
    
    /* Data storage - compatibility with existing code */
    void* data;                      /**< Primary data pointer (same as cpu_data) */
    void* cpu_data;                  /**< CPU memory pointer */
    void* gpu_data;                  /**< GPU memory pointer */
    bool cpu_valid;                  /**< True if CPU data is up to date */
    bool gpu_valid;                  /**< True if GPU data is up to date */
    vsla_backend_t location;         /**< Current primary location of data */
    
    /* Memory management */
    size_t data_size;                /**< Total allocated size in bytes */
    size_t alignment;                /**< Memory alignment requirement */
    
    /* Context reference - compatibility */
    void* ctx;                       /**< Context pointer (for compatibility) */
};

/* Helper function declarations */

/**
 * @brief Calculate total number of elements in tensor shape
 */
uint64_t vsla_numel(const vsla_tensor_t* tensor);

/**
 * @brief Calculate total capacity (based on cap array)
 */
uint64_t vsla_capacity(const vsla_tensor_t* tensor);

/**
 * @brief Get size in bytes of a data type
 */
size_t vsla_dtype_size(vsla_dtype_t dtype);

/**
 * @brief Find next power of 2 >= value
 */
uint64_t vsla_next_pow2(uint64_t value);

/**
 * @brief Calculate strides for a tensor
 */
vsla_error_t vsla_calculate_strides(vsla_tensor_t* tensor);

/**
 * @brief Validate tensor pointer and basic properties
 */
vsla_error_t vsla_validate_tensor(const vsla_tensor_t* tensor);

/**
 * @brief Check if two tensors are compatible for operations
 */
vsla_error_t vsla_check_compatibility(const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * @brief Calculate linear index from multi-dimensional indices
 */
size_t vsla_calc_linear_index(const vsla_tensor_t* tensor, const uint64_t indices[]);

/**
 * @brief Check if indices are within tensor bounds
 */
bool vsla_indices_valid(const vsla_tensor_t* tensor, const uint64_t indices[]);

/**
 * @brief Copy tensor metadata (shape, model, dtype) without data
 */
vsla_error_t vsla_copy_metadata(vsla_tensor_t* dst, const vsla_tensor_t* src);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_TENSOR_INTERNAL_H */