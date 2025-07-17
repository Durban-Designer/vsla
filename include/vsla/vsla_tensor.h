/**
 * @file vsla_tensor.h
 * @brief Core tensor data structure and basic operations
 * 
 * @copyright MIT License
 */

#ifndef VSLA_TENSOR_H
#define VSLA_TENSOR_H

#include "vsla_core.h"
#include <stdbool.h>

/* Forward declarations */
typedef struct vsla_context vsla_context_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Core tensor structure for VSLA
 * 
 * This structure represents a multi-dimensional tensor with variable shape.
 * The tensor supports automatic zero-padding to make operations between
 * tensors of different shapes well-defined.
 */
typedef struct vsla_tensor {
    /* Core tensor info */
    uint8_t    rank;      /**< Number of axes (dimensions), 0-255 */
    uint8_t    model;     /**< Model: 0 = convolution, 1 = Kronecker */
    uint8_t    dtype;     /**< Data type: 0 = f64, 1 = f32 */
    uint8_t    flags;     /**< Reserved for future use */

    uint64_t  *shape;     /**< Logical extent per axis (length = rank) */
    uint64_t  *cap;       /**< Padded/allocated extent per axis */
    uint64_t  *stride;    /**< Byte strides for row-major traversal */
    
    /* Memory management - backend specific */
    void      *data;      /**< CPU data buffer (for compatibility) */
    void      *cpu_data;  /**< CPU memory pointer */
    void      *gpu_data;  /**< GPU memory pointer (if available) */
    size_t     data_size; /**< Total data size in bytes */
    
    /* Data location and validity */
    vsla_backend_t location;  /**< Current data location (CPU/GPU) */
    bool       cpu_valid;     /**< CPU data is up-to-date */
    bool       gpu_valid;     /**< GPU data is up-to-date */
    
    /* Context reference */
    vsla_context_t *ctx; /**< Reference to owning context */
} vsla_tensor_t;

/**
 * @brief Create a new tensor
 * 
 * Allocates a new tensor with the specified rank, shape, model, and data type.
 * The capacity (cap) for each dimension is set to the next power of 2 >= shape[i].
 * 
 * @param rank Number of dimensions (0-255)
 * @param shape Array of dimension sizes (length = rank)
 * @param model Model type (VSLA_MODEL_A or VSLA_MODEL_B)
 * @param dtype Data type (VSLA_DTYPE_F64 or VSLA_DTYPE_F32)
 * @return Pointer to new tensor, or NULL on error
 */
vsla_tensor_t* vsla_new(uint8_t rank, const uint64_t shape[], 
                        vsla_model_t model, vsla_dtype_t dtype);

/**
 * @brief Free a tensor and all its allocated memory
 * 
 * @param tensor Tensor to free (can be NULL)
 */
void vsla_free(vsla_tensor_t* tensor);

/**
 * @brief Create a copy of a tensor
 * 
 * @param tensor Tensor to copy
 * @return New tensor with copied data, or NULL on error
 */
vsla_tensor_t* vsla_copy_basic(const vsla_tensor_t* tensor);

/**
 * @brief Create a tensor filled with zeros
 * 
 * @param rank Number of dimensions
 * @param shape Array of dimension sizes
 * @param model Model type
 * @param dtype Data type
 * @return New zero tensor, or NULL on error
 */
vsla_tensor_t* vsla_zeros(uint8_t rank, const uint64_t shape[],
                          vsla_model_t model, vsla_dtype_t dtype);

/**
 * @brief Create a tensor filled with ones
 * 
 * @param rank Number of dimensions
 * @param shape Array of dimension sizes
 * @param model Model type
 * @param dtype Data type
 * @return New tensor filled with ones, or NULL on error
 */
vsla_tensor_t* vsla_ones(uint8_t rank, const uint64_t shape[],
                         vsla_model_t model, vsla_dtype_t dtype);

/**
 * @brief Get the total number of elements in the tensor (based on shape)
 * 
 * @param tensor Input tensor
 * @return Number of elements, or 0 if tensor is NULL
 */
uint64_t vsla_numel(const vsla_tensor_t* tensor);

/**
 * @brief Get the total allocated capacity (based on cap)
 * 
 * @param tensor Input tensor
 * @return Total capacity, or 0 if tensor is NULL
 */
uint64_t vsla_capacity(const vsla_tensor_t* tensor);

/**
 * @brief Get a pointer to an element in the tensor
 * 
 * @param tensor Input tensor
 * @param indices Array of indices (length = rank)
 * @return Pointer to element, or NULL if out of bounds
 */
void* vsla_get_ptr(const vsla_tensor_t* tensor, const uint64_t indices[]);

/**
 * @brief Get a double value from the tensor (with type conversion if needed)
 * 
 * @param tensor Input tensor
 * @param indices Array of indices
 * @param value Output value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_get_f64(const vsla_tensor_t* tensor, const uint64_t indices[], 
                          double* value);

/**
 * @brief Set a double value in the tensor (with type conversion if needed)
 * 
 * @param tensor Input tensor
 * @param indices Array of indices
 * @param value Value to set
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_set_f64(vsla_tensor_t* tensor, const uint64_t indices[], 
                          double value);

/**
 * @brief Fill tensor with a constant value
 * 
 * @param tensor Tensor to fill
 * @param value Value to fill with
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_fill_basic(vsla_tensor_t* tensor, double value);

/**
 * @brief Print tensor information to stdout
 * 
 * @param tensor Tensor to print
 * @param name Optional name for the tensor
 */
void vsla_print(const vsla_tensor_t* tensor, const char* name);

/**
 * @brief Check if two tensors have the same shape
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return 1 if shapes match, 0 otherwise
 */
int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * @brief Create the zero element for the semiring
 * 
 * @param model Model type
 * @param dtype Data type
 * @return Zero tensor (empty tensor), or NULL on error
 */
vsla_tensor_t* vsla_zero_element(vsla_model_t model, vsla_dtype_t dtype);

/**
 * @brief Create the one element for the semiring
 * 
 * @param model Model type
 * @param dtype Data type
 * @return One tensor (1D tensor with single element 1), or NULL on error
 */
vsla_tensor_t* vsla_one_element(vsla_model_t model, vsla_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_TENSOR_H */