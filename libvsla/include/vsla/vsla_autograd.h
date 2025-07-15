/**
 * @file vsla_autograd.h
 * @brief Automatic differentiation support for VSLA
 * 
 * @copyright MIT License
 */

#ifndef VSLA_AUTOGRAD_H
#define VSLA_AUTOGRAD_H

#include "vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Operation types for autograd tape
 */
typedef enum {
    VSLA_OP_ADD,        /**< Element-wise addition */
    VSLA_OP_SUB,        /**< Element-wise subtraction */
    VSLA_OP_SCALE,      /**< Scalar multiplication */
    VSLA_OP_HADAMARD,   /**< Element-wise multiplication */
    VSLA_OP_CONV,       /**< Convolution (Model A) */
    VSLA_OP_KRON,       /**< Kronecker product (Model B) */
    VSLA_OP_MATMUL,     /**< Matrix multiplication */
    VSLA_OP_TRANSPOSE,  /**< Matrix transpose */
    VSLA_OP_RESHAPE,    /**< Reshape operation */
    VSLA_OP_PAD_RANK    /**< Rank padding */
} vsla_op_type_t;

/**
 * @brief Operation record for autograd tape
 */
typedef struct {
    vsla_op_type_t op;          /**< Operation type */
    vsla_tensor_t** inputs;     /**< Array of input tensors */
    size_t num_inputs;          /**< Number of inputs */
    vsla_tensor_t* output;      /**< Output tensor */
    void* extra_data;           /**< Operation-specific data */
    size_t extra_size;          /**< Size of extra data */
} vsla_op_record_t;

/**
 * @brief Gradient tape for automatic differentiation
 */
typedef struct {
    vsla_op_record_t* ops;      /**< Array of operation records */
    size_t num_ops;             /**< Number of operations */
    size_t capacity;            /**< Allocated capacity */
    vsla_tensor_t** gradients;  /**< Gradient storage */
    size_t num_gradients;       /**< Number of gradients */
} vsla_tape_t;

/**
 * @brief Create a new gradient tape
 * 
 * @return New tape or NULL on error
 */
vsla_tape_t* vsla_tape_new(void);

/**
 * @brief Free a gradient tape and all associated memory
 * 
 * @param tape Tape to free
 */
void vsla_tape_free(vsla_tape_t* tape);

/**
 * @brief Record an operation on the tape
 * 
 * @param tape Gradient tape
 * @param op Operation type
 * @param inputs Array of input tensors
 * @param num_inputs Number of inputs
 * @param output Output tensor
 * @param extra_data Operation-specific data (can be NULL)
 * @param extra_size Size of extra data
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_tape_record(vsla_tape_t* tape, vsla_op_type_t op,
                              vsla_tensor_t** inputs, size_t num_inputs,
                              vsla_tensor_t* output, void* extra_data,
                              size_t extra_size);

/**
 * @brief Perform backward pass on a tape
 * 
 * Computes gradients for all operations on the tape in reverse order.
 * The gradient of the final output should be set before calling this.
 * 
 * @param tape Gradient tape with recorded operations
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_backward(vsla_tape_t* tape);

/**
 * @brief Get gradient for a tensor
 * 
 * @param tape Gradient tape
 * @param tensor Tensor to get gradient for
 * @return Gradient tensor or NULL if not found
 */
vsla_tensor_t* vsla_get_gradient(const vsla_tape_t* tape, 
                                 const vsla_tensor_t* tensor);

/**
 * @brief Set gradient for a tensor
 * 
 * @param tape Gradient tape
 * @param tensor Tensor to set gradient for
 * @param gradient Gradient value
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_set_gradient(vsla_tape_t* tape, const vsla_tensor_t* tensor,
                               const vsla_tensor_t* gradient);

/**
 * @brief Clear all gradients on the tape
 * 
 * @param tape Gradient tape
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_clear_gradients(vsla_tape_t* tape);

/**
 * @brief Backward function for addition
 */
vsla_error_t vsla_add_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out);

/**
 * @brief Backward function for subtraction
 */
vsla_error_t vsla_sub_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out);

/**
 * @brief Backward function for scaling
 */
vsla_error_t vsla_scale_backward(vsla_tensor_t* grad_in, double* grad_scalar,
                                 const vsla_tensor_t* grad_out,
                                 const vsla_tensor_t* input, double scalar);

/**
 * @brief Backward function for Hadamard product
 */
vsla_error_t vsla_hadamard_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                                    const vsla_tensor_t* grad_out,
                                    const vsla_tensor_t* a, const vsla_tensor_t* b);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_AUTOGRAD_H */