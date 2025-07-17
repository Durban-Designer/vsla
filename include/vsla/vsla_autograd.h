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

typedef struct vsla_op_record_s {
    vsla_op_type_t op;
    vsla_tensor_t** inputs;
    size_t num_inputs;
    vsla_tensor_t* output;
    void* extra_data;
    size_t extra_size;
} vsla_op_record_t;

typedef struct vsla_tape_s {
    vsla_op_record_t* ops;
    size_t num_ops;
    size_t capacity;
    vsla_tensor_t** gradients;
    size_t num_gradients;
    size_t grad_capacity;
} vsla_tape_t;

vsla_tape_t* vsla_tape_new(void);
void vsla_tape_free(vsla_tape_t* tape);
vsla_error_t vsla_tape_record(vsla_tape_t* tape, vsla_op_type_t op, vsla_tensor_t** inputs, size_t num_inputs, vsla_tensor_t* output, void* extra_data, size_t extra_size);
vsla_error_t vsla_backward(vsla_tape_t* tape);
vsla_tensor_t* vsla_get_gradient(const vsla_tape_t* tape, const vsla_tensor_t* tensor);
vsla_error_t vsla_set_gradient(vsla_tape_t* tape, const vsla_tensor_t* tensor, const vsla_tensor_t* gradient);
vsla_error_t vsla_clear_gradients(vsla_tape_t* tape);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_AUTOGRAD_H */
