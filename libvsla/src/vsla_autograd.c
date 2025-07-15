/**
 * @file vsla_autograd.c
 * @brief Automatic differentiation support for VSLA
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "vsla/vsla_autograd.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_ops.h"
#include "vsla/vsla_conv.h"
#include "vsla/vsla_kron.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define INITIAL_TAPE_CAPACITY 64
#define INITIAL_GRADIENT_CAPACITY 64

vsla_tape_t* vsla_tape_new(void) {
    vsla_tape_t* tape = malloc(sizeof(vsla_tape_t));
    if (!tape) return NULL;
    
    tape->ops = malloc(INITIAL_TAPE_CAPACITY * sizeof(vsla_op_record_t));
    if (!tape->ops) {
        free(tape);
        return NULL;
    }
    
    tape->gradients = malloc(INITIAL_GRADIENT_CAPACITY * sizeof(vsla_tensor_t*));
    if (!tape->gradients) {
        free(tape->ops);
        free(tape);
        return NULL;
    }
    
    tape->num_ops = 0;
    tape->capacity = INITIAL_TAPE_CAPACITY;
    tape->num_gradients = 0;
    
    // Initialize gradients to NULL
    for (size_t i = 0; i < INITIAL_GRADIENT_CAPACITY; i++) {
        tape->gradients[i] = NULL;
    }
    
    return tape;
}

void vsla_tape_free(vsla_tape_t* tape) {
    if (!tape) return;
    
    // Free operation records and extra data
    for (size_t i = 0; i < tape->num_ops; i++) {
        free(tape->ops[i].inputs);
        free(tape->ops[i].extra_data);
    }
    free(tape->ops);
    
    // Free gradients (only gradient tensors, not tensor pointers)
    for (size_t i = 1; i < tape->num_gradients; i += 2) {
        if (tape->gradients[i]) {
            vsla_free(tape->gradients[i]);
        }
    }
    free(tape->gradients);
    
    free(tape);
}

static vsla_error_t resize_tape_if_needed(vsla_tape_t* tape) {
    if (tape->num_ops >= tape->capacity) {
        size_t new_capacity = tape->capacity * 2;
        vsla_op_record_t* new_ops = realloc(tape->ops, 
                                           new_capacity * sizeof(vsla_op_record_t));
        if (!new_ops) return VSLA_ERROR_MEMORY;
        
        tape->ops = new_ops;
        tape->capacity = new_capacity;
    }
    return VSLA_SUCCESS;
}

vsla_error_t vsla_tape_record(vsla_tape_t* tape, vsla_op_type_t op,
                              vsla_tensor_t** inputs, size_t num_inputs,
                              vsla_tensor_t* output, void* extra_data,
                              size_t extra_size) {
    if (!tape || !output) return VSLA_ERROR_NULL_POINTER;
    if (num_inputs > 0 && !inputs) return VSLA_ERROR_NULL_POINTER;
    
    vsla_error_t err = resize_tape_if_needed(tape);
    if (err != VSLA_SUCCESS) return err;
    
    vsla_op_record_t* record = &tape->ops[tape->num_ops];
    record->op = op;
    record->num_inputs = num_inputs;
    record->output = output;
    
    // Copy input pointers
    if (num_inputs > 0) {
        record->inputs = malloc(num_inputs * sizeof(vsla_tensor_t*));
        if (!record->inputs) return VSLA_ERROR_MEMORY;
        memcpy(record->inputs, inputs, num_inputs * sizeof(vsla_tensor_t*));
    } else {
        record->inputs = NULL;
    }
    
    // Copy extra data if provided
    if (extra_data && extra_size > 0) {
        record->extra_data = malloc(extra_size);
        if (!record->extra_data) {
            free(record->inputs);
            return VSLA_ERROR_MEMORY;
        }
        memcpy(record->extra_data, extra_data, extra_size);
        record->extra_size = extra_size;
    } else {
        record->extra_data = NULL;
        record->extra_size = 0;
    }
    
    tape->num_ops++;
    return VSLA_SUCCESS;
}

// Helper function to find tensor index in gradient array
static int find_tensor_index(const vsla_tape_t* tape, const vsla_tensor_t* tensor) {
    // Use tensor pointer as unique identifier
    for (size_t i = 0; i < tape->num_gradients; i += 2) {
        if (i + 1 < tape->num_gradients && 
            tape->gradients[i] == (vsla_tensor_t*)tensor) {
            return (int)(i + 1);  // Return gradient index
        }
    }
    return -1;
}

vsla_tensor_t* vsla_get_gradient(const vsla_tape_t* tape, const vsla_tensor_t* tensor) {
    if (!tape || !tensor) return NULL;
    
    int grad_idx = find_tensor_index(tape, tensor);
    if (grad_idx >= 0 && grad_idx < (int)tape->num_gradients) {
        return tape->gradients[grad_idx];
    }
    return NULL;
}

vsla_error_t vsla_set_gradient(vsla_tape_t* tape, const vsla_tensor_t* tensor,
                               const vsla_tensor_t* gradient) {
    if (!tape || !tensor || !gradient) return VSLA_ERROR_NULL_POINTER;
    
    // Find existing gradient or add new one
    int grad_idx = find_tensor_index(tape, tensor);
    
    if (grad_idx >= 0) {
        // Update existing gradient
        vsla_free(tape->gradients[grad_idx]);
        tape->gradients[grad_idx] = vsla_copy(gradient);
        if (!tape->gradients[grad_idx]) return VSLA_ERROR_MEMORY;
    } else {
        // Add new tensor-gradient pair
        if (tape->num_gradients + 2 > INITIAL_GRADIENT_CAPACITY) {
            // For simplicity, use fixed capacity. In production, would resize dynamically
            return VSLA_ERROR_MEMORY;
        }
        
        // Store tensor pointer and gradient
        tape->gradients[tape->num_gradients] = (vsla_tensor_t*)tensor;
        tape->gradients[tape->num_gradients + 1] = vsla_copy(gradient);
        if (!tape->gradients[tape->num_gradients + 1]) return VSLA_ERROR_MEMORY;
        
        tape->num_gradients += 2;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_clear_gradients(vsla_tape_t* tape) {
    if (!tape) return VSLA_ERROR_NULL_POINTER;
    
    // Free all gradient tensors
    for (size_t i = 1; i < tape->num_gradients; i += 2) {
        vsla_free(tape->gradients[i]);
        tape->gradients[i] = NULL;
    }
    
    // Clear tensor pointers
    for (size_t i = 0; i < tape->num_gradients; i += 2) {
        tape->gradients[i] = NULL;
    }
    
    tape->num_gradients = 0;
    return VSLA_SUCCESS;
}

// Backward pass implementations for each operation type
vsla_error_t vsla_add_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out) {
    if (!grad_a || !grad_b || !grad_out) return VSLA_ERROR_NULL_POINTER;
    
    // For addition: grad_a = grad_out, grad_b = grad_out
    // But need to handle broadcasting/padding correctly
    vsla_error_t err = vsla_add(grad_a, grad_a, grad_out);
    if (err != VSLA_SUCCESS) return err;
    
    err = vsla_add(grad_b, grad_b, grad_out);
    return err;
}

vsla_error_t vsla_sub_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out) {
    if (!grad_a || !grad_b || !grad_out) return VSLA_ERROR_NULL_POINTER;
    
    // For subtraction: grad_a = grad_out, grad_b = -grad_out
    vsla_error_t err = vsla_add(grad_a, grad_a, grad_out);
    if (err != VSLA_SUCCESS) return err;
    
    err = vsla_sub(grad_b, grad_b, grad_out);
    return err;
}

vsla_error_t vsla_scale_backward(vsla_tensor_t* grad_in, double* grad_scalar,
                                 const vsla_tensor_t* grad_out,
                                 const vsla_tensor_t* input, double scalar) {
    if (!grad_in || !grad_out || !input) return VSLA_ERROR_NULL_POINTER;
    
    // For scaling: grad_input = scalar * grad_out
    vsla_tensor_t* scaled_grad = vsla_copy(grad_out);
    if (!scaled_grad) return VSLA_ERROR_MEMORY;
    
    vsla_error_t err = vsla_scale(scaled_grad, grad_out, scalar);
    if (err != VSLA_SUCCESS) {
        vsla_free(scaled_grad);
        return err;
    }
    
    err = vsla_add(grad_in, grad_in, scaled_grad);
    vsla_free(scaled_grad);
    
    if (grad_scalar) {
        // grad_scalar = sum(input * grad_out)
        // This is a simplified implementation
        // In practice, would need element-wise multiplication and summation
        vsla_tensor_t* hadamard = vsla_copy(input);
        if (hadamard) {
            // Would implement hadamard product here
            // For now, just approximate
            double input_sum, grad_sum;
            vsla_sum(input, &input_sum);
            vsla_sum(grad_out, &grad_sum);
            *grad_scalar += input_sum * grad_sum / vsla_numel(input);
            vsla_free(hadamard);
        }
    }
    
    return err;
}

vsla_error_t vsla_hadamard_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                                    const vsla_tensor_t* grad_out,
                                    const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!grad_a || !grad_b || !grad_out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For Hadamard product: grad_a = b * grad_out, grad_b = a * grad_out
    // This is a placeholder implementation
    // Would need actual element-wise multiplication
    
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t backward_operation(vsla_tape_t* tape, const vsla_op_record_t* record) {
    vsla_tensor_t* grad_out = vsla_get_gradient(tape, record->output);
    if (!grad_out) {
        // No gradient for this output, skip
        return VSLA_SUCCESS;
    }
    
    switch (record->op) {
        case VSLA_OP_ADD:
            if (record->num_inputs == 2) {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                if (!grad_a) {
                    grad_a = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], grad_a);
                    vsla_free(grad_a);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                }
                
                if (!grad_b) {
                    grad_b = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], grad_b);
                    vsla_free(grad_b);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                }
                
                return vsla_add_backward(grad_a, grad_b, grad_out);
            }
            break;
            
        case VSLA_OP_SUB:
            if (record->num_inputs == 2) {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                if (!grad_a) {
                    grad_a = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], grad_a);
                    vsla_free(grad_a);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                }
                
                if (!grad_b) {
                    grad_b = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], grad_b);
                    vsla_free(grad_b);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                }
                
                return vsla_sub_backward(grad_a, grad_b, grad_out);
            }
            break;
            
        case VSLA_OP_SCALE:
            if (record->num_inputs == 1 && record->extra_data) {
                double* scalar = (double*)record->extra_data;
                vsla_tensor_t* grad_in = vsla_get_gradient(tape, record->inputs[0]);
                
                if (!grad_in) {
                    grad_in = vsla_zeros(record->inputs[0]->rank, 
                                        record->inputs[0]->shape,
                                        record->inputs[0]->model,
                                        record->inputs[0]->dtype);
                    if (!grad_in) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], grad_in);
                    vsla_free(grad_in);
                    grad_in = vsla_get_gradient(tape, record->inputs[0]);
                }
                
                return vsla_scale_backward(grad_in, NULL, grad_out, 
                                         record->inputs[0], *scalar);
            }
            break;
            
        case VSLA_OP_CONV:
        case VSLA_OP_KRON:
        case VSLA_OP_HADAMARD:
        case VSLA_OP_MATMUL:
        case VSLA_OP_TRANSPOSE:
        case VSLA_OP_RESHAPE:
        case VSLA_OP_PAD_RANK:
            // These operations need specialized backward implementations
            return VSLA_ERROR_NOT_IMPLEMENTED;
            
        default:
            return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_backward(vsla_tape_t* tape) {
    if (!tape) return VSLA_ERROR_NULL_POINTER;
    
    // Process operations in reverse order
    for (int i = (int)tape->num_ops - 1; i >= 0; i--) {
        vsla_error_t err = backward_operation(tape, &tape->ops[i]);
        if (err != VSLA_SUCCESS && err != VSLA_ERROR_NOT_IMPLEMENTED) {
            return err;
        }
    }
    
    return VSLA_SUCCESS;
}