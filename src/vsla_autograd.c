/**
 * @file vsla_autograd.c
 * @brief Automatic differentiation support for VSLA
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "vsla/vsla_autograd.h"
#include "vsla/vsla_tensor_internal.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_backend_cpu.h"
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
    tape->grad_capacity = INITIAL_GRADIENT_CAPACITY;
    
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
    
    int grad_idx = find_tensor_index(tape, tensor);
    vsla_tensor_t* new_grad_copy = vsla_copy_basic(gradient); // Create copy once
    if (!new_grad_copy) {
        return VSLA_ERROR_MEMORY;
    }
    
    if (grad_idx >= 0) {
        // Update existing gradient
        vsla_free(tape->gradients[grad_idx]); // Free old gradient
        tape->gradients[grad_idx] = new_grad_copy; // Assign new copy
    } else {
        // Add new tensor-gradient pair
        if (tape->num_gradients + 2 > tape->grad_capacity) {
            size_t new_capacity = tape->grad_capacity * 2;
            vsla_tensor_t** new_gradients = realloc(tape->gradients,
                                                   new_capacity * sizeof(vsla_tensor_t*));
            if (!new_gradients) {
                vsla_free(new_grad_copy); // Clean up new_grad_copy if realloc fails
                return VSLA_ERROR_MEMORY;
            }
            
            // Initialize new slots to NULL
            for (size_t i = tape->grad_capacity; i < new_capacity; i++) {
                new_gradients[i] = NULL;
            }
            
            tape->gradients = new_gradients;
            tape->grad_capacity = new_capacity;
        }
        
        // Store tensor pointer and gradient
        tape->gradients[tape->num_gradients] = (vsla_tensor_t*)tensor;
        tape->gradients[tape->num_gradients + 1] = new_grad_copy;
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
    vsla_error_t err = vsla_cpu_add(grad_a, grad_a, grad_out);
    if (err != VSLA_SUCCESS) return err;
    
    err = vsla_cpu_add(grad_b, grad_b, grad_out);
    return err;
}

vsla_error_t vsla_sub_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out) {
    if (!grad_a || !grad_b || !grad_out) return VSLA_ERROR_NULL_POINTER;
    
    // For subtraction: grad_a = grad_out, grad_b = -grad_out
    vsla_error_t err = vsla_cpu_add(grad_a, grad_a, grad_out);
    if (err != VSLA_SUCCESS) return err;
    
    // For grad_b = grad_b - grad_out, we need to negate grad_out
    vsla_tensor_t* neg_grad_out = vsla_copy_basic(grad_out);
    if (!neg_grad_out) return VSLA_ERROR_MEMORY;
    
    err = vsla_cpu_scale(neg_grad_out, neg_grad_out, -1.0);
    if (err != VSLA_SUCCESS) {
        vsla_free(neg_grad_out);
        return err;
    }
    
    err = vsla_cpu_add(grad_b, grad_b, neg_grad_out);
    vsla_free(neg_grad_out);
    
    return err;
}

vsla_error_t vsla_scale_backward(vsla_tensor_t* grad_in, double* grad_scalar,
                                 const vsla_tensor_t* grad_out,
                                 const vsla_tensor_t* input, double scalar) {
    if (!grad_in || !grad_out || !input) return VSLA_ERROR_NULL_POINTER;
    
    // For scaling: grad_input = scalar * grad_out
    vsla_tensor_t* scaled_grad = vsla_copy_basic(grad_out);
    if (!scaled_grad) return VSLA_ERROR_MEMORY;
    
    vsla_error_t err = vsla_cpu_scale(scaled_grad, scaled_grad, scalar);
    if (err != VSLA_SUCCESS) {
        vsla_free(scaled_grad);
        return err;
    }
    
    err = vsla_cpu_add(grad_in, grad_in, scaled_grad);
    vsla_free(scaled_grad);
    
    if (grad_scalar) {
        // grad_scalar = sum(input * grad_out)
        // This is a simplified implementation
        // In practice, would need element-wise multiplication and summation
        vsla_tensor_t* hadamard = vsla_copy_basic(input);
        if (hadamard) {
            // Would implement hadamard product here
            // For now, just approximate
            double input_sum, grad_sum;
            vsla_cpu_sum(input, &input_sum);
            vsla_cpu_sum(grad_out, &grad_sum);
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
    
    // For Hadamard product C = A ⊙ B, the gradients are:
    // grad_a = B ⊙ grad_out (element-wise multiplication)
    // grad_b = A ⊙ grad_out (element-wise multiplication)
    
    vsla_error_t err;
    
    // Compute grad_a = b * grad_out
    err = vsla_cpu_hadamard(grad_a, b, grad_out);
    if (err != VSLA_SUCCESS) {
        return err;
    }
    
    // Compute grad_b = a * grad_out  
    err = vsla_cpu_hadamard(grad_b, a, grad_out);
    if (err != VSLA_SUCCESS) {
        return err;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_matmul_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                                  const vsla_tensor_t* grad_out,
                                  const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!grad_a || !grad_b || !grad_out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For matrix multiplication C = A × B, the gradients are:
    // grad_a = grad_out × B^T (matrix multiplication with transpose)
    // grad_b = A^T × grad_out (matrix multiplication with transpose)
    
    vsla_error_t err;
    vsla_tensor_t* b_transposed = NULL;
    vsla_tensor_t* a_transposed = NULL;
    
    // Create transposed versions
    b_transposed = vsla_copy_basic(b);
    if (!b_transposed) return VSLA_ERROR_MEMORY;
    
    err = vsla_cpu_transpose(b_transposed, b_transposed);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_transposed);
        return err;
    }
    
    a_transposed = vsla_copy_basic(a);
    if (!a_transposed) {
        vsla_free(b_transposed);
        return VSLA_ERROR_MEMORY;
    }
    
    err = vsla_cpu_transpose(a_transposed, a_transposed);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_transposed);
        vsla_free(a_transposed);
        return err;
    }
    
    // Compute grad_a = grad_out × B^T
    err = vsla_cpu_matmul(grad_a, grad_out, b_transposed);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_transposed);
        vsla_free(a_transposed);
        return err;
    }
    
    // Compute grad_b = A^T × grad_out
    err = vsla_cpu_matmul(grad_b, a_transposed, grad_out);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_transposed);
        vsla_free(a_transposed);
        return err;
    }
    
    vsla_free(b_transposed);
    vsla_free(a_transposed);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_transpose_backward(vsla_tensor_t* grad_input,
                                     const vsla_tensor_t* grad_out,
                                     const vsla_tensor_t* input) {
    if (!grad_input || !grad_out || !input) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For transpose B = A^T, the gradient is:
    // grad_a = (grad_out)^T (transpose the output gradient)
    
    return vsla_cpu_transpose(grad_input, grad_out);
}

vsla_error_t vsla_reshape_backward(vsla_tensor_t* grad_input,
                                   const vsla_tensor_t* grad_out,
                                   const vsla_tensor_t* input) {
    if (!grad_input || !grad_out || !input) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For reshape operation, the gradient is:
    // grad_a = reshape(grad_out, original_shape) (reshape gradient back to original shape)
    
    // Create a copy of grad_out and reshape it
    vsla_tensor_t* temp_grad = vsla_copy_basic(grad_out);
    if (!temp_grad) return VSLA_ERROR_MEMORY;
    
    vsla_error_t err = vsla_cpu_reshape(temp_grad, input->rank, input->shape);
    if (err != VSLA_SUCCESS) {
        vsla_free(temp_grad);
        return err;
    }
    
    // Copy the reshaped data to grad_input
    size_t element_size = vsla_dtype_size(input->dtype);
    size_t total_elements = vsla_numel(input);
    memcpy(grad_input->data, temp_grad->data, total_elements * element_size);
    
    vsla_free(temp_grad);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_pad_rank_backward(vsla_tensor_t* grad_input,
                                    const vsla_tensor_t* grad_out,
                                    const vsla_tensor_t* input) {
    if (!grad_input || !grad_out || !input) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For rank padding operation, the gradient is:
    // grad_a = unpad_rank(grad_out) (remove the padding to get gradient of original tensor)
    // This means copying only the first 'input->rank' dimensions
    
    // For pad_rank_backward, grad_input should already be properly allocated
    // We just need to copy the relevant data from grad_out
    
    // Copy data from grad_out to grad_input (effectively "unpadding")
    // Since grad_out has higher rank, we copy the relevant portion
    size_t input_size = vsla_numel(input);
    size_t element_size = vsla_dtype_size(input->dtype);
    
    memcpy(grad_input->data, grad_out->data, input_size * element_size);
    
    return VSLA_SUCCESS;
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
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_add_backward(grad_a, grad_b, grad_out);
            }
            break;
            
        case VSLA_OP_SUB:
            if (record->num_inputs == 2) {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                if (!grad_a) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
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
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                        record->inputs[0]->shape,
                                        record->inputs[0]->model,
                                        record->inputs[0]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_in = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_in) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_scale_backward(grad_in, NULL, grad_out, 
                                         record->inputs[0], *scalar);
            }
            break;
            
        case VSLA_OP_CONV:
            if (record->num_inputs == 2) {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                if (!grad_a) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_conv_backward(grad_a, grad_b, grad_out,
                                        record->inputs[0], record->inputs[1]);
            }
            break;
            
        case VSLA_OP_KRON:
            if (record->num_inputs == 2) {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                if (!grad_a) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                       record->inputs[0]->shape,
                                       record->inputs[0]->model,
                                       record->inputs[0]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                       record->inputs[1]->shape,
                                       record->inputs[1]->model,
                                       record->inputs[1]->dtype);
                    if (!zero_grad) return VSLA_ERROR_MEMORY;
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_kron_backward(grad_a, grad_b, grad_out,
                                        record->inputs[0], record->inputs[1]);
            }
            break;
            
        case VSLA_OP_HADAMARD:
            {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                // Create zero gradients if they don't exist
                if (!grad_a) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                    record->inputs[0]->shape, 
                                    record->inputs[0]->model,
                                    record->inputs[0]->dtype);
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                    record->inputs[1]->shape, 
                                    record->inputs[1]->model,
                                    record->inputs[1]->dtype);
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_hadamard_backward(grad_a, grad_b, grad_out,
                                            record->inputs[0], record->inputs[1]);
            }
            
        case VSLA_OP_MATMUL:
            {
                vsla_tensor_t* grad_a = vsla_get_gradient(tape, record->inputs[0]);
                vsla_tensor_t* grad_b = vsla_get_gradient(tape, record->inputs[1]);
                
                // Create zero gradients if they don't exist
                if (!grad_a) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                    record->inputs[0]->shape, 
                                    record->inputs[0]->model,
                                    record->inputs[0]->dtype);
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_a = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_a) return VSLA_ERROR_MEMORY;
                }
                
                if (!grad_b) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[1]->rank, 
                                    record->inputs[1]->shape, 
                                    record->inputs[1]->model,
                                    record->inputs[1]->dtype);
                    vsla_set_gradient(tape, record->inputs[1], zero_grad);
                    vsla_free(zero_grad);
                    grad_b = vsla_get_gradient(tape, record->inputs[1]);
                    if (!grad_b) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_matmul_backward(grad_a, grad_b, grad_out,
                                          record->inputs[0], record->inputs[1]);
            }
            
        case VSLA_OP_TRANSPOSE:
            {
                vsla_tensor_t* grad_input = vsla_get_gradient(tape, record->inputs[0]);
                
                // Create zero gradient if it doesn't exist
                if (!grad_input) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                    record->inputs[0]->shape, 
                                    record->inputs[0]->model,
                                    record->inputs[0]->dtype);
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_input = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_input) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_transpose_backward(grad_input, grad_out, record->inputs[0]);
            }
            
        case VSLA_OP_RESHAPE:
            {
                vsla_tensor_t* grad_input = vsla_get_gradient(tape, record->inputs[0]);
                
                // Create zero gradient if it doesn't exist
                if (!grad_input) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                    record->inputs[0]->shape, 
                                    record->inputs[0]->model,
                                    record->inputs[0]->dtype);
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_input = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_input) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_reshape_backward(grad_input, grad_out, record->inputs[0]);
            }
            
        case VSLA_OP_PAD_RANK:
            {
                vsla_tensor_t* grad_input = vsla_get_gradient(tape, record->inputs[0]);
                
                // Create zero gradient if it doesn't exist
                if (!grad_input) {
                    vsla_tensor_t* zero_grad = vsla_zeros(record->inputs[0]->rank, 
                                    record->inputs[0]->shape, 
                                    record->inputs[0]->model,
                                    record->inputs[0]->dtype);
                    vsla_set_gradient(tape, record->inputs[0], zero_grad);
                    vsla_free(zero_grad);
                    grad_input = vsla_get_gradient(tape, record->inputs[0]);
                    if (!grad_input) return VSLA_ERROR_MEMORY;
                }
                
                return vsla_pad_rank_backward(grad_input, grad_out, record->inputs[0]);
            }
            
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