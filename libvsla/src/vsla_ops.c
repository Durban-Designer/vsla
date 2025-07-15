/**
 * @file vsla_ops.c
 * @brief Basic operations on VSLA tensors
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

vsla_error_t vsla_pad_rank(vsla_tensor_t* tensor, uint8_t new_rank, 
                           const uint64_t target_cap[]) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (new_rank < tensor->rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (new_rank == tensor->rank) {
        return VSLA_SUCCESS;
    }
    
    /* Allocate new metadata arrays */
    uint64_t* new_shape = (uint64_t*)calloc(new_rank, sizeof(uint64_t));
    uint64_t* new_cap = (uint64_t*)calloc(new_rank, sizeof(uint64_t));
    uint64_t* new_stride = (uint64_t*)calloc(new_rank, sizeof(uint64_t));
    
    if (!new_shape || !new_cap || !new_stride) {
        free(new_shape);
        free(new_cap);
        free(new_stride);
        return VSLA_ERROR_MEMORY;
    }
    
    /* Copy existing dimensions */
    if (tensor->rank > 0) {
        memcpy(new_shape, tensor->shape, tensor->rank * sizeof(uint64_t));
        memcpy(new_cap, tensor->cap, tensor->rank * sizeof(uint64_t));
    }
    
    /* Set new dimensions */
    for (uint8_t i = tensor->rank; i < new_rank; i++) {
        new_shape[i] = 0;  /* New dimensions have shape 0 (implicit zeros) */
        if (target_cap && target_cap[i - tensor->rank] > 0) {
            new_cap[i] = vsla_next_pow2(target_cap[i - tensor->rank]);
        } else {
            new_cap[i] = 1;  /* Default capacity */
        }
    }
    
    /* Recompute strides */
    size_t elem_size = vsla_dtype_size(tensor->dtype);
    new_stride[new_rank - 1] = elem_size;
    for (int i = new_rank - 2; i >= 0; i--) {
        new_stride[i] = new_stride[i + 1] * new_cap[i + 1];
    }
    
    /* Update tensor metadata */
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    
    tensor->shape = new_shape;
    tensor->cap = new_cap;
    tensor->stride = new_stride;
    tensor->rank = new_rank;
    
    /* Note: We don't need to reallocate data because the original data
     * is still valid - new dimensions are implicitly zero */
    
    return VSLA_SUCCESS;
}

static vsla_error_t ensure_compatible_shapes(vsla_tensor_t* out, 
                                            const vsla_tensor_t* a, 
                                            const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (a->model != b->model || a->dtype != b->dtype) {
        return VSLA_ERROR_INCOMPATIBLE_MODELS;
    }
    
    /* Determine output rank */
    uint8_t max_rank = a->rank > b->rank ? a->rank : b->rank;
    
    /* Ensure output has correct rank */
    if (out->rank < max_rank) {
        vsla_error_t err = vsla_pad_rank(out, max_rank, NULL);
        if (err != VSLA_SUCCESS) return err;
    }
    
    /* Update output shape to max of inputs */
    for (uint8_t i = 0; i < max_rank; i++) {
        uint64_t dim_a = i < a->rank ? a->shape[i] : 0;
        uint64_t dim_b = i < b->rank ? b->shape[i] : 0;
        out->shape[i] = dim_a > dim_b ? dim_a : dim_b;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_add(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b) {
    vsla_error_t err = ensure_compatible_shapes(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    /* Handle zero element case */
    if (a->rank == 0) {
        return vsla_copy(b) ? VSLA_SUCCESS : VSLA_ERROR_MEMORY;
    }
    if (b->rank == 0) {
        return vsla_copy(a) ? VSLA_SUCCESS : VSLA_ERROR_MEMORY;
    }
    
    /* Perform element-wise addition */
    uint8_t max_rank = out->rank;
    uint64_t* indices = (uint64_t*)calloc(max_rank, sizeof(uint64_t));
    if (!indices) return VSLA_ERROR_MEMORY;
    
    /* Iterate over all elements in output shape */
    int done = 0;
    while (!done) {
        double val_a = 0.0, val_b = 0.0;
        
        /* Get value from a (0 if out of bounds) */
        int in_bounds_a = 1;
        for (uint8_t i = 0; i < max_rank; i++) {
            if (i >= a->rank || indices[i] >= a->shape[i]) {
                in_bounds_a = 0;
                break;
            }
        }
        if (in_bounds_a) {
            vsla_get_f64(a, indices, &val_a);
        }
        
        /* Get value from b (0 if out of bounds) */
        int in_bounds_b = 1;
        for (uint8_t i = 0; i < max_rank; i++) {
            if (i >= b->rank || indices[i] >= b->shape[i]) {
                in_bounds_b = 0;
                break;
            }
        }
        if (in_bounds_b) {
            vsla_get_f64(b, indices, &val_b);
        }
        
        /* Set output value */
        vsla_set_f64(out, indices, val_a + val_b);
        
        /* Increment indices */
        int carry = 1;
        for (int i = max_rank - 1; i >= 0 && carry; i--) {
            indices[i]++;
            if (indices[i] < out->shape[i]) {
                carry = 0;
            } else {
                indices[i] = 0;
            }
        }
        if (carry) done = 1;
    }
    
    free(indices);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_sub(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b) {
    vsla_error_t err = ensure_compatible_shapes(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    /* Similar to add but with subtraction */
    uint8_t max_rank = out->rank;
    uint64_t* indices = (uint64_t*)calloc(max_rank, sizeof(uint64_t));
    if (!indices) return VSLA_ERROR_MEMORY;
    
    int done = 0;
    while (!done) {
        double val_a = 0.0, val_b = 0.0;
        
        int in_bounds_a = 1;
        for (uint8_t i = 0; i < max_rank; i++) {
            if (i >= a->rank || indices[i] >= a->shape[i]) {
                in_bounds_a = 0;
                break;
            }
        }
        if (in_bounds_a) {
            vsla_get_f64(a, indices, &val_a);
        }
        
        int in_bounds_b = 1;
        for (uint8_t i = 0; i < max_rank; i++) {
            if (i >= b->rank || indices[i] >= b->shape[i]) {
                in_bounds_b = 0;
                break;
            }
        }
        if (in_bounds_b) {
            vsla_get_f64(b, indices, &val_b);
        }
        
        vsla_set_f64(out, indices, val_a - val_b);
        
        int carry = 1;
        for (int i = max_rank - 1; i >= 0 && carry; i--) {
            indices[i]++;
            if (indices[i] < out->shape[i]) {
                carry = 0;
            } else {
                indices[i] = 0;
            }
        }
        if (carry) done = 1;
    }
    
    free(indices);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, 
                        double scalar) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    /* Copy input to output if different */
    if (out != tensor) {
        vsla_tensor_t* temp = vsla_copy(tensor);
        if (!temp) return VSLA_ERROR_MEMORY;
        
        /* Free old output data and copy */
        vsla_free(out);
        *out = *temp;
        free(temp);  /* Just free the struct, not the data */
    }
    
    /* Scale all elements */
    uint64_t n = vsla_numel(out);
    
    if (out->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)out->data;
        for (uint64_t i = 0; i < n; i++) {
            data[i] *= scalar;
        }
    } else if (out->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)out->data;
        float fscalar = (float)scalar;
        for (uint64_t i = 0; i < n; i++) {
            data[i] *= fscalar;
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_norm(const vsla_tensor_t* tensor, double* norm) {
    if (!tensor || !norm) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    double sum = 0.0;
    uint64_t n = vsla_numel(tensor);
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            sum += data[i] * data[i];
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            sum += (double)(data[i] * data[i]);
        }
    }
    
    *norm = sqrt(sum);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_sum(const vsla_tensor_t* tensor, double* sum) {
    if (!tensor || !sum) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    *sum = 0.0;
    uint64_t n = vsla_numel(tensor);
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            *sum += data[i];
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            *sum += (double)data[i];
        }
    }
    
    return VSLA_SUCCESS;
}