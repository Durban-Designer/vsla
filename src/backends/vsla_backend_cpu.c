/**
 * @file vsla_backend_cpu.c
 * @brief CPU backend for VSLA operations.
 *
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L

#include "vsla/vsla_backend_cpu.h"
#include "vsla/vsla_ops.h"
#include "vsla/vsla_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ALIGNMENT 64

#ifdef _WIN32
#include <malloc.h>
static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

static void aligned_free_wrapper(void* ptr) {
    _aligned_free(ptr);
}
#else
#include <stdlib.h>
static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void aligned_free_wrapper(void* ptr) {
    free(ptr);
}
#endif

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

vsla_error_t vsla_cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, 
                            const vsla_tensor_t* b) {
    vsla_error_t err = ensure_compatible_shapes(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    /* Handle zero element case */
    if (a->rank == 0) {
        return vsla_copy_basic(b) ? VSLA_SUCCESS : VSLA_ERROR_MEMORY;
    }
    if (b->rank == 0) {
        return vsla_copy_basic(a) ? VSLA_SUCCESS : VSLA_ERROR_MEMORY;
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

vsla_error_t vsla_cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, 
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

vsla_error_t vsla_cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, 
                              double scalar) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (out != tensor) {
        free(out->shape);
        free(out->cap);
        free(out->stride);
        if (out->data) {
            aligned_free_wrapper(out->data);
        }
        
        out->rank = tensor->rank;
        out->model = tensor->model;
        out->dtype = tensor->dtype;
        out->flags = tensor->flags;
        
        if (tensor->rank > 0) {
            out->shape = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            out->cap = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            out->stride = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            
            if (!out->shape || !out->cap || !out->stride) {
                free(out->shape);
                free(out->cap);
                free(out->stride);
                out->shape = NULL; out->cap = NULL; out->stride = NULL;
                return VSLA_ERROR_MEMORY;
            }
            
            memcpy(out->shape, tensor->shape, tensor->rank * sizeof(uint64_t));
            memcpy(out->cap, tensor->cap, tensor->rank * sizeof(uint64_t));
            memcpy(out->stride, tensor->stride, tensor->rank * sizeof(uint64_t));
            
            size_t data_size = vsla_capacity(tensor) * vsla_dtype_size(tensor->dtype);
            out->data = aligned_alloc_wrapper(ALIGNMENT, data_size);
            if (!out->data) {
                free(out->shape); free(out->cap); free(out->stride);
                out->shape = NULL; out->cap = NULL; out->stride = NULL;
                return VSLA_ERROR_MEMORY;
            }
            memcpy(out->data, tensor->data, data_size);
        } else {
            out->shape = NULL;
            out->cap = NULL;
            out->stride = NULL;
            out->data = NULL;
        }
    }
    
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

vsla_error_t vsla_cpu_norm(const vsla_tensor_t* tensor, double* norm) {
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

vsla_error_t vsla_cpu_sum(const vsla_tensor_t* tensor, double* sum) {
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

vsla_error_t vsla_cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, 
                                 const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    vsla_tensor_t* padded_a = vsla_copy_basic(a);
    vsla_tensor_t* padded_b = vsla_copy_basic(b);
    if (!padded_a || !padded_b) {
        vsla_free(padded_a);
        vsla_free(padded_b);
        return VSLA_ERROR_MEMORY;
    }
    
    uint8_t max_rank = (a->rank > b->rank) ? a->rank : b->rank;
    
    vsla_error_t err = vsla_pad_rank(padded_a, max_rank, NULL);
    if (err != VSLA_SUCCESS) {
        vsla_free(padded_a);
        vsla_free(padded_b);
        return err;
    }
    
    err = vsla_pad_rank(padded_b, max_rank, NULL);
    if (err != VSLA_SUCCESS) {
        vsla_free(padded_a);
        vsla_free(padded_b);
        return err;
    }
    
    uint64_t n = vsla_numel(padded_a);
    
    if (padded_a->dtype == VSLA_DTYPE_F64 && padded_b->dtype == VSLA_DTYPE_F64) {
        double* data_a = (double*)padded_a->data;
        double* data_b = (double*)padded_b->data;
        double* data_out = (double*)out->data;
        
        for (uint64_t i = 0; i < n; i++) {
            data_out[i] = data_a[i] * data_b[i];
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            uint64_t idx = i;
            double val_a, val_b;
            vsla_get_f64(padded_a, &idx, &val_a);
            vsla_get_f64(padded_b, &idx, &val_b);
            vsla_set_f64(out, &idx, val_a * val_b);
        }
    }
    
    vsla_free(padded_a);
    vsla_free(padded_b);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, 
                               const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (a->rank != 2 || b->rank != 2 || out->rank != 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t m = a->shape[0];
    uint64_t k = a->shape[1];
    uint64_t n = b->shape[1];
    
    if (b->shape[0] != k) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    if (out->shape[0] != m || out->shape[1] != n) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    vsla_error_t err = vsla_fill_basic(out, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint64_t ki = 0; ki < k; ki++) {
                double a_val, b_val;
                uint64_t a_indices[2] = {i, ki};
                uint64_t b_indices[2] = {ki, j};
                
                err = vsla_get_f64(a, a_indices, &a_val);
                if (err != VSLA_SUCCESS) return err;
                
                err = vsla_get_f64(b, b_indices, &b_val);
                if (err != VSLA_SUCCESS) return err;
                
                sum += a_val * b_val;
            }
            
            uint64_t out_indices[2] = {i, j};
            err = vsla_set_f64(out, out_indices, sum);
            if (err != VSLA_SUCCESS) return err;
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->rank != 2) {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    uint64_t rows = tensor->shape[0];
    uint64_t cols = tensor->shape[1];
    
    if (out->rank != 2 || out->shape[0] != cols || out->shape[1] != rows) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j < cols; j++) {
            uint64_t src_idx[] = {i, j};
            uint64_t dst_idx[] = {j, i};
            double val;
            
            vsla_get_f64(tensor, src_idx, &val);
            vsla_set_f64(out, dst_idx, val);
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_reshape(vsla_tensor_t* tensor, uint8_t new_rank, 
                                const uint64_t new_shape[]) {
    if (!tensor || !new_shape) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t new_numel = 1;
    for (uint8_t i = 0; i < new_rank; i++) {
        new_numel *= new_shape[i];
    }
    
    if (new_numel != vsla_numel(tensor)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t* shape = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    uint64_t* cap = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    uint64_t* stride = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    
    if (!shape || !cap || !stride) {
        free(shape);
        free(cap);
        free(stride);
        return VSLA_ERROR_MEMORY;
    }
    
    for (uint8_t i = 0; i < new_rank; i++) {
        shape[i] = new_shape[i];
        cap[i] = new_shape[i];
    }
    
    stride[new_rank - 1] = 1;
    for (int8_t i = new_rank - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * cap[i + 1];
    }
    
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    
    tensor->rank = new_rank;
    tensor->shape = shape;
    tensor->cap = cap;
    tensor->stride = stride;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_max(const vsla_tensor_t* tensor, double* max) {
    if (!tensor || !max) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    *max = -INFINITY;
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            if (data[i] > *max) {
                *max = data[i];
            }
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            double val = (double)data[i];
            if (val > *max) {
                *max = val;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_min(const vsla_tensor_t* tensor, double* min) {
    if (!tensor || !min) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    *min = INFINITY;
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            if (data[i] < *min) {
                *min = data[i];
            }
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            double val = (double)data[i];
            if (val < *min) {
                *min = val;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cpu_fill(vsla_tensor_t* tensor, double value) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_SUCCESS;
    }
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->data;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = value;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->data;
        float fvalue = (float)value;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = fvalue;
        }
    }
    
    return VSLA_SUCCESS;
}
