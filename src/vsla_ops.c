/**
 * @file vsla_ops.c
 * @brief Basic operations on VSLA tensors
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L

#include "vsla/vsla_ops.h"
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
    
    if (out != tensor) {
        // We need 'out' to become an exact copy of 'tensor' before scaling it.
        // This means 'out' must have the same rank, shape, capacity, stride, and data.
        
        // First, free all existing internally allocated memory for 'out'.
        // Important: Do NOT free 'out' itself, as it's a pointer to a struct that was
        // likely allocated by the caller or exists on the stack.
        free(out->shape);
        free(out->cap);
        free(out->stride);
        if (out->data) {
            aligned_free_wrapper(out->data);
        }
        
        // Now, deep copy the contents of 'tensor' into 'out'.
        // This is essentially doing what vsla_copy does, but into an existing struct.
        
        // Copy basic metadata
        out->rank = tensor->rank;
        out->model = tensor->model;
        out->dtype = tensor->dtype;
        out->flags = tensor->flags;
        
        if (tensor->rank > 0) {
            // Allocate new metadata arrays for 'out'
            out->shape = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            out->cap = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            out->stride = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
            
            if (!out->shape || !out->cap || !out->stride) {
                // If any allocation fails, we're in a bad state.
                // Try to free what was allocated, then return error.
                free(out->shape);
                free(out->cap);
                free(out->stride);
                out->shape = NULL; out->cap = NULL; out->stride = NULL;
                return VSLA_ERROR_MEMORY;
            }
            
            // Copy contents of metadata arrays
            memcpy(out->shape, tensor->shape, tensor->rank * sizeof(uint64_t));
            memcpy(out->cap, tensor->cap, tensor->rank * sizeof(uint64_t));
            memcpy(out->stride, tensor->stride, tensor->rank * sizeof(uint64_t));
            
            // Allocate and copy data buffer
            size_t data_size = vsla_capacity(tensor) * vsla_dtype_size(tensor->dtype);
            out->data = aligned_alloc_wrapper(ALIGNMENT, data_size);
            if (!out->data) {
                free(out->shape); free(out->cap); free(out->stride);
                out->shape = NULL; out->cap = NULL; out->stride = NULL;
                return VSLA_ERROR_MEMORY;
            }
            memcpy(out->data, tensor->data, data_size);
        } else {
            // Rank 0 tensor has no allocated shape, cap, stride, data
            out->shape = NULL;
            out->cap = NULL;
            out->stride = NULL;
            out->data = NULL;
        }
    }
    
    // Now 'out' is either the same as 'tensor' or a deep copy of 'tensor'.
    // Proceed with scaling its data.
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

vsla_error_t vsla_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, 
                           const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Pad to common rank
    vsla_tensor_t* padded_a = vsla_copy(a);
    vsla_tensor_t* padded_b = vsla_copy(b);
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
    
    // Element-wise multiplication
    uint64_t n = vsla_numel(padded_a);
    
    if (padded_a->dtype == VSLA_DTYPE_F64 && padded_b->dtype == VSLA_DTYPE_F64) {
        double* data_a = (double*)padded_a->data;
        double* data_b = (double*)padded_b->data;
        double* data_out = (double*)out->data;
        
        for (uint64_t i = 0; i < n; i++) {
            data_out[i] = data_a[i] * data_b[i];
        }
    } else {
        // Handle mixed types through get/set interface
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

vsla_error_t vsla_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, 
                         const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Only support 2D matrix multiplication for now
    if (a->rank != 2 || b->rank != 2 || out->rank != 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint64_t m = a->shape[0];    // rows in A
    uint64_t k = a->shape[1];    // cols in A (must match rows in B)
    uint64_t n = b->shape[1];    // cols in B
    
    // Check dimensions match
    if (b->shape[0] != k) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    // Check output dimensions
    if (out->shape[0] != m || out->shape[1] != n) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    // Initialize output to zero
    vsla_error_t err = vsla_fill(out, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    // Perform matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
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

vsla_error_t vsla_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For now, only implement 2D transpose
    if (tensor->rank != 2) {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    uint64_t rows = tensor->shape[0];
    uint64_t cols = tensor->shape[1];
    
    // Ensure output has transposed dimensions
    if (out->rank != 2 || out->shape[0] != cols || out->shape[1] != rows) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Transpose elements
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

vsla_error_t vsla_reshape(vsla_tensor_t* tensor, uint8_t new_rank, 
                          const uint64_t new_shape[]) {
    if (!tensor || !new_shape) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Calculate total elements in new shape
    uint64_t new_numel = 1;
    for (uint8_t i = 0; i < new_rank; i++) {
        new_numel *= new_shape[i];
    }
    
    // Must preserve total number of elements
    if (new_numel != vsla_numel(tensor)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Allocate new metadata
    uint64_t* shape = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    uint64_t* cap = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    uint64_t* stride = (uint64_t*)malloc(new_rank * sizeof(uint64_t));
    
    if (!shape || !cap || !stride) {
        free(shape);
        free(cap);
        free(stride);
        return VSLA_ERROR_MEMORY;
    }
    
    // Copy new shape and calculate capacities and strides
    for (uint8_t i = 0; i < new_rank; i++) {
        shape[i] = new_shape[i];
        cap[i] = new_shape[i]; // Minimal capacity
    }
    
    // Calculate strides (row-major)
    stride[new_rank - 1] = 1;
    for (int8_t i = new_rank - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * cap[i + 1];
    }
    
    // Update tensor metadata
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    
    tensor->rank = new_rank;
    tensor->shape = shape;
    tensor->cap = cap;
    tensor->stride = stride;
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_max(const vsla_tensor_t* tensor, double* max) {
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

vsla_error_t vsla_min(const vsla_tensor_t* tensor, double* min) {
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