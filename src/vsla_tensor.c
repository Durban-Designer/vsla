/**
 * @file vsla_tensor.c
 * @brief Core tensor data structure implementation
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200112L

#include "vsla/vsla_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <limits.h>

#ifdef _WIN32
#include <malloc.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#define ALIGNMENT 64
#define MAX_TENSOR_SIZE (1ULL << 40)  /* 1TB limit */

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    if (size == 0) return NULL;
    if (size > MAX_TENSOR_SIZE) return NULL;
    
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void aligned_free_wrapper(void* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

vsla_tensor_t* vsla_new(uint8_t rank, const uint64_t shape[], 
                        vsla_model_t model, vsla_dtype_t dtype) {
    /* Validate inputs */
    if (model != VSLA_MODEL_A && model != VSLA_MODEL_B) {
        return NULL;
    }
    if (dtype != VSLA_DTYPE_F64 && dtype != VSLA_DTYPE_F32) {
        return NULL;
    }
    if (rank > 0 && !shape) {
        return NULL;
    }
    
    /* Validate shape dimensions */
    if (rank > 0) {
        for (uint8_t i = 0; i < rank; i++) {
            if (shape[i] == 0) {
                return NULL;  /* Empty dimensions not allowed in shape */
            }
            if (shape[i] > (UINT64_MAX >> 1)) {
                return NULL;  /* Too large */
            }
        }
    }
    
    vsla_tensor_t* tensor = (vsla_tensor_t*)calloc(1, sizeof(vsla_tensor_t));
    if (!tensor) {
        return NULL;
    }
    
    tensor->rank = rank;
    tensor->model = (uint8_t)model;
    tensor->dtype = (uint8_t)dtype;
    tensor->flags = 0;
    
    if (rank > 0) {
        /* Allocate metadata arrays */
        tensor->shape = (uint64_t*)calloc(rank, sizeof(uint64_t));
        tensor->cap = (uint64_t*)calloc(rank, sizeof(uint64_t));
        tensor->stride = (uint64_t*)calloc(rank, sizeof(uint64_t));
        
        if (!tensor->shape || !tensor->cap || !tensor->stride) {
            vsla_free(tensor);
            return NULL;
        }
        
        /* Copy shape and compute capacities with overflow checking */
        uint64_t total_cap = 1;
        size_t elem_size = vsla_dtype_size(dtype);
        
        for (uint8_t i = 0; i < rank; i++) {
            tensor->shape[i] = shape[i];
            tensor->cap[i] = vsla_next_pow2(shape[i]);
            
            if (tensor->cap[i] == 0) {
                /* Overflow in next_pow2 */
                vsla_free(tensor);
                return NULL;
            }
            
            /* Check for overflow in total capacity */
            if (total_cap > UINT64_MAX / tensor->cap[i]) {
                vsla_free(tensor);
                return NULL;
            }
            total_cap *= tensor->cap[i];
        }
        
        /* Check total data size doesn't exceed limits */
        if (total_cap > MAX_TENSOR_SIZE / elem_size) {
            vsla_free(tensor);
            return NULL;
        }
        
        /* Compute strides (row-major order) */
        tensor->stride[rank - 1] = elem_size;
        for (int i = rank - 2; i >= 0; i--) {
            /* Check for stride overflow */
            if (tensor->stride[i + 1] > UINT64_MAX / tensor->cap[i + 1]) {
                vsla_free(tensor);
                return NULL;
            }
            tensor->stride[i] = tensor->stride[i + 1] * tensor->cap[i + 1];
        }
        
        /* Allocate data buffer */
        size_t data_size = total_cap * elem_size;
        tensor->data = aligned_alloc_wrapper(ALIGNMENT, data_size);
        if (!tensor->data) {
            vsla_free(tensor);
            return NULL;
        }
        
        /* Zero-initialize data */
        memset(tensor->data, 0, data_size);
        
        /* Initialize new fields */
        tensor->cpu_data = tensor->data;  /* For now, data and cpu_data point to same memory */
        tensor->gpu_data = NULL;
        tensor->data_size = data_size;
        tensor->location = VSLA_BACKEND_CPU;
        tensor->cpu_valid = true;
        tensor->gpu_valid = false;
        tensor->ctx = NULL;  /* Will be set by context when creating tensor */
    } else {
        /* For rank 0 tensors */
        tensor->cpu_data = NULL;
        tensor->gpu_data = NULL;
        tensor->data_size = 0;
        tensor->location = VSLA_BACKEND_CPU;
        tensor->cpu_valid = true;
        tensor->gpu_valid = false;
        tensor->ctx = NULL;
    }
    
    return tensor;
}

void vsla_free(vsla_tensor_t* tensor) {
    if (!tensor) return;
    
    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);
    
    /* Free CPU data (data and cpu_data point to same memory) */
    if (tensor->data) {
        aligned_free_wrapper(tensor->data);
    }
    
    /* GPU data would be freed by the backend/context if it exists */
    /* The context is responsible for freeing GPU memory */
    
    free(tensor);
}

vsla_tensor_t* vsla_copy_basic(const vsla_tensor_t* tensor) {
    if (!tensor) return NULL;
    
    vsla_tensor_t* copy = vsla_new(tensor->rank, tensor->shape, 
                                   (vsla_model_t)tensor->model, 
                                   (vsla_dtype_t)tensor->dtype);
    if (!copy) return NULL;
    
    /* Copy data - only copy CPU data for basic copy */
    if (tensor->rank > 0 && tensor->data && tensor->cpu_valid) {
        size_t data_size = vsla_capacity(tensor) * vsla_dtype_size(tensor->dtype);
        memcpy(copy->data, tensor->data, data_size);
    }
    
    /* Copy inherits the context from the original */
    copy->ctx = tensor->ctx;
    
    return copy;
}

vsla_tensor_t* vsla_zeros(uint8_t rank, const uint64_t shape[],
                          vsla_model_t model, vsla_dtype_t dtype) {
    return vsla_new(rank, shape, model, dtype);
}

vsla_tensor_t* vsla_ones(uint8_t rank, const uint64_t shape[],
                         vsla_model_t model, vsla_dtype_t dtype) {
    vsla_tensor_t* tensor = vsla_new(rank, shape, model, dtype);
    if (!tensor) return NULL;
    
    vsla_error_t err = vsla_fill_basic(tensor, 1.0);
    if (err != VSLA_SUCCESS) {
        vsla_free(tensor);
        return NULL;
    }
    
    return tensor;
}

uint64_t vsla_numel(const vsla_tensor_t* tensor) {
    if (!tensor || tensor->rank == 0) return 0;
    
    uint64_t n = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        n *= tensor->shape[i];
    }
    return n;
}

uint64_t vsla_capacity(const vsla_tensor_t* tensor) {
    if (!tensor || tensor->rank == 0) return 0;
    
    uint64_t n = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        n *= tensor->cap[i];
    }
    return n;
}

void* vsla_get_ptr(const vsla_tensor_t* tensor, const uint64_t indices[]) {
    if (!tensor || !tensor->data || !indices) return NULL;
    
    /* Check bounds */
    for (uint8_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->shape[i]) {
            return NULL;
        }
    }
    
    /* Compute offset */
    size_t offset = 0;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        offset += indices[i] * tensor->stride[i];
    }
    
    return (char*)tensor->data + offset;
}

vsla_error_t vsla_get_f64(const vsla_tensor_t* tensor, const uint64_t indices[], 
                          double* value) {
    if (!tensor || !indices || !value) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    void* ptr = vsla_get_ptr(tensor, indices);
    if (!ptr) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        *value = *(double*)ptr;
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        *value = (double)(*(float*)ptr);
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_set_f64(vsla_tensor_t* tensor, const uint64_t indices[], 
                          double value) {
    if (!tensor || !indices) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    void* ptr = vsla_get_ptr(tensor, indices);
    if (!ptr) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (tensor->dtype == VSLA_DTYPE_F64) {
        *(double*)ptr = value;
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        *(float*)ptr = (float)value;
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_fill_basic(vsla_tensor_t* tensor, double value) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->rank == 0) {
        return VSLA_SUCCESS;  /* Empty tensor, nothing to fill */
    }
    
    if (!tensor->data) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    /* Handle NaN and infinity values */
    if (isnan(value) || isinf(value)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    /* Iterate through all valid indices using strides */
    uint64_t* indices = (uint64_t*)calloc(tensor->rank, sizeof(uint64_t));
    if (!indices) {
        return VSLA_ERROR_MEMORY;
    }
    
    vsla_error_t result = VSLA_SUCCESS;
    int done = 0;
    
    while (!done) {
        /* Set value at current indices */
        vsla_error_t err = vsla_set_f64(tensor, indices, value);
        if (err != VSLA_SUCCESS) {
            result = err;
            break;
        }
        
        /* Increment indices */
        int carry = 1;
        for (int i = tensor->rank - 1; i >= 0 && carry; i--) {
            indices[i]++;
            if (indices[i] < tensor->shape[i]) {
                carry = 0;
            } else {
                indices[i] = 0;
            }
        }
        if (carry) done = 1;
    }
    
    free(indices);
    return result;
}

void vsla_print(const vsla_tensor_t* tensor, const char* name) {
    if (!tensor) {
        printf("%s: NULL\n", name ? name : "Tensor");
        return;
    }
    
    printf("%s:\n", name ? name : "Tensor");
    printf("  Rank: %u\n", tensor->rank);
    printf("  Model: %s\n", tensor->model == 0 ? "A (Convolution)" : "B (Kronecker)");
    printf("  Dtype: %s\n", tensor->dtype == 0 ? "f64" : "f32");
    
    if (tensor->rank > 0) {
        printf("  Shape: [");
        for (uint8_t i = 0; i < tensor->rank; i++) {
            printf("%llu%s", (unsigned long long)tensor->shape[i], 
                   i < tensor->rank - 1 ? ", " : "");
        }
        printf("]\n");
        
        printf("  Cap: [");
        for (uint8_t i = 0; i < tensor->rank; i++) {
            printf("%llu%s", (unsigned long long)tensor->cap[i], 
                   i < tensor->rank - 1 ? ", " : "");
        }
        printf("]\n");
        
        /* Print first few elements for 1D and 2D tensors */
        if (tensor->rank == 1 && tensor->shape[0] <= 10) {
            printf("  Data: [");
            for (uint64_t i = 0; i < tensor->shape[0]; i++) {
                double val = 0.0;
                uint64_t idx = i;
                vsla_get_f64(tensor, &idx, &val);
                printf("%.3f%s", val, i < tensor->shape[0] - 1 ? ", " : "");
            }
            printf("]\n");
        }
    }
}

int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!a || !b) return 0;
    if (a->rank != b->rank) return 0;
    
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    
    return 1;
}

vsla_tensor_t* vsla_zero_element(vsla_model_t model, vsla_dtype_t dtype) {
    return vsla_new(0, NULL, model, dtype);
}

vsla_tensor_t* vsla_one_element(vsla_model_t model, vsla_dtype_t dtype) {
    uint64_t shape = 1;
    vsla_tensor_t* one = vsla_new(1, &shape, model, dtype);
    if (!one) return NULL;
    
    vsla_error_t err = vsla_fill_basic(one, 1.0);
    if (err != VSLA_SUCCESS) {
        vsla_free(one);
        return NULL;
    }
    
    return one;
}