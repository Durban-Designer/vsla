/**
 * @file vsla_core.c
 * @brief Core utility functions for VSLA library
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_core.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <string.h>
#include <stdlib.h>

const char* vsla_error_string(vsla_error_t error) {
    switch (error) {
        case VSLA_SUCCESS:
            return "Success";
        case VSLA_ERROR_NULL_POINTER:
            return "Null pointer passed where not allowed";
        case VSLA_ERROR_INVALID_ARGUMENT:
            return "Invalid argument provided";
        case VSLA_ERROR_MEMORY:
            return "Memory allocation failed";
        case VSLA_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch in operation";
        case VSLA_ERROR_INVALID_MODEL:
            return "Invalid model specified";
        case VSLA_ERROR_INVALID_DTYPE:
            return "Invalid data type specified";
        case VSLA_ERROR_IO:
            return "I/O operation failed";
        case VSLA_ERROR_NOT_IMPLEMENTED:
            return "Feature not yet implemented";
        case VSLA_ERROR_INVALID_RANK:
            return "Invalid rank (must be 0-255)";
        case VSLA_ERROR_OVERFLOW:
            return "Numeric overflow detected";
        case VSLA_ERROR_FFT:
            return "FFT operation failed";
        case VSLA_ERROR_INVALID_FILE:
            return "Invalid file format";
        case VSLA_ERROR_INCOMPATIBLE_MODELS:
            return "Incompatible models in operation";
        case VSLA_ERROR_GPU_FAILURE:
            return "GPU operation failed";
        case VSLA_ERROR_INVALID_STATE:
            return "Invalid object state";
        default:
            return "Unknown error";
    }
}

size_t vsla_dtype_size(vsla_dtype_t dtype) {
    switch (dtype) {
        case VSLA_DTYPE_F64:
            return sizeof(double);
        case VSLA_DTYPE_F32:
            return sizeof(float);
        default:
            return 0;
    }
}

uint64_t vsla_next_pow2(uint64_t n) {
    if (n == 0) return 1;
    if (n > (UINT64_MAX >> 1)) return 0; /* Overflow check */
    
    /* Round up to next power of 2 */
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    
    return n;
}

int vsla_is_pow2(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

vsla_error_t vsla_calculate_strides(vsla_tensor_t* tensor) {
    if (!tensor || !tensor->cap) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->rank == 0) {
        return VSLA_SUCCESS; /* No strides for scalar */
    }
    
    size_t elem_size = vsla_dtype_size(tensor->dtype);
    if (elem_size == 0) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    /* Calculate strides in reverse order (C-style layout) */
    tensor->stride[tensor->rank - 1] = elem_size;
    for (int i = tensor->rank - 2; i >= 0; i--) {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->cap[i + 1];
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_validate_tensor(const vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    if (tensor->rank > 0) {
        if (!tensor->shape || !tensor->cap || !tensor->stride) {
            return VSLA_ERROR_INVALID_STATE;
        }
    }
    
    if (tensor->dtype != VSLA_DTYPE_F32 && tensor->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    if (tensor->model != VSLA_MODEL_A && tensor->model != VSLA_MODEL_B) {
        return VSLA_ERROR_INVALID_MODEL;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_check_compatibility(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    vsla_error_t err = vsla_validate_tensor(a);
    if (err != VSLA_SUCCESS) return err;
    
    err = vsla_validate_tensor(b);
    if (err != VSLA_SUCCESS) return err;
    
    if (a->model != b->model) {
        return VSLA_ERROR_INCOMPATIBLE_MODELS;
    }
    
    if (a->dtype != b->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    return VSLA_SUCCESS;
}

size_t vsla_calc_linear_index(const vsla_tensor_t* tensor, const uint64_t indices[]) {
    if (!tensor || !indices || tensor->rank == 0) {
        return 0;
    }
    
    size_t linear_index = 0;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        linear_index += indices[i] * tensor->stride[i];
    }
    
    return linear_index;
}

bool vsla_indices_valid(const vsla_tensor_t* tensor, const uint64_t indices[]) {
    if (!tensor || !indices) {
        return false;
    }
    
    for (uint8_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->shape[i]) {
            return false;
        }
    }
    
    return true;
}

vsla_error_t vsla_copy_metadata(vsla_tensor_t* dst, const vsla_tensor_t* src) {
    if (!dst || !src) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    dst->rank = src->rank;
    dst->model = src->model;
    dst->dtype = src->dtype;
    dst->flags = src->flags;
    
    if (src->rank > 0) {
        /* Allocate arrays */
        dst->shape = (uint64_t*)malloc(src->rank * sizeof(uint64_t));
        dst->cap = (uint64_t*)malloc(src->rank * sizeof(uint64_t));
        dst->stride = (uint64_t*)malloc(src->rank * sizeof(uint64_t));
        
        if (!dst->shape || !dst->cap || !dst->stride) {
            free(dst->shape);
            free(dst->cap);
            free(dst->stride);
            return VSLA_ERROR_MEMORY;
        }
        
        /* Copy arrays */
        memcpy(dst->shape, src->shape, src->rank * sizeof(uint64_t));
        memcpy(dst->cap, src->cap, src->rank * sizeof(uint64_t));
        memcpy(dst->stride, src->stride, src->rank * sizeof(uint64_t));
    }
    
    return VSLA_SUCCESS;
}