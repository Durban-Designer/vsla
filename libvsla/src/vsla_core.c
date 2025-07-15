/**
 * @file vsla_core.c
 * @brief Core utility functions for VSLA library
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_core.h"
#include <string.h>

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