/**
 * @file vsla_backend_cpu.c
 * @brief CPU backend implementation for VSLA
 * 
 * Pure CPU implementations using optimized C code and vendor CPU libraries.
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_ops.h"
#include "vsla/vsla_conv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// CPU backend state
static struct {
    bool initialized;
    char cpu_name[256];
    size_t cache_size;
    int num_cores;
} cpu_state = {0};

// CPU backend capabilities
typedef struct {
    bool supports_avx2;
    bool supports_fma;
    bool supports_openmp;
    const char* blas_library;  // "OpenBLAS", "Intel MKL", etc.
} cpu_capabilities_t;

static cpu_capabilities_t cpu_caps = {0};

// Initialize CPU backend
vsla_error_t vsla_backend_cpu_init(void) {
    if (cpu_state.initialized) return VSLA_SUCCESS;
    
    // Detect CPU capabilities
    strcpy(cpu_state.cpu_name, "Generic CPU");
    cpu_state.cache_size = 8 * 1024 * 1024;  // 8MB default
    cpu_state.num_cores = 4;  // Default assumption
    
    // TODO: Use cpuid or similar to detect actual CPU features
    cpu_caps.supports_avx2 = false;  // Conservative default
    cpu_caps.supports_fma = false;
    cpu_caps.supports_openmp = false;
    cpu_caps.blas_library = "None";
    
    cpu_state.initialized = true;
    return VSLA_SUCCESS;
}

// Cleanup CPU backend
void vsla_backend_cpu_cleanup(void) {
    if (!cpu_state.initialized) return;
    cpu_state.initialized = false;
}

// Get CPU backend info
vsla_error_t vsla_backend_cpu_get_info(char* name, size_t name_size, 
                                       size_t* memory_gb, 
                                       vsla_backend_t* backend) {
    if (!cpu_state.initialized) {
        vsla_error_t err = vsla_backend_cpu_init();
        if (err != VSLA_SUCCESS) return err;
    }
    
    if (name && name_size > 0) {
        strncpy(name, cpu_state.cpu_name, name_size - 1);
        name[name_size - 1] = '\0';
    }
    
    if (memory_gb) {
        // Estimate available system memory
        *memory_gb = 8;  // Conservative default
    }
    
    if (backend) {
        *backend = VSLA_BACKEND_CPU;
    }
    
    return VSLA_SUCCESS;
}

// CPU tensor addition
vsla_error_t vsla_backend_cpu_add(vsla_tensor_t* out,
                                  const vsla_tensor_t* a,
                                  const vsla_tensor_t* b) {
    if (!out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Use existing VSLA CPU implementation
    return vsla_add(out, a, b);
}

// CPU tensor scaling
vsla_error_t vsla_backend_cpu_scale(vsla_tensor_t* out,
                                    const vsla_tensor_t* in,
                                    double scalar) {
    if (!out || !in) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Use existing VSLA CPU implementation
    return vsla_scale(out, in, scalar);
}

// CPU matrix multiplication
vsla_error_t vsla_backend_cpu_matmul(vsla_tensor_t* out,
                                     const vsla_tensor_t* a,
                                     const vsla_tensor_t* b) {
    if (!out || !a || !b) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Check tensor dimensions
    if (a->rank != 2 || b->rank != 2) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    uint8_t out_rank;
    const uint64_t* out_shape;
    vsla_tensor_get_info(out, &out_rank, &out_shape, NULL, NULL);
    
    if (out_rank != 2) return VSLA_ERROR_INVALID_ARGUMENT;
    
    uint64_t m = a->shape[0];
    uint64_t k = a->shape[1];
    uint64_t n = b->shape[1];
    
    if (b->shape[0] != k || out_shape[0] != m || out_shape[1] != n) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Simple CPU matrix multiplication (could be optimized with BLAS)
    if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)vsla_tensor_data(a, NULL);
        const float* b_data = (const float*)vsla_tensor_data(b, NULL);
        float* out_data = (float*)vsla_tensor_data_mut(out, NULL);
        
        for (uint64_t i = 0; i < m; i++) {
            for (uint64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (uint64_t l = 0; l < k; l++) {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                out_data[i * n + j] = sum;
            }
        }
    } else if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)vsla_tensor_data(a, NULL);
        const double* b_data = (const double*)vsla_tensor_data(b, NULL);
        double* out_data = (double*)vsla_tensor_data_mut(out, NULL);
        
        for (uint64_t i = 0; i < m; i++) {
            for (uint64_t j = 0; j < n; j++) {
                double sum = 0.0;
                for (uint64_t l = 0; l < k; l++) {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                out_data[i * n + j] = sum;
            }
        }
    } else {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    return VSLA_SUCCESS;
}

// CPU convolution
vsla_error_t vsla_backend_cpu_conv(vsla_tensor_t* out,
                                   const vsla_tensor_t* signal,
                                   const vsla_tensor_t* kernel) {
    if (!out || !signal || !kernel) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Use existing VSLA CPU convolution implementation
    return vsla_conv(out, signal, kernel);
}

// CPU activation functions
vsla_error_t vsla_backend_cpu_relu(vsla_tensor_t* out,
                                   const vsla_tensor_t* in) {
    if (!out || !in) return VSLA_ERROR_INVALID_ARGUMENT;
    
    size_t total_elements = 1;
    for (uint8_t i = 0; i < in->rank; i++) {
        total_elements *= in->shape[i];
    }
    
    if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)vsla_tensor_data(in, NULL);
        float* out_data = (float*)vsla_tensor_data_mut(out, NULL);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_data[i] = fmaxf(0.0f, in_data[i]);
        }
    } else if (in->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)vsla_tensor_data(in, NULL);
        double* out_data = (double*)vsla_tensor_data_mut(out, NULL);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_data[i] = fmax(0.0, in_data[i]);
        }
    } else {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_backend_cpu_sigmoid(vsla_tensor_t* out,
                                      const vsla_tensor_t* in) {
    if (!out || !in) return VSLA_ERROR_INVALID_ARGUMENT;
    
    size_t total_elements = 1;
    for (uint8_t i = 0; i < in->rank; i++) {
        total_elements *= in->shape[i];
    }
    
    if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)vsla_tensor_data(in, NULL);
        float* out_data = (float*)vsla_tensor_data_mut(out, NULL);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_data[i] = 1.0f / (1.0f + expf(-in_data[i]));
        }
    } else if (in->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)vsla_tensor_data(in, NULL);
        double* out_data = (double*)vsla_tensor_data_mut(out, NULL);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_data[i] = 1.0 / (1.0 + exp(-in_data[i]));
        }
    } else {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    return VSLA_SUCCESS;
}

// CPU reduction operations
vsla_error_t vsla_backend_cpu_sum(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) return VSLA_ERROR_INVALID_ARGUMENT;
    
    // Use existing VSLA CPU implementation
    return vsla_sum(tensor, result);
}

vsla_error_t vsla_backend_cpu_mean(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) return VSLA_ERROR_INVALID_ARGUMENT;
    
    vsla_error_t err = vsla_sum(tensor, result);
    if (err != VSLA_SUCCESS) return err;
    
    // Calculate total elements
    uint64_t total = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        total *= tensor->shape[i];
    }
    
    *result /= (double)total;
    return VSLA_SUCCESS;
}

// Backend registration structure
typedef struct {
    const char* name;
    vsla_backend_t backend_type;
    vsla_error_t (*init)(void);
    void (*cleanup)(void);
    vsla_error_t (*get_info)(char*, size_t, size_t*, vsla_backend_t*);
    
    // Operations
    vsla_error_t (*add)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*scale)(vsla_tensor_t*, const vsla_tensor_t*, double);
    vsla_error_t (*matmul)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*conv)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*relu)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sigmoid)(vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*sum)(const vsla_tensor_t*, double*);
    vsla_error_t (*mean)(const vsla_tensor_t*, double*);
} vsla_backend_interface_t;

// CPU backend interface
static const vsla_backend_interface_t cpu_backend = {
    .name = "CPU",
    .backend_type = VSLA_BACKEND_CPU,
    .init = vsla_backend_cpu_init,
    .cleanup = vsla_backend_cpu_cleanup,
    .get_info = vsla_backend_cpu_get_info,
    .add = vsla_backend_cpu_add,
    .scale = vsla_backend_cpu_scale,
    .matmul = vsla_backend_cpu_matmul,
    .conv = vsla_backend_cpu_conv,
    .relu = vsla_backend_cpu_relu,
    .sigmoid = vsla_backend_cpu_sigmoid,
    .sum = vsla_backend_cpu_sum,
    .mean = vsla_backend_cpu_mean
};

// Get CPU backend interface
const vsla_backend_interface_t* vsla_get_cpu_backend(void) {
    return &cpu_backend;
}