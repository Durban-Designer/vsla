/**
 * @file vsla_backend_oneapi.c
 * @brief Intel oneAPI backend implementation for VSLA
 * 
 * Provides GPU/CPU acceleration using Intel oneAPI and vendor libraries (Intel MKL).
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_core.h"
#include <stdlib.h>
#include <string.h>

#ifdef VSLA_ENABLE_ONEAPI
#include <mkl.h>
#include <sycl/sycl.hpp>

// oneAPI backend state
static struct {
    bool initialized;
    bool gpu_available;
    bool cpu_optimized;
    sycl::queue* gpu_queue;
    sycl::queue* cpu_queue;
} oneapi_state = {0};

// Initialize oneAPI backend
vsla_error_t vsla_backend_oneapi_init(void) {
    if (oneapi_state.initialized) return VSLA_SUCCESS;
    
    try {
        // Try to create GPU queue
        oneapi_state.gpu_queue = new sycl::queue(sycl::gpu_selector_v);
        oneapi_state.gpu_available = true;
    } catch (const sycl::exception& e) {
        oneapi_state.gpu_queue = nullptr;
        oneapi_state.gpu_available = false;
    }
    
    try {
        // Create CPU queue for optimized CPU execution
        oneapi_state.cpu_queue = new sycl::queue(sycl::cpu_selector_v);
        oneapi_state.cpu_optimized = true;
    } catch (const sycl::exception& e) {
        oneapi_state.cpu_queue = nullptr;
        oneapi_state.cpu_optimized = false;
    }
    
    // If neither GPU nor optimized CPU is available, fail
    if (!oneapi_state.gpu_available && !oneapi_state.cpu_optimized) {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    oneapi_state.initialized = true;
    return VSLA_SUCCESS;
}

// Cleanup oneAPI backend
void vsla_backend_oneapi_cleanup(void) {
    if (!oneapi_state.initialized) return;
    
    if (oneapi_state.gpu_queue) {
        delete oneapi_state.gpu_queue;
        oneapi_state.gpu_queue = nullptr;
    }
    
    if (oneapi_state.cpu_queue) {
        delete oneapi_state.cpu_queue;
        oneapi_state.cpu_queue = nullptr;
    }
    
    oneapi_state.initialized = false;
}

#else // !VSLA_ENABLE_ONEAPI

// Stub implementations when oneAPI is not available
vsla_error_t vsla_backend_oneapi_init(void) {
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

void vsla_backend_oneapi_cleanup(void) {
    // Nothing to do
}

#endif // VSLA_ENABLE_ONEAPI

// Get oneAPI backend info
vsla_error_t vsla_backend_oneapi_get_info(char* name, size_t name_size,
                                          size_t* memory_gb,
                                          vsla_backend_t* backend) {
#ifdef VSLA_ENABLE_ONEAPI
    if (!oneapi_state.initialized) {
        vsla_error_t err = vsla_backend_oneapi_init();
        if (err != VSLA_SUCCESS) return err;
    }
    
    if (name && name_size > 0) {
        if (oneapi_state.gpu_available && oneapi_state.gpu_queue) {
            auto device = oneapi_state.gpu_queue->get_device();
            auto device_name = device.get_info<sycl::info::device::name>();
            snprintf(name, name_size, "Intel %s (oneAPI)", device_name.c_str());
        } else {
            strncpy(name, "Intel CPU (oneAPI)", name_size - 1);
            name[name_size - 1] = '\0';
        }
    }
    
    if (memory_gb) {
        if (oneapi_state.gpu_available && oneapi_state.gpu_queue) {
            auto device = oneapi_state.gpu_queue->get_device();
            auto mem_size = device.get_info<sycl::info::device::global_mem_size>();
            *memory_gb = mem_size / (1024.0 * 1024.0 * 1024.0);
        } else {
            *memory_gb = 8;  // Default for CPU
        }
    }
#else
    if (name && name_size > 0) {
        strncpy(name, "oneAPI (not available)", name_size - 1);
        name[name_size - 1] = '\0';
    }
    if (memory_gb) *memory_gb = 0;
#endif
    
    if (backend) *backend = VSLA_BACKEND_ONEAPI;
    return VSLA_SUCCESS;
}

// oneAPI tensor operations (stub implementations)
vsla_error_t vsla_backend_oneapi_add(vsla_tensor_t* out,
                                     const vsla_tensor_t* a,
                                     const vsla_tensor_t* b) {
    // TODO: Implement oneAPI GPU/CPU addition using SYCL
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_backend_oneapi_matmul(vsla_tensor_t* out,
                                        const vsla_tensor_t* a,
                                        const vsla_tensor_t* b) {
    // TODO: Implement oneAPI matrix multiplication using Intel MKL
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

vsla_error_t vsla_backend_oneapi_conv(vsla_tensor_t* out,
                                      const vsla_tensor_t* signal,
                                      const vsla_tensor_t* kernel) {
    // TODO: Implement oneAPI convolution using Intel MKL FFT
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

// Backend interface for oneAPI
typedef struct {
    const char* name;
    vsla_backend_t backend_type;
    vsla_error_t (*init)(void);
    void (*cleanup)(void);
    vsla_error_t (*get_info)(char*, size_t, size_t*, vsla_backend_t*);
    vsla_error_t (*add)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*matmul)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
    vsla_error_t (*conv)(vsla_tensor_t*, const vsla_tensor_t*, const vsla_tensor_t*);
} vsla_backend_interface_t;

static const vsla_backend_interface_t oneapi_backend = {
    .name = "oneAPI",
    .backend_type = VSLA_BACKEND_ONEAPI,
    .init = vsla_backend_oneapi_init,
    .cleanup = vsla_backend_oneapi_cleanup,
    .get_info = vsla_backend_oneapi_get_info,
    .add = vsla_backend_oneapi_add,
    .matmul = vsla_backend_oneapi_matmul,
    .conv = vsla_backend_oneapi_conv
};

// Get oneAPI backend interface
const vsla_backend_interface_t* vsla_get_oneapi_backend(void) {
    return &oneapi_backend;
}