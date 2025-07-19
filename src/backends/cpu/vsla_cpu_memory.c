#define _POSIX_C_SOURCE 200809L
#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <stdlib.h>
#include <string.h>

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

vsla_error_t cpu_allocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }

    size_t size = vsla_numel(tensor) * vsla_dtype_size(tensor->dtype);
    tensor->cpu_data = aligned_alloc_wrapper(ALIGNMENT, size);
    
    if (!tensor->cpu_data) {
        return VSLA_ERROR_MEMORY;
    }

    memset(tensor->cpu_data, 0, size);
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

vsla_error_t cpu_deallocate(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (tensor->cpu_data) {
        aligned_free_wrapper(tensor->cpu_data);
        tensor->cpu_data = NULL;
    }
    
    tensor->cpu_valid = false;
    
    return VSLA_SUCCESS;
}

vsla_error_t cpu_copy_to_device(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    /* For CPU backend, device and host are the same */
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

vsla_error_t cpu_copy_to_host(vsla_tensor_t* tensor) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    /* For CPU backend, device and host are the same */
    tensor->cpu_valid = true;
    tensor->location = VSLA_BACKEND_CPU;
    
    return VSLA_SUCCESS;
}

vsla_error_t cpu_synchronize(void) {
    /* CPU operations are synchronous, nothing to do */
    return VSLA_SUCCESS;
}

vsla_error_t cpu_init(void* config) {
    (void)config; /* Unused */
    /* CPU backend doesn't require initialization */
    return VSLA_SUCCESS;
}

void cpu_cleanup(void) {
    /* CPU backend doesn't require cleanup */
}