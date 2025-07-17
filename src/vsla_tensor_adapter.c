/**
 * @file vsla_tensor_adapter.c
 * @brief Implementation of adapter functions for converting between different tensor types.
 *
 * @copyright MIT License
 */

#include "vsla/vsla_tensor_adapter.h"
#include <stdlib.h>
#include <string.h>

// Note: vsla_tensor is an opaque type defined in vsla_unified.c
// We cannot directly access its members here

vsla_tensor_t* vsla_unified_to_basic_tensor(vsla_tensor_t* unified_tensor) {
    if (!unified_tensor) return NULL;

    vsla_tensor_t* basic_tensor = (vsla_tensor_t*)malloc(sizeof(vsla_tensor_t));
    if (!basic_tensor) return NULL;

    basic_tensor->rank = unified_tensor->rank;
    basic_tensor->model = unified_tensor->model;
    basic_tensor->dtype = unified_tensor->dtype;
    basic_tensor->flags = 0; // Basic tensor doesn't have flags

    size_t shape_size = basic_tensor->rank * sizeof(uint64_t);
    basic_tensor->shape = (uint64_t*)malloc(shape_size);
    basic_tensor->cap = (uint64_t*)malloc(shape_size);
    basic_tensor->stride = (uint64_t*)malloc(shape_size);

    if (!basic_tensor->shape || !basic_tensor->cap || !basic_tensor->stride) {
        free(basic_tensor->shape);
        free(basic_tensor->cap);
        free(basic_tensor->stride);
        free(basic_tensor);
        return NULL;
    }

    memcpy(basic_tensor->shape, unified_tensor->shape, shape_size);
    memcpy(basic_tensor->cap, unified_tensor->cap, shape_size);
    memcpy(basic_tensor->stride, unified_tensor->stride, shape_size);

    basic_tensor->data = unified_tensor->cpu_data;

    return basic_tensor;
}

vsla_tensor_t* vsla_basic_to_unified_tensor(vsla_tensor_t* basic_tensor, vsla_context_t* ctx) {
    if (!basic_tensor) return NULL;

    vsla_tensor_t* unified_tensor = (vsla_tensor_t*)malloc(sizeof(vsla_tensor_t));
    if (!unified_tensor) return NULL;

    unified_tensor->rank = basic_tensor->rank;
    unified_tensor->model = basic_tensor->model;
    unified_tensor->dtype = basic_tensor->dtype;

    size_t shape_size = unified_tensor->rank * sizeof(uint64_t);
    unified_tensor->shape = (uint64_t*)malloc(shape_size);
    unified_tensor->cap = (uint64_t*)malloc(shape_size);
    unified_tensor->stride = (uint64_t*)malloc(shape_size);

    if (!unified_tensor->shape || !unified_tensor->cap || !unified_tensor->stride) {
        free(unified_tensor->shape);
        free(unified_tensor->cap);
        free(unified_tensor->stride);
        free(unified_tensor);
        return NULL;
    }

    memcpy(unified_tensor->shape, basic_tensor->shape, shape_size);
    memcpy(unified_tensor->cap, basic_tensor->cap, shape_size);
    memcpy(unified_tensor->stride, basic_tensor->stride, shape_size);

    unified_tensor->cpu_data = basic_tensor->data;
    unified_tensor->gpu_data = NULL;
    unified_tensor->data_size = 0; // This should be calculated
    unified_tensor->location = VSLA_BACKEND_CPU;
    unified_tensor->cpu_valid = true;
    unified_tensor->gpu_valid = false;
    unified_tensor->ctx = ctx;

    return unified_tensor;
}

vsla_gpu_tensor_t* vsla_unified_to_gpu_tensor(vsla_tensor_t* unified_tensor) {
    if (!unified_tensor) return NULL;

    vsla_gpu_tensor_t* gpu_tensor = (vsla_gpu_tensor_t*)malloc(sizeof(vsla_gpu_tensor_t));
    if (!gpu_tensor) return NULL;

    gpu_tensor->rank = unified_tensor->rank;
    gpu_tensor->model = unified_tensor->model;
    gpu_tensor->dtype = unified_tensor->dtype;
    gpu_tensor->flags = 0;

    size_t shape_size = gpu_tensor->rank * sizeof(uint64_t);
    gpu_tensor->shape = (uint64_t*)malloc(shape_size);
    gpu_tensor->cap = (uint64_t*)malloc(shape_size);
    gpu_tensor->stride = (uint64_t*)malloc(shape_size);

    if (!gpu_tensor->shape || !gpu_tensor->cap || !gpu_tensor->stride) {
        free(gpu_tensor->shape);
        free(gpu_tensor->cap);
        free(gpu_tensor->stride);
        free(gpu_tensor);
        return NULL;
    }

    memcpy(gpu_tensor->shape, unified_tensor->shape, shape_size);
    memcpy(gpu_tensor->cap, unified_tensor->cap, shape_size);
    memcpy(gpu_tensor->stride, unified_tensor->stride, shape_size);

    gpu_tensor->data = unified_tensor->cpu_data;
    gpu_tensor->gpu_data = unified_tensor->gpu_data;
    gpu_tensor->location = unified_tensor->location;
    gpu_tensor->gpu_id = 0; // This should be set from context
    gpu_tensor->gpu_capacity = 0; // This should be calculated

    return gpu_tensor;
}

vsla_tensor_t* vsla_gpu_to_unified_tensor(vsla_gpu_tensor_t* gpu_tensor, vsla_context_t* ctx) {
    if (!gpu_tensor) return NULL;

    vsla_tensor_t* unified_tensor = (vsla_tensor_t*)malloc(sizeof(vsla_tensor_t));
    if (!unified_tensor) return NULL;

    unified_tensor->rank = gpu_tensor->rank;
    unified_tensor->model = gpu_tensor->model;
    unified_tensor->dtype = gpu_tensor->dtype;

    size_t shape_size = unified_tensor->rank * sizeof(uint64_t);
    unified_tensor->shape = (uint64_t*)malloc(shape_size);
    unified_tensor->cap = (uint64_t*)malloc(shape_size);
    unified_tensor->stride = (uint64_t*)malloc(shape_size);

    if (!unified_tensor->shape || !unified_tensor->cap || !unified_tensor->stride) {
        free(unified_tensor->shape);
        free(unified_tensor->cap);
        free(unified_tensor->stride);
        free(unified_tensor);
        return NULL;
    }

    memcpy(unified_tensor->shape, gpu_tensor->shape, shape_size);
    memcpy(unified_tensor->cap, gpu_tensor->cap, shape_size);
    memcpy(unified_tensor->stride, gpu_tensor->stride, shape_size);

    unified_tensor->cpu_data = gpu_tensor->data;
    unified_tensor->gpu_data = gpu_tensor->gpu_data;
    unified_tensor->data_size = 0; // This should be calculated
    unified_tensor->location = gpu_tensor->location;
    unified_tensor->cpu_valid = (gpu_tensor->location == VSLA_GPU_LOCATION_CPU || gpu_tensor->location == VSLA_GPU_LOCATION_UNIFIED);
    unified_tensor->gpu_valid = (gpu_tensor->location == VSLA_GPU_LOCATION_GPU || gpu_tensor->location == VSLA_GPU_LOCATION_UNIFIED);
    unified_tensor->ctx = ctx;

    return unified_tensor;
}
