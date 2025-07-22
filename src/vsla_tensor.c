#include "vsla/vsla_core.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_tensor_internal.h"

vsla_error_t vsla_get_f64(const vsla_tensor_t* tensor, const uint64_t* indices, double* value) {
    // This is a placeholder implementation
    return VSLA_SUCCESS;
}

vsla_error_t vsla_set_f64(vsla_tensor_t* tensor, const uint64_t* indices, double value) {
    // This is a placeholder implementation
    return VSLA_SUCCESS;
}

uint8_t vsla_get_rank(const vsla_tensor_t* tensor) {
    return tensor->rank;
}

vsla_error_t vsla_get_shape(const vsla_tensor_t* tensor, uint64_t* shape) {
    if (!tensor || !shape) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    memcpy(shape, tensor->shape, tensor->rank * sizeof(uint64_t));
    return VSLA_SUCCESS;
}

vsla_model_t vsla_get_model(const vsla_tensor_t* tensor) {
    return tensor->model;
}

vsla_dtype_t vsla_get_dtype(const vsla_tensor_t* tensor) {
    return tensor->dtype;
}

size_t vsla_numel(const vsla_tensor_t* tensor) {
    size_t numel = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        numel *= tensor->shape[i];
    }
    return numel;
}

int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (a->rank != b->rank) {
        return 0;
    }
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

vsla_backend_t vsla_get_location(const vsla_tensor_t* tensor) {
    return tensor->location;
}