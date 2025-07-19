#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"

vsla_error_t cpu_reshape(vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]) {
    if (!tensor || !new_shape) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t old_numel = vsla_numel(tensor);
    uint64_t new_numel = 1;
    for (uint8_t i = 0; i < new_rank; ++i) {
        new_numel *= new_shape[i];
    }

    if (old_numel != new_numel) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }

    free(tensor->shape);
    free(tensor->cap);
    free(tensor->stride);

    tensor->rank = new_rank;
    tensor->shape = (uint64_t*)calloc(new_rank, sizeof(uint64_t));
    tensor->cap = (uint64_t*)calloc(new_rank, sizeof(uint64_t));
    tensor->stride = (uint64_t*)calloc(new_rank, sizeof(uint64_t));

    if (!tensor->shape || !tensor->cap || !tensor->stride) {
        return VSLA_ERROR_MEMORY;
    }

    size_t elem_size = vsla_dtype_size(tensor->dtype);
    for (uint8_t i = 0; i < new_rank; ++i) {
        tensor->shape[i] = new_shape[i];
        tensor->cap[i] = vsla_next_pow2(new_shape[i]);
    }

    tensor->stride[new_rank - 1] = elem_size;
    for (int i = new_rank - 2; i >= 0; --i) {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->cap[i + 1];
    }

    return VSLA_SUCCESS;
}

vsla_error_t cpu_broadcast(vsla_tensor_t* out, const vsla_tensor_t* in) {
    if (!out || !in) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (in->rank != 0) {
        return VSLA_ERROR_NOT_IMPLEMENTED; // Only scalar broadcasting is supported for now
    }

    if (in->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    if (in->dtype == VSLA_DTYPE_F64) {
        double value = *(const double*)in->cpu_data;
        return cpu_fill(out, value);
    } else if (in->dtype == VSLA_DTYPE_F32) {
        float value = *(const float*)in->cpu_data;
        return cpu_fill(out, (double)value);
    }

    return VSLA_ERROR_INVALID_DTYPE;
}
