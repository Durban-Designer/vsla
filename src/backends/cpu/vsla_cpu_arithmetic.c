#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"

vsla_error_t cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    }

    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;

    return VSLA_SUCCESS;
}

vsla_error_t cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] - b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] - b_data[i];
        }
    }

    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;

    return VSLA_SUCCESS;
}

vsla_error_t cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    if (!out || !in) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (in->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t n = vsla_numel(in);
    if (n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (in->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)in->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] * scalar;
        }
    } else if (in->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)in->cpu_data;
        float* out_data = (float*)out->cpu_data;
        float fscalar = (float)scalar;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] * fscalar;
        }
    }

    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;

    return VSLA_SUCCESS;
}

vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    uint64_t n = vsla_numel(a);
    if (n != vsla_numel(b) || n != vsla_numel(out)) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < n; i++) {
            out_data[i] = a_data[i] * b_data[i];
        }
    }

    out->cpu_valid = true;
    out->gpu_valid = false;
    out->location = VSLA_BACKEND_CPU;

    return VSLA_SUCCESS;
}

vsla_error_t cpu_fill(vsla_tensor_t* tensor, double value) {
    if (!tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_SUCCESS; /* Nothing to fill */
    }

    if (tensor->dtype == VSLA_DTYPE_F64) {
        double* data = (double*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = value;
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        float* data = (float*)tensor->cpu_data;
        float val_f32 = (float)value;
        for (uint64_t i = 0; i < n; i++) {
            data[i] = val_f32;
        }
    } else {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    tensor->cpu_valid = true;
    tensor->gpu_valid = false;
    tensor->location = VSLA_BACKEND_CPU;

    return VSLA_SUCCESS;
}
