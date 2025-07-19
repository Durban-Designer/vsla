#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"

vsla_error_t cpu_matmul(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->rank != 2 || b->rank != 2 || out->rank != 2) {
        return VSLA_ERROR_INVALID_RANK;
    }

    if (a->shape[1] != b->shape[0]) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }

    if (out->shape[0] != a->shape[0] || out->shape[1] != b->shape[1]) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < a->shape[0]; ++i) {
            for (uint64_t j = 0; j < b->shape[1]; ++j) {
                double sum = 0.0;
                for (uint64_t k = 0; k < a->shape[1]; ++k) {
                    sum += a_data[i * a->shape[1] + k] * b_data[k * b->shape[1] + j];
                }
                out_data[i * out->shape[1] + j] = sum;
            }
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < a->shape[0]; ++i) {
            for (uint64_t j = 0; j < b->shape[1]; ++j) {
                float sum = 0.0f;
                for (uint64_t k = 0; k < a->shape[1]; ++k) {
                    sum += a_data[i * a->shape[1] + k] * b_data[k * b->shape[1] + j];
                }
                out_data[i * out->shape[1] + j] = sum;
            }
        }
    }

    out->cpu_valid = true;
    return VSLA_SUCCESS;
}

vsla_error_t cpu_transpose(vsla_tensor_t* out, const vsla_tensor_t* tensor) {
    if (!out || !tensor) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (tensor->rank != 2 || out->rank != 2) {
        return VSLA_ERROR_INVALID_RANK;
    }

    if (out->shape[0] != tensor->shape[1] || out->shape[1] != tensor->shape[0]) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }

    if (tensor->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* in_data = (const double*)tensor->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t i = 0; i < tensor->shape[0]; ++i) {
            for (uint64_t j = 0; j < tensor->shape[1]; ++j) {
                out_data[j * out->shape[1] + i] = in_data[i * tensor->shape[1] + j];
            }
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* in_data = (const float*)tensor->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < tensor->shape[0]; ++i) {
            for (uint64_t j = 0; j < tensor->shape[1]; ++j) {
                out_data[j * out->shape[1] + i] = in_data[i * tensor->shape[1] + j];
            }
        }
    }

    out->cpu_valid = true;
    return VSLA_SUCCESS;
}
