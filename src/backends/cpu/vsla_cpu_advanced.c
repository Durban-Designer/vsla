#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"

vsla_error_t cpu_conv(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->rank != 1 || b->rank != 1 || out->rank != 1) {
        return VSLA_ERROR_INVALID_RANK;
    }

    if (out->shape[0] != a->shape[0] + b->shape[0] - 1) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }

    if (a->dtype == VSLA_DTYPE_F64) {
        const double* a_data = (const double*)a->cpu_data;
        const double* b_data = (const double*)b->cpu_data;
        double* out_data = (double*)out->cpu_data;

        for (uint64_t k = 0; k < out->shape[0]; ++k) {
            double sum = 0.0;
            for (uint64_t i = 0; i < a->shape[0]; ++i) {
                if (k >= i && k - i < b->shape[0]) {
                    sum += a_data[i] * b_data[k - i];
                }
            }
            out_data[k] = sum;
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t k = 0; k < out->shape[0]; ++k) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < a->shape[0]; ++i) {
                if (k >= i && k - i < b->shape[0]) {
                    sum += a_data[i] * b_data[k - i];
                }
            }
            out_data[k] = sum;
        }
    }

    out->cpu_valid = true;
    return VSLA_SUCCESS;
}

vsla_error_t cpu_kron(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }

    if (a->rank != 1 || b->rank != 1 || out->rank != 1) {
        return VSLA_ERROR_INVALID_RANK;
    }

    if (out->shape[0] != a->shape[0] * b->shape[0]) {
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
            for (uint64_t j = 0; j < b->shape[0]; ++j) {
                out_data[i * b->shape[0] + j] = a_data[i] * b_data[j];
            }
        }
    } else if (a->dtype == VSLA_DTYPE_F32) {
        const float* a_data = (const float*)a->cpu_data;
        const float* b_data = (const float*)b->cpu_data;
        float* out_data = (float*)out->cpu_data;

        for (uint64_t i = 0; i < a->shape[0]; ++i) {
            for (uint64_t j = 0; j < b->shape[0]; ++j) {
                out_data[i * b->shape[0] + j] = a_data[i] * b_data[j];
            }
        }
    }

    out->cpu_valid = true;
    return VSLA_SUCCESS;
}
