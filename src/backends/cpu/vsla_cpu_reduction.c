#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor_internal.h"
#include <math.h>

vsla_error_t cpu_sum(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        *result = 0.0;
        return VSLA_SUCCESS;
    }

    *result = 0.0;

    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            *result += data[i];
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; i++) {
            *result += (double)data[i];
        }
    }

    return VSLA_SUCCESS;
}

vsla_error_t cpu_mean(const vsla_tensor_t* tensor, double* result) {
    if (!tensor || !result) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    vsla_error_t err = cpu_sum(tensor, result);
    if (err != VSLA_SUCCESS) {
        return err;
    }

    *result /= (double)n;
    return VSLA_SUCCESS;
}

vsla_error_t cpu_norm(const vsla_tensor_t* tensor, double* norm) {
    if (!tensor || !norm) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        *norm = 0.0;
        return VSLA_SUCCESS;
    }

    double sum_sq = 0.0;
    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; ++i) {
            sum_sq += data[i] * data[i];
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->cpu_data;
        for (uint64_t i = 0; i < n; ++i) {
            sum_sq += (double)data[i] * (double)data[i];
        }
    }

    *norm = sqrt(sum_sq);
    return VSLA_SUCCESS;
}

vsla_error_t cpu_max(const vsla_tensor_t* tensor, double* max) {
    if (!tensor || !max) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->cpu_data;
        *max = data[0];
        for (uint64_t i = 1; i < n; ++i) {
            if (data[i] > *max) {
                *max = data[i];
            }
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->cpu_data;
        *max = (double)data[0];
        for (uint64_t i = 1; i < n; ++i) {
            if ((double)data[i] > *max) {
                *max = (double)data[i];
            }
        }
    }

    return VSLA_SUCCESS;
}

vsla_error_t cpu_min(const vsla_tensor_t* tensor, double* min) {
    if (!tensor || !min) {
        return VSLA_ERROR_NULL_POINTER;
    }

    uint64_t n = vsla_numel(tensor);
    if (n == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }

    if (tensor->dtype == VSLA_DTYPE_F64) {
        const double* data = (const double*)tensor->cpu_data;
        *min = data[0];
        for (uint64_t i = 1; i < n; ++i) {
            if (data[i] < *min) {
                *min = data[i];
            }
        }
    } else if (tensor->dtype == VSLA_DTYPE_F32) {
        const float* data = (const float*)tensor->cpu_data;
        *min = (double)data[0];
        for (uint64_t i = 1; i < n; ++i) {
            if ((double)data[i] < *min) {
                *min = (double)data[i];
            }
        }
    }

    return VSLA_SUCCESS;
}
