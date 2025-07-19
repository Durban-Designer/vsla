/**
 * @file vsla_backend_cuda_kernels.cu
 * @brief CUDA kernels for the VSLA CUDA backend
 * @copyright MIT License
 */

#include "vsla_backend_cuda_kernels.h"
#include "vsla/vsla_gpu_types.h"
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        return VSLA_ERROR_CUDA; 
    } 
} while(0)

__global__ void add_kernel_f64(double* out, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

vsla_error_t vsla_cuda_kernel_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64 || a->dtype != VSLA_DTYPE_F64 || b->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(out);
    double* d_out = (double*)out->gpu_data;
    const double* d_a = (const double*)a->gpu_data;
    const double* d_b = (const double*)b->gpu_data;

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    add_kernel_f64<<<grid_size, block_size>>>((double*)d_out, (const double*)d_a, (const double*)d_b, n);

    CUDA_CHECK(cudaGetLastError());
    return VSLA_SUCCESS;
}

__global__ void sub_kernel_f64(double* out, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

vsla_error_t vsla_cuda_kernel_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64 || a->dtype != VSLA_DTYPE_F64 || b->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(out);
    double* d_out = (double*)out->gpu_data;
    const double* d_a = (const double*)a->gpu_data;
    const double* d_b = (const double*)b->gpu_data;

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    sub_kernel_f64<<<grid_size, block_size>>>((double*)d_out, (const double*)d_a, (const double*)d_b, n);

    CUDA_CHECK(cudaGetLastError());
    return VSLA_SUCCESS;
}

__global__ void scale_kernel_f64(double* out, const double* in, double scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scalar;
    }
}

vsla_error_t vsla_cuda_kernel_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar) {
    if (out->dtype != VSLA_DTYPE_F64 || in->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(out);
    double* d_out = (double*)out->gpu_data;
    const double* d_in = (const double*)in->gpu_data;

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    scale_kernel_f64<<<grid_size, block_size>>>((double*)d_out, (const double*)d_in, scalar, n);

    CUDA_CHECK(cudaGetLastError());
    return VSLA_SUCCESS;
}

__global__ void hadamard_kernel_f64(double* out, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

vsla_error_t vsla_cuda_kernel_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (out->dtype != VSLA_DTYPE_F64 || a->dtype != VSLA_DTYPE_F64 || b->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(out);
    double* d_out = (double*)out->gpu_data;
    const double* d_a = (const double*)a->gpu_data;
    const double* d_b = (const double*)b->gpu_data;

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    hadamard_kernel_f64<<<grid_size, block_size>>>((double*)d_out, (const double*)d_a, (const double*)d_b, n);

    CUDA_CHECK(cudaGetLastError());
    return VSLA_SUCCESS;
}

__global__ void fill_kernel_f64(double* out, double value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

vsla_error_t vsla_cuda_kernel_fill(vsla_tensor_t* tensor, double value) {
    if (tensor->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(tensor);
    double* d_out = (double*)tensor->gpu_data;

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    fill_kernel_f64<<<grid_size, block_size>>>((double*)d_out, value, n);

    CUDA_CHECK(cudaGetLastError());
    return VSLA_SUCCESS;
}

__global__ void sum_kernel_f64(const double* in, double* out, size_t n) {
    extern __shared__ double sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

vsla_error_t vsla_cuda_kernel_sum(const vsla_tensor_t* tensor, double* result) {
    if (tensor->dtype != VSLA_DTYPE_F64) {
        return VSLA_ERROR_DTYPE_MISMATCH;
    }

    size_t n = vsla_tensor_get_num_elements(tensor);
    const double* d_in = (const double*)tensor->gpu_data;
    double* d_out;

    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    sum_kernel_f64<<<grid_size, block_size, block_size * sizeof(double)>>>((const double*)d_in, d_out, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    return VSLA_SUCCESS;
}
