/**
 * @file cublas_benchmark.c
 * @brief cuBLAS benchmark implementation for VSLA comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct {
    double mean_time_us;
    double std_time_us;
    double min_time_us;
    double max_time_us;
    size_t memory_mb;
} benchmark_result_t;

static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static size_t get_gpu_memory_usage(void) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return (total_mem - free_mem) / (1024 * 1024); // MB
}

static void print_result_json(const char* method, const char* operation, 
                             size_t size1, size_t size2, size_t iterations,
                             benchmark_result_t* result) {
    printf("{\n");
    printf("  \"method\": \"%s\",\n", method);
    printf("  \"operation\": \"%s\",\n", operation);
    printf("  \"size1\": %zu,\n", size1);
    printf("  \"size2\": %zu,\n", size2);
    printf("  \"iterations\": %zu,\n", iterations);
    printf("  \"mean_time_us\": %.3f,\n", result->mean_time_us);
    printf("  \"std_time_us\": %.3f,\n", result->std_time_us);
    printf("  \"min_time_us\": %.3f,\n", result->min_time_us);
    printf("  \"max_time_us\": %.3f,\n", result->max_time_us);
    printf("  \"memory_mb\": %zu\n", result->memory_mb);
    printf("}\n");
}

static benchmark_result_t benchmark_vector_addition(size_t size1, size_t size2, size_t iterations) {
    benchmark_result_t result = {0};
    
    // Allocate GPU memory
    size_t max_size = (size1 > size2) ? size1 : size2;
    double *d_a, *d_b, *d_result;
    
    cudaMalloc(&d_a, max_size * sizeof(double));
    cudaMalloc(&d_b, max_size * sizeof(double));
    cudaMalloc(&d_result, max_size * sizeof(double));
    
    // Initialize data
    double *h_a = calloc(max_size, sizeof(double));
    double *h_b = calloc(max_size, sizeof(double));
    
    for (size_t i = 0; i < size1; i++) {
        h_a[i] = (double)i;
    }
    for (size_t i = 0; i < size2; i++) {
        h_b[i] = (double)i;
    }
    
    cudaMemcpy(d_a, h_a, max_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, max_size * sizeof(double), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Warmup
    for (size_t i = 0; i < 5; i++) {
        cublasDcopy(handle, max_size, d_a, 1, d_result, 1);
        cublasDaxpy(handle, max_size, &(double){1.0}, d_b, 1, d_result, 1);
    }
    
    // Benchmark
    double *times = malloc(iterations * sizeof(double));
    
    for (size_t i = 0; i < iterations; i++) {
        cudaDeviceSynchronize();
        double start = get_wall_time();
        
        cublasDcopy(handle, max_size, d_a, 1, d_result, 1);
        cublasDaxpy(handle, max_size, &(double){1.0}, d_b, 1, d_result, 1);
        
        cudaDeviceSynchronize();
        double end = get_wall_time();
        
        times[i] = (end - start) * 1e6; // microseconds
    }
    
    // Calculate statistics
    double sum = 0.0;
    result.min_time_us = times[0];
    result.max_time_us = times[0];
    
    for (size_t i = 0; i < iterations; i++) {
        sum += times[i];
        if (times[i] < result.min_time_us) result.min_time_us = times[i];
        if (times[i] > result.max_time_us) result.max_time_us = times[i];
    }
    
    result.mean_time_us = sum / iterations;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (size_t i = 0; i < iterations; i++) {
        double diff = times[i] - result.mean_time_us;
        variance += diff * diff;
    }
    result.std_time_us = sqrt(variance / iterations);
    
    result.memory_mb = get_gpu_memory_usage();
    
    // Cleanup
    free(times);
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cublasDestroy(handle);
    
    return result;
}

static benchmark_result_t benchmark_matrix_multiplication(size_t m, size_t n, size_t k, size_t iterations) {
    benchmark_result_t result = {0};
    
    // Allocate GPU memory
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(double));
    cudaMalloc(&d_b, k * n * sizeof(double));
    cudaMalloc(&d_c, m * n * sizeof(double));
    
    // Initialize data
    double *h_a = malloc(m * k * sizeof(double));
    double *h_b = malloc(k * n * sizeof(double));
    
    for (size_t i = 0; i < m * k; i++) {
        h_a[i] = (double)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; i++) {
        h_b[i] = (double)rand() / RAND_MAX;
    }
    
    cudaMemcpy(d_a, h_a, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, k * n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const double alpha = 1.0, beta = 0.0;
    
    // Warmup
    for (size_t i = 0; i < 5; i++) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m);
    }
    
    // Benchmark
    double *times = malloc(iterations * sizeof(double));
    
    for (size_t i = 0; i < iterations; i++) {
        cudaDeviceSynchronize();
        double start = get_wall_time();
        
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m);
        
        cudaDeviceSynchronize();
        double end = get_wall_time();
        
        times[i] = (end - start) * 1e6; // microseconds
    }
    
    // Calculate statistics (same as vector addition)
    double sum = 0.0;
    result.min_time_us = times[0];
    result.max_time_us = times[0];
    
    for (size_t i = 0; i < iterations; i++) {
        sum += times[i];
        if (times[i] < result.min_time_us) result.min_time_us = times[i];
        if (times[i] > result.max_time_us) result.max_time_us = times[i];
    }
    
    result.mean_time_us = sum / iterations;
    
    double variance = 0.0;
    for (size_t i = 0; i < iterations; i++) {
        double diff = times[i] - result.mean_time_us;
        variance += diff * diff;
    }
    result.std_time_us = sqrt(variance / iterations);
    
    result.memory_mb = get_gpu_memory_usage();
    
    // Cleanup
    free(times);
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    
    return result;
}

static void print_usage(const char* program_name) {
    printf("Usage: %s --operation <op> --size1 <n> --size2 <n> [--size3 <n>] [--iterations <n>]\n", program_name);
    printf("Operations:\n");
    printf("  vector_add       - Variable-shape vector addition\n");
    printf("  matrix_multiply  - Dense matrix multiplication\n");
    printf("Options:\n");
    printf("  --size1 <n>      - First dimension size (default: 1024)\n");
    printf("  --size2 <n>      - Second dimension size (default: 1024)\n");
    printf("  --size3 <n>      - Third dimension size for matrix multiply (default: 1024)\n");
    printf("  --iterations <n> - Number of iterations (default: 100)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    char* operation = NULL;
    size_t size1 = 1024, size2 = 1024, size3 = 1024;
    size_t iterations = 100;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--operation") == 0 && i + 1 < argc) {
            operation = argv[++i];
        } else if (strcmp(argv[i], "--size1") == 0 && i + 1 < argc) {
            size1 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size2") == 0 && i + 1 < argc) {
            size2 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size3") == 0 && i + 1 < argc) {
            size3 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        }
    }
    
    if (!operation) {
        printf("{\"error\": \"Operation not specified\"}\n");
        return 1;
    }
    
    // Initialize CUDA
    cudaError_t cuda_err = cudaSetDevice(0);
    if (cuda_err != cudaSuccess) {
        printf("{\"error\": \"CUDA initialization failed\"}\n");
        return 1;
    }
    
    benchmark_result_t result;
    
    if (strcmp(operation, "vector_add") == 0) {
        result = benchmark_vector_addition(size1, size2, iterations);
        print_result_json("cublas_vector_add", "vector_addition", size1, size2, iterations, &result);
    } else if (strcmp(operation, "matrix_multiply") == 0) {
        result = benchmark_matrix_multiplication(size1, size2, size3, iterations);
        print_result_json("cublas_gemm", "matrix_multiplication", size1, size2, iterations, &result);
    } else {
        printf("{\"error\": \"Unknown operation: %s\"}\n", operation);
        return 1;
    }
    
    return 0;
}