#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vsla/vsla.h"

double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

void benchmark_vector_addition(size_t size1, size_t size2, int iterations) {
    printf("=== Vector Addition Benchmark ===\n");
    printf("Size1: %zu, Size2: %zu, Iterations: %d\n\n", size1, size2, iterations);
    
    // Create test tensors
    uint64_t shape1[] = {size1};
    uint64_t shape2[] = {size2};
    uint64_t max_size = (size1 > size2) ? size1 : size2;
    uint64_t result_shape[] = {max_size};
    
    vsla_tensor_t* a = vsla_new(1, shape1, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_new(1, shape2, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (size_t i = 0; i < size1; i++) {
        a_data[i] = (float)i;
    }
    for (size_t i = 0; i < size2; i++) {
        b_data[i] = (float)i;
    }
    
    // CPU Benchmark
    printf("CPU Benchmark:\n");
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_add(result, a, b);
    }
    
    double cpu_start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        vsla_add(result, a, b);
    }
    double cpu_end = get_time_us();
    double cpu_time_per_iter = (cpu_end - cpu_start) / iterations;
    
    printf("  Time per iteration: %.3f μs\n", cpu_time_per_iter);
    printf("  Total elements: %zu\n", max_size);
    printf("  Throughput: %.2f MOPS\n\n", max_size / cpu_time_per_iter);
    
    // GPU Benchmark (if available)
    if (vsla_has_gpu() && vsla_gpu_is_available()) {
        printf("GPU Benchmark:\n");
        
        vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
        if (ctx) {
            // Create GPU tensors
            vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(a, ctx);
            vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(b, ctx);
            vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(result, ctx);
            
            if (gpu_a && gpu_b && gpu_result) {
                // Allocate GPU memory
                if (vsla_gpu_tensor_alloc(gpu_a, ctx) == VSLA_SUCCESS &&
                    vsla_gpu_tensor_alloc(gpu_b, ctx) == VSLA_SUCCESS &&
                    vsla_gpu_tensor_alloc(gpu_result, ctx) == VSLA_SUCCESS) {
                    
                    // Copy data to GPU
                    vsla_gpu_tensor_copy_to_gpu(gpu_a, a->data, false);
                    vsla_gpu_tensor_copy_to_gpu(gpu_b, b->data, false);
                    
                    // Warmup
                    for (int i = 0; i < 5; i++) {
                        vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
                        vsla_gpu_tensor_sync(gpu_result);
                    }
                    
                    double gpu_start = get_time_us();
                    for (int i = 0; i < iterations; i++) {
                        vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
                        vsla_gpu_tensor_sync(gpu_result);
                    }
                    double gpu_end = get_time_us();
                    double gpu_time_per_iter = (gpu_end - gpu_start) / iterations;
                    
                    printf("  Time per iteration: %.3f μs\n", gpu_time_per_iter);
                    printf("  Total elements: %zu\n", max_size);
                    printf("  Throughput: %.2f MOPS\n", max_size / gpu_time_per_iter);
                    printf("  Speedup: %.2fx\n\n", cpu_time_per_iter / gpu_time_per_iter);
                } else {
                    printf("  GPU memory allocation failed\n\n");
                }
                
                vsla_gpu_tensor_free(gpu_a);
                vsla_gpu_tensor_free(gpu_b);
                vsla_gpu_tensor_free(gpu_result);
            } else {
                printf("  GPU tensor creation failed\n\n");
            }
            
            vsla_gpu_destroy(ctx);
        } else {
            printf("  GPU context creation failed\n\n");
        }
    } else {
        printf("GPU not available\n\n");
    }
    
    // Cleanup
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

void benchmark_matrix_multiplication(size_t m, size_t n, size_t k, int iterations) {
    printf("=== Matrix Multiplication Benchmark ===\n");
    printf("Matrix A: %zux%zu, Matrix B: %zux%zu, Iterations: %d\n\n", m, k, k, n, iterations);
    
    // Create test matrices
    uint64_t shape_a[] = {m, k};
    uint64_t shape_b[] = {k, n};
    uint64_t shape_result[] = {m, n};
    
    vsla_tensor_t* a = vsla_new(2, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_new(2, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(2, shape_result, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU Benchmark
    printf("CPU Benchmark:\n");
    
    // Warmup - skip CPU matrix multiplication for now since VSLA uses different approach
    printf("  CPU matrix multiplication not implemented for dense matrices\n");
    printf("  (VSLA uses Model A/B with convolution/Kronecker products)\n\n");
    
    // GPU Benchmark (if available)
    if (vsla_has_gpu() && vsla_gpu_is_available()) {
        printf("GPU Benchmark:\n");
        
        vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
        if (ctx) {
            // Create GPU tensors
            vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(a, ctx);
            vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(b, ctx);
            vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(result, ctx);
            
            if (gpu_a && gpu_b && gpu_result) {
                // Allocate GPU memory
                if (vsla_gpu_tensor_alloc(gpu_a, ctx) == VSLA_SUCCESS &&
                    vsla_gpu_tensor_alloc(gpu_b, ctx) == VSLA_SUCCESS &&
                    vsla_gpu_tensor_alloc(gpu_result, ctx) == VSLA_SUCCESS) {
                    
                    // Copy data to GPU
                    vsla_gpu_tensor_copy_to_gpu(gpu_a, a->data, false);
                    vsla_gpu_tensor_copy_to_gpu(gpu_b, b->data, false);
                    
                    // Warmup
                    for (int i = 0; i < 3; i++) {
                        vsla_gpu_matmul(gpu_result, gpu_a, gpu_b, ctx);
                        vsla_gpu_tensor_sync(gpu_result);
                    }
                    
                    double gpu_start = get_time_us();
                    for (int i = 0; i < iterations; i++) {
                        vsla_gpu_matmul(gpu_result, gpu_a, gpu_b, ctx);
                        vsla_gpu_tensor_sync(gpu_result);
                    }
                    double gpu_end = get_time_us();
                    double gpu_time_per_iter = (gpu_end - gpu_start) / iterations;
                    
                    double flops = 2.0 * m * n * k; // 2 * m * n * k FLOPs per matrix multiplication
                    printf("  Time per iteration: %.3f μs\n", gpu_time_per_iter);
                    printf("  GFLOPS: %.2f\n", flops / (gpu_time_per_iter * 1000));
                    printf("  GPU-only timing (no CPU comparison)\n\n");
                } else {
                    printf("  GPU memory allocation failed\n\n");
                }
                
                vsla_gpu_tensor_free(gpu_a);
                vsla_gpu_tensor_free(gpu_b);
                vsla_gpu_tensor_free(gpu_result);
            } else {
                printf("  GPU tensor creation failed\n\n");
            }
            
            vsla_gpu_destroy(ctx);
        } else {
            printf("  GPU context creation failed\n\n");
        }
    } else {
        printf("GPU not available\n\n");
    }
    
    // Cleanup
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

int main() {
    printf("VSLA GPU vs CPU Benchmark Suite\n");
    printf("================================\n\n");
    
    // Initialize VSLA
    vsla_init();
    
    // Check GPU availability
    if (vsla_has_gpu()) {
        printf("GPU support: Available\n");
        if (vsla_gpu_is_available()) {
            printf("GPU hardware: Detected\n");
        } else {
            printf("GPU hardware: Not detected\n");
        }
    } else {
        printf("GPU support: Not compiled\n");
    }
    printf("\n");
    
    // Vector addition benchmarks
    benchmark_vector_addition(1000, 1500, 100);
    benchmark_vector_addition(10000, 15000, 50);
    benchmark_vector_addition(100000, 150000, 10);
    
    // Matrix multiplication benchmarks
    benchmark_matrix_multiplication(64, 64, 64, 50);
    benchmark_matrix_multiplication(128, 128, 128, 20);
    benchmark_matrix_multiplication(256, 256, 256, 10);
    
    vsla_cleanup();
    return 0;
}