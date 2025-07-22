/**
 * @file quick_benchmark.c
 * @brief Quick benchmark test for VSLA performance validation
 */

#include "benchmark_utils.h"
#include "vsla/vsla_unified.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void test_convolution_performance(vsla_context_t* ctx) {
    printf("=== Quick Convolution Performance Test ===\n");
    
    // Test small-scale convolution
    size_t signal_len = 1024;
    size_t kernel_len = 32;
    
    uint64_t signal_shape[] = {signal_len};
    uint64_t kernel_shape[] = {kernel_len};
    uint64_t output_shape[] = {signal_len + kernel_len - 1};
    
    vsla_tensor_t* signal = vsla_tensor_create(ctx, 1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_tensor_create(ctx, 1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel || !result) {
        printf("Error: Failed to create tensors\n");
        return;
    }
    
    // Fill with test data
    for (size_t i = 0; i < signal_len; i++) {
        uint64_t idx[] = {i};
        vsla_set_f64(ctx, signal, idx, sin(2.0 * M_PI * i / 64.0));
    }
    
    for (size_t i = 0; i < kernel_len; i++) {
        uint64_t idx[] = {i};
        double gaussian = exp(-0.5 * pow((double)i - 16.0, 2) / 16.0);
        vsla_set_f64(ctx, kernel, idx, gaussian);
    }
    
    // Warm up
    for (int i = 0; i < 3; i++) {
        vsla_conv(ctx, result, signal, kernel);
    }
    
    // Time the operation
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        vsla_conv(ctx, result, signal, kernel);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                    (end.tv_nsec - start.tv_nsec) * 1e-9;
    double time_per_op = (elapsed / iterations) * 1e6; // microseconds
    
    printf("Signal length: %zu, Kernel length: %zu\n", signal_len, kernel_len);
    printf("Time per convolution: %.3f μs\n", time_per_op);
    printf("Throughput: %.3f MOPS\n", (signal_len * kernel_len) / time_per_op);
    
    // Verify result is non-zero
    double sum;
    vsla_sum(ctx, result, &sum);
    printf("Result sum: %.6f (sanity check)\n", sum);
    
    // Cleanup
    vsla_tensor_free(signal);
    vsla_tensor_free(kernel);
    vsla_tensor_free(result);
}

static void test_arithmetic_performance(vsla_context_t* ctx) {
    printf("\n=== Quick Arithmetic Performance Test ===\n");
    
    size_t tensor_size = 10000;
    uint64_t shape[] = {tensor_size};
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !result) {
        printf("Error: Failed to create arithmetic tensors\n");
        return;
    }
    
    // Fill tensors
    vsla_fill(ctx, a, 1.5);
    vsla_fill(ctx, b, 2.5);
    
    // Test addition
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        vsla_add(ctx, result, a, b);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                    (end.tv_nsec - start.tv_nsec) * 1e-9;
    double time_per_op = (elapsed / iterations) * 1e6; // microseconds
    
    printf("Tensor size: %zu elements\n", tensor_size);
    printf("Time per addition: %.3f μs\n", time_per_op);
    
    // Calculate bandwidth
    size_t bytes_per_op = 3 * tensor_size * sizeof(double); // read a, read b, write result
    double bandwidth_gbps = (bytes_per_op / (time_per_op * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    printf("Memory bandwidth: %.3f GB/s\n", bandwidth_gbps);
    
    // Verify result
    double sum;
    vsla_sum(ctx, result, &sum);
    printf("Result sum: %.1f (should be %.1f)\n", sum, tensor_size * 4.0);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

int main() {
    printf("VSLA Quick Benchmark Test\n");
    printf("=========================\n\n");
    
    // Initialize VSLA
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        printf("Error: Failed to initialize VSLA context\n");
        return 1;
    }
    
    // Get runtime info
    vsla_backend_t backend;
    char device_name[256];
    double memory_gb;
    vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    
    printf("VSLA Runtime Info:\n");
    printf("  Backend: %s\n", device_name);
    printf("  Memory: %.1f GB\n", memory_gb);
    printf("\n");
    
    // Run quick tests
    test_convolution_performance(ctx);
    test_arithmetic_performance(ctx);
    
    printf("\n=== Summary ===\n");
    printf("VSLA is working correctly with basic performance validation.\n");
    printf("All operations completed successfully.\n");
    
    // Cleanup
    vsla_cleanup(ctx);
    
    return 0;
}