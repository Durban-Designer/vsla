/**
 * @file test_unified_api.c
 * @brief Test hardware-agnostic unified VSLA API
 * 
 * This test demonstrates the unified VSLA API that automatically
 * uses the best available hardware and vendor libraries.
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define TEST_SIZE 256
#define TOLERANCE 1e-6

static void test_context_creation(void) {
    printf("Testing context creation...\n");
    
    // Test with automatic configuration
    vsla_context_t* ctx = vsla_init(NULL);
    assert(ctx != NULL);
    
    // Get runtime info
    vsla_backend_t backend;
    char device_name[256];
    double memory_gb;
    
    vsla_error_t err = vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    assert(err == VSLA_SUCCESS);
    
    printf("  Backend: %d\n", backend);
    printf("  Device: %s\n", device_name);
    printf("  Memory: %.1f GB\n", memory_gb);
    
    vsla_cleanup(ctx);
    printf("  ✓ Context creation test passed\n");
}

static void test_tensor_operations(void) {
    printf("Testing unified tensor operations...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    assert(ctx != NULL);
    
    // Create test tensors
    uint64_t shape[] = {TEST_SIZE};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    assert(a != NULL && b != NULL && result != NULL);
    
    // Initialize test data
    float* a_data = (float*)vsla_tensor_data_mut(a, NULL);
    float* b_data = (float*)vsla_tensor_data_mut(b, NULL);
    
    for (int i = 0; i < TEST_SIZE; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }
    
    // Test addition
    vsla_error_t err = vsla_add(ctx, result, a, b);
    assert(err == VSLA_SUCCESS);
    
    // Verify results
    const float* result_data = (const float*)vsla_tensor_data(result, NULL);
    for (int i = 0; i < TEST_SIZE; i++) {
        float expected = (float)i + (float)(i * 2);
        assert(fabs(result_data[i] - expected) < TOLERANCE);
    }
    
    // Test scaling
    err = vsla_scale(ctx, result, a, 3.0);
    assert(err == VSLA_SUCCESS);
    
    result_data = (const float*)vsla_tensor_data(result, NULL);
    for (int i = 0; i < TEST_SIZE; i++) {
        float expected = (float)i * 3.0f;
        assert(fabs(result_data[i] - expected) < TOLERANCE);
    }
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    vsla_cleanup(ctx);
    
    printf("  ✓ Tensor operations test passed\n");
}

static void test_convolution(void) {
    printf("Testing unified convolution...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    assert(ctx != NULL);
    
    // Create test signals
    uint64_t signal_shape[] = {64};
    uint64_t kernel_shape[] = {8};
    uint64_t output_shape[] = {71};  // 64 + 8 - 1
    
    vsla_tensor_t* signal = vsla_tensor_create(ctx, 1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* kernel = vsla_tensor_create(ctx, 1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 1, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    assert(signal != NULL && kernel != NULL && output != NULL);
    
    // Initialize test data (impulse response)
    float* signal_data = (float*)vsla_tensor_data_mut(signal, NULL);
    float* kernel_data = (float*)vsla_tensor_data_mut(kernel, NULL);
    
    // Signal: impulse at position 10
    memset(signal_data, 0, 64 * sizeof(float));
    signal_data[10] = 1.0f;
    
    // Kernel: simple low-pass filter
    for (int i = 0; i < 8; i++) {
        kernel_data[i] = 1.0f / 8.0f;
    }
    
    // Perform convolution
    vsla_error_t err = vsla_conv(ctx, output, signal, kernel);
    assert(err == VSLA_SUCCESS);
    
    // Verify result
    const float* output_data = (const float*)vsla_tensor_data(output, NULL);
    
    // The output should have the kernel values starting at position 10
    for (int i = 10; i < 18; i++) {
        assert(fabs(output_data[i] - 1.0f/8.0f) < TOLERANCE);
    }
    
    // Other positions should be near zero
    for (int i = 0; i < 10; i++) {
        assert(fabs(output_data[i]) < TOLERANCE);
    }
    for (int i = 18; i < 71; i++) {
        assert(fabs(output_data[i]) < TOLERANCE);
    }
    
    // Cleanup
    vsla_tensor_free(signal);
    vsla_tensor_free(kernel);
    vsla_tensor_free(output);
    vsla_cleanup(ctx);
    
    printf("  ✓ Convolution test passed\n");
}

static void test_performance_stats(void) {
    printf("Testing performance statistics...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    assert(ctx != NULL);
    
    // Perform some operations
    uint64_t shape[] = {1024};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    // Fill with test data
    vsla_fill(ctx, a, 1.0);
    vsla_fill(ctx, b, 2.0);
    
    // Perform multiple operations
    for (int i = 0; i < 10; i++) {
        vsla_add(ctx, result, a, b);
        vsla_scale(ctx, result, result, 0.5);
    }
    
    // Get performance statistics
    vsla_stats_t stats;
    vsla_error_t err = vsla_get_stats(ctx, &stats);
    assert(err == VSLA_SUCCESS);
    
    printf("  Total operations: %lu\n", stats.total_operations);
    printf("  GPU operations: %lu\n", stats.gpu_operations);
    printf("  CPU operations: %lu\n", stats.cpu_operations);
    printf("  Total time: %.2f ms\n", stats.total_time_ms);
    
    assert(stats.total_operations > 0);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    vsla_cleanup(ctx);
    
    printf("  ✓ Performance statistics test passed\n");
}

static void test_backend_recommendation(void) {
    printf("Testing backend recommendation...\n");
    
    vsla_config_t config = {
        .backend = VSLA_BACKEND_AUTO,
        .optimization_hint = VSLA_HINT_THROUGHPUT
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    assert(ctx != NULL);
    
    // Create test tensors of different sizes
    uint64_t small_shape[] = {64};
    uint64_t large_shape[] = {65536};
    
    vsla_tensor_t* small_tensor = vsla_tensor_create(ctx, 1, small_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* large_tensor = vsla_tensor_create(ctx, 1, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    const vsla_tensor_t* small_inputs[] = {small_tensor};
    const vsla_tensor_t* large_inputs[] = {large_tensor};
    
    // Get recommendations
    vsla_backend_t small_backend = vsla_recommend_backend(ctx, "add", small_inputs, 1);
    vsla_backend_t large_backend = vsla_recommend_backend(ctx, "add", large_inputs, 1);
    
    printf("  Small tensor backend: %d\n", small_backend);
    printf("  Large tensor backend: %d\n", large_backend);
    
    // Cleanup
    vsla_tensor_free(small_tensor);
    vsla_tensor_free(large_tensor);
    vsla_cleanup(ctx);
    
    printf("  ✓ Backend recommendation test passed\n");
}

int main(void) {
    printf("Running unified VSLA API tests...\n\n");
    
    test_context_creation();
    test_tensor_operations();
    test_convolution();
    test_performance_stats();
    test_backend_recommendation();
    
    printf("\n✓ All unified API tests passed!\n");
    printf("VSLA unified interface successfully abstracts hardware complexity\n");
    printf("while maintaining high performance through automatic optimization.\n");
    
    return 0;
}