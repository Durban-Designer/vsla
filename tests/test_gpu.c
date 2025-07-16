/**
 * @file test_gpu.c
 * @brief Comprehensive GPU acceleration tests for VSLA
 * 
 * This file contains extensible tests for all GPU functionality.
 * Tests are designed to be robust across different optimization levels
 * and implementation changes.
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <math.h>

// Test configuration
#define GPU_TEST_TOLERANCE_F32 1e-6f
#define GPU_TEST_TOLERANCE_F64 1e-12

// Test helper macros
#define ASSERT_GPU_SUCCESS(expr) \
    do { \
        vsla_error_t __err = (expr); \
        if (__err != VSLA_SUCCESS) { \
            printf("\n    GPU assertion failed: %s returned %d\n", #expr, __err); \
            return 0; \
        } \
    } while(0)

#define ASSERT_GPU_NOT_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf("\n    GPU assertion failed: %s is NULL\n", #ptr); \
            return 0; \
        } \
    } while(0)

#define ASSERT_GPU_NULL(ptr) \
    do { \
        if ((ptr) != NULL) { \
            printf("\n    GPU assertion failed: %s is not NULL\n", #ptr); \
            return 0; \
        } \
    } while(0)

// Helper function to compare floating point values with tolerance
static int gpu_values_close_f32(float a, float b, float tolerance) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && ((a > 0) == (b > 0))) return 1;
    return fabsf(a - b) <= tolerance;
}

static int gpu_values_close_f64(double a, double b, double tolerance) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && ((a > 0) == (b > 0))) return 1;
    return fabs(a - b) <= tolerance;
}

// Helper function to create test tensor with known values
static vsla_tensor_t* create_test_tensor_f32(uint8_t rank, uint64_t* shape, float start_val, float increment) {
    vsla_tensor_t* tensor = vsla_new(rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (!tensor) return NULL;
    
    size_t total_elements = 1;
    for (uint8_t i = 0; i < rank; i++) {
        total_elements *= shape[i];
    }
    
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < total_elements; i++) {
        data[i] = start_val + i * increment;
    }
    
    return tensor;
}

static vsla_tensor_t* create_test_tensor_f64(uint8_t rank, uint64_t* shape, double start_val, double increment) {
    vsla_tensor_t* tensor = vsla_new(rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!tensor) return NULL;
    
    size_t total_elements = 1;
    for (uint8_t i = 0; i < rank; i++) {
        total_elements *= shape[i];
    }
    
    double* data = (double*)tensor->data;
    for (size_t i = 0; i < total_elements; i++) {
        data[i] = start_val + i * increment;
    }
    
    return tensor;
}

// Test GPU device detection and availability
static int test_gpu_device_detection(void) {
    printf("    Testing GPU device detection...\n");
    
    // Test basic availability
    int has_gpu = vsla_has_gpu();
    printf("      GPU support compiled: %s\n", has_gpu ? "YES" : "NO");
    
    if (!has_gpu) {
        printf("      Skipping GPU tests - no GPU support compiled\n");
        return 1; // This is not a failure, just no GPU support
    }
    
    bool available = vsla_gpu_is_available();
    printf("      GPU hardware available: %s\n", available ? "YES" : "NO");
    
    if (!available) {
        printf("      Skipping GPU tests - no GPU hardware available\n");
        return 1; // This is not a failure, just no GPU
    }
    
    // Test device info retrieval
    char device_name[256];
    double memory_gb;
    vsla_error_t err = vsla_gpu_get_device_info(0, device_name, &memory_gb);
    ASSERT_GPU_SUCCESS(err);
    
    printf("      Device 0: %s (%.2f GB)\n", device_name, memory_gb);
    ASSERT_TRUE(strlen(device_name) > 0);
    ASSERT_TRUE(memory_gb > 0.0);
    
    // Test invalid device ID
    err = vsla_gpu_get_device_info(999, device_name, &memory_gb);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    // Test NULL parameters
    err = vsla_gpu_get_device_info(0, NULL, &memory_gb);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    err = vsla_gpu_get_device_info(0, device_name, NULL);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    return 1;
}

// Test GPU context management
static int test_gpu_context_management(void) {
    printf("    Testing GPU context management...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    // Test context creation and destruction
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1); // Auto-select device
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test context destruction
    vsla_gpu_destroy(ctx);
    
    // Test explicit device selection
    ctx = vsla_gpu_init(0);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test memory usage query
    size_t used_mb, total_mb;
    vsla_error_t err = vsla_gpu_get_memory_usage(ctx, &used_mb, &total_mb);
    ASSERT_GPU_SUCCESS(err);
    
    printf("      GPU memory: %zu MB used / %zu MB total\n", used_mb, total_mb);
    ASSERT_TRUE(total_mb > 0);
    
    // Test launch configuration
    size_t block_size, grid_size;
    err = vsla_gpu_get_launch_config(1000, &block_size, &grid_size);
    ASSERT_GPU_SUCCESS(err);
    ASSERT_TRUE(block_size > 0);
    ASSERT_TRUE(grid_size > 0);
    
    printf("      Launch config for 1000 elements: block=%zu, grid=%zu\n", block_size, grid_size);
    
    vsla_gpu_destroy(ctx);
    
    // Test invalid device ID
    ctx = vsla_gpu_init(999);
    ASSERT_GPU_NULL(ctx);
    
    // Test NULL destruction (should not crash)
    vsla_gpu_destroy(NULL);
    
    return 1;
}

// Test GPU tensor memory management
static int test_gpu_tensor_memory(void) {
    printf("    Testing GPU tensor memory management...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Create test CPU tensor
    uint64_t shape[] = {10, 20};
    vsla_tensor_t* cpu_tensor = create_test_tensor_f32(2, shape, 1.0f, 0.5f);
    ASSERT_NOT_NULL(cpu_tensor);
    
    // Test GPU tensor creation from CPU tensor
    vsla_gpu_tensor_t* gpu_tensor = vsla_gpu_tensor_from_cpu(cpu_tensor, ctx);
    ASSERT_GPU_NOT_NULL(gpu_tensor);
    
    // Verify tensor properties
    ASSERT_TRUE(gpu_tensor->rank == 2);
    ASSERT_TRUE(gpu_tensor->shape[0] == 10);
    ASSERT_TRUE(gpu_tensor->shape[1] == 20);
    ASSERT_TRUE(gpu_tensor->dtype == VSLA_DTYPE_F32);
    
    // Test GPU memory allocation
    vsla_error_t err = vsla_gpu_tensor_alloc(gpu_tensor, ctx);
    ASSERT_GPU_SUCCESS(err);
    ASSERT_GPU_NOT_NULL(gpu_tensor->gpu_data);
    
    // Test data copy to GPU
    err = vsla_gpu_tensor_copy_to_gpu(gpu_tensor, cpu_tensor->data, false);
    ASSERT_GPU_SUCCESS(err);
    
    // Test synchronous data copy back to CPU
    float* cpu_data = (float*)malloc(10 * 20 * sizeof(float));
    ASSERT_NOT_NULL(cpu_data);
    
    err = vsla_gpu_tensor_copy_to_cpu(gpu_tensor, cpu_data, false);
    ASSERT_GPU_SUCCESS(err);
    
    // Verify data integrity
    float* original_data = (float*)cpu_tensor->data;
    for (size_t i = 0; i < 200; i++) {
        ASSERT_TRUE(gpu_values_close_f32(cpu_data[i], original_data[i], GPU_TEST_TOLERANCE_F32));
    }
    
    // Test asynchronous operations
    err = vsla_gpu_tensor_copy_to_gpu(gpu_tensor, cpu_tensor->data, true);
    ASSERT_GPU_SUCCESS(err);
    
    err = vsla_gpu_tensor_sync(gpu_tensor);
    ASSERT_GPU_SUCCESS(err);
    
    // Test GPU tensor to CPU tensor conversion
    vsla_tensor_t* reconstructed_tensor = vsla_gpu_tensor_to_cpu(gpu_tensor);
    ASSERT_NOT_NULL(reconstructed_tensor);
    
    // Verify reconstructed tensor
    float* reconstructed_data = (float*)reconstructed_tensor->data;
    for (size_t i = 0; i < 200; i++) {
        ASSERT_TRUE(gpu_values_close_f32(reconstructed_data[i], original_data[i], GPU_TEST_TOLERANCE_F32));
    }
    
    // Cleanup
    free(cpu_data);
    vsla_gpu_tensor_free(gpu_tensor);
    vsla_free(cpu_tensor);
    vsla_free(reconstructed_tensor);
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Test GPU tensor addition
static int test_gpu_tensor_addition(void) {
    printf("    Testing GPU tensor addition...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test F32 addition
    uint64_t shape[] = {5, 8};
    vsla_tensor_t* cpu_a = create_test_tensor_f32(2, shape, 1.0f, 0.1f);
    vsla_tensor_t* cpu_b = create_test_tensor_f32(2, shape, 2.0f, 0.2f);
    vsla_tensor_t* cpu_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    ASSERT_NOT_NULL(cpu_a);
    ASSERT_NOT_NULL(cpu_b);
    ASSERT_NOT_NULL(cpu_result);
    
    // Create GPU tensors
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(cpu_a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(cpu_b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(cpu_result, ctx);
    
    ASSERT_GPU_NOT_NULL(gpu_a);
    ASSERT_GPU_NOT_NULL(gpu_b);
    ASSERT_GPU_NOT_NULL(gpu_result);
    
    // Allocate GPU memory and copy data
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_a, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_b, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_result, ctx));
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_a, cpu_a->data, false));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_b, cpu_b->data, false));
    
    // Perform GPU addition
    vsla_error_t err = vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
    ASSERT_GPU_SUCCESS(err);
    
    // Copy result back to CPU
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_cpu(gpu_result, cpu_result->data, false));
    
    // Verify results against CPU computation
    float* a_data = (float*)cpu_a->data;
    float* b_data = (float*)cpu_b->data;
    float* result_data = (float*)cpu_result->data;
    
    for (size_t i = 0; i < 40; i++) {
        float expected = a_data[i] + b_data[i];
        ASSERT_TRUE(gpu_values_close_f32(result_data[i], expected, GPU_TEST_TOLERANCE_F32));
    }
    
    // Test F64 addition
    vsla_tensor_t* cpu_a_f64 = create_test_tensor_f64(2, shape, 1.0, 0.1);
    vsla_tensor_t* cpu_b_f64 = create_test_tensor_f64(2, shape, 2.0, 0.2);
    vsla_tensor_t* cpu_result_f64 = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(cpu_a_f64);
    ASSERT_NOT_NULL(cpu_b_f64);
    ASSERT_NOT_NULL(cpu_result_f64);
    
    vsla_gpu_tensor_t* gpu_a_f64 = vsla_gpu_tensor_from_cpu(cpu_a_f64, ctx);
    vsla_gpu_tensor_t* gpu_b_f64 = vsla_gpu_tensor_from_cpu(cpu_b_f64, ctx);
    vsla_gpu_tensor_t* gpu_result_f64 = vsla_gpu_tensor_from_cpu(cpu_result_f64, ctx);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_a_f64, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_b_f64, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_result_f64, ctx));
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_a_f64, cpu_a_f64->data, false));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_b_f64, cpu_b_f64->data, false));
    
    err = vsla_gpu_add(gpu_result_f64, gpu_a_f64, gpu_b_f64, ctx);
    ASSERT_GPU_SUCCESS(err);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_cpu(gpu_result_f64, cpu_result_f64->data, false));
    
    // Verify F64 results
    double* a_data_f64 = (double*)cpu_a_f64->data;
    double* b_data_f64 = (double*)cpu_b_f64->data;
    double* result_data_f64 = (double*)cpu_result_f64->data;
    
    for (size_t i = 0; i < 40; i++) {
        double expected = a_data_f64[i] + b_data_f64[i];
        ASSERT_TRUE(gpu_values_close_f64(result_data_f64[i], expected, GPU_TEST_TOLERANCE_F64));
    }
    
    // Cleanup
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    vsla_gpu_tensor_free(gpu_a_f64);
    vsla_gpu_tensor_free(gpu_b_f64);
    vsla_gpu_tensor_free(gpu_result_f64);
    
    vsla_free(cpu_a);
    vsla_free(cpu_b);
    vsla_free(cpu_result);
    vsla_free(cpu_a_f64);
    vsla_free(cpu_b_f64);
    vsla_free(cpu_result_f64);
    
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Test GPU tensor scaling
static int test_gpu_tensor_scaling(void) {
    printf("    Testing GPU tensor scaling...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test F32 scaling
    uint64_t shape[] = {6, 4};
    vsla_tensor_t* cpu_tensor = create_test_tensor_f32(2, shape, 1.0f, 0.5f);
    vsla_tensor_t* cpu_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    ASSERT_NOT_NULL(cpu_tensor);
    ASSERT_NOT_NULL(cpu_result);
    
    vsla_gpu_tensor_t* gpu_tensor = vsla_gpu_tensor_from_cpu(cpu_tensor, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(cpu_result, ctx);
    
    ASSERT_GPU_NOT_NULL(gpu_tensor);
    ASSERT_GPU_NOT_NULL(gpu_result);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_tensor, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_result, ctx));
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_tensor, cpu_tensor->data, false));
    
    // Test various scale factors
    double scale_factors[] = {0.5, 2.0, -1.0, 0.0, 1.0, 3.14159};
    size_t num_scales = sizeof(scale_factors) / sizeof(scale_factors[0]);
    
    for (size_t s = 0; s < num_scales; s++) {
        double scale = scale_factors[s];
        
        vsla_error_t err = vsla_gpu_scale(gpu_result, gpu_tensor, scale, ctx);
        ASSERT_GPU_SUCCESS(err);
        
        ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_cpu(gpu_result, cpu_result->data, false));
        
        // Verify results
        float* tensor_data = (float*)cpu_tensor->data;
        float* result_data = (float*)cpu_result->data;
        
        for (size_t i = 0; i < 24; i++) {
            float expected = tensor_data[i] * (float)scale;
            ASSERT_TRUE(gpu_values_close_f32(result_data[i], expected, GPU_TEST_TOLERANCE_F32));
        }
    }
    
    // Cleanup
    vsla_gpu_tensor_free(gpu_tensor);
    vsla_gpu_tensor_free(gpu_result);
    vsla_free(cpu_tensor);
    vsla_free(cpu_result);
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Test GPU matrix multiplication
static int test_gpu_matrix_multiplication(void) {
    printf("    Testing GPU matrix multiplication...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test small matrix multiplication: 3x4 * 4x5 = 3x5
    uint64_t shape_a[] = {3, 4};
    uint64_t shape_b[] = {4, 5};
    uint64_t shape_result[] = {3, 5};
    
    vsla_tensor_t* cpu_a = create_test_tensor_f32(2, shape_a, 1.0f, 0.1f);
    vsla_tensor_t* cpu_b = create_test_tensor_f32(2, shape_b, 0.5f, 0.2f);
    vsla_tensor_t* cpu_result = vsla_new(2, shape_result, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    ASSERT_NOT_NULL(cpu_a);
    ASSERT_NOT_NULL(cpu_b);
    ASSERT_NOT_NULL(cpu_result);
    
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(cpu_a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(cpu_b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(cpu_result, ctx);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_a, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_b, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_result, ctx));
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_a, cpu_a->data, false));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_b, cpu_b->data, false));
    
    // Perform GPU matrix multiplication
    vsla_error_t err = vsla_gpu_matmul(gpu_result, gpu_a, gpu_b, ctx);
    ASSERT_GPU_SUCCESS(err);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_cpu(gpu_result, cpu_result->data, false));
    
    // Verify results using manual computation
    float* a_data = (float*)cpu_a->data;
    float* b_data = (float*)cpu_b->data;
    float* result_data = (float*)cpu_result->data;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            float expected = 0.0f;
            for (int k = 0; k < 4; k++) {
                expected += a_data[i * 4 + k] * b_data[k * 5 + j];
            }
            float actual = result_data[i * 5 + j];
            ASSERT_TRUE(gpu_values_close_f32(actual, expected, GPU_TEST_TOLERANCE_F32));
        }
    }
    
    // Cleanup
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    vsla_free(cpu_a);
    vsla_free(cpu_b);
    vsla_free(cpu_result);
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Test GPU error handling and edge cases
static int test_gpu_error_handling(void) {
    printf("    Testing GPU error handling...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test NULL parameter handling
    vsla_error_t err = vsla_gpu_add(NULL, NULL, NULL, ctx);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    err = vsla_gpu_scale(NULL, NULL, 1.0, ctx);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    err = vsla_gpu_matmul(NULL, NULL, NULL, ctx);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    // Test operations on unallocated tensors
    uint64_t shape[] = {2, 2};
    vsla_tensor_t* cpu_tensor = create_test_tensor_f32(2, shape, 1.0f, 0.1f);
    vsla_gpu_tensor_t* gpu_tensor = vsla_gpu_tensor_from_cpu(cpu_tensor, ctx);
    
    // Don't allocate GPU memory - should fail
    err = vsla_gpu_tensor_copy_to_gpu(gpu_tensor, cpu_tensor->data, false);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_STATE);
    
    // Test operations with rank less than 2 for matrix multiplication
    uint64_t shape_1d[] = {5};
    vsla_tensor_t* cpu_1d = create_test_tensor_f32(1, shape_1d, 1.0f, 0.1f);
    vsla_gpu_tensor_t* gpu_1d = vsla_gpu_tensor_from_cpu(cpu_1d, ctx);
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_1d, ctx));
    
    err = vsla_gpu_matmul(gpu_tensor, gpu_1d, gpu_1d, ctx);
    ASSERT_TRUE(err == VSLA_ERROR_INVALID_ARGUMENT);
    
    // Test synchronization
    err = vsla_gpu_tensor_sync(gpu_1d);
    ASSERT_GPU_SUCCESS(err);
    
    err = vsla_gpu_tensor_sync(NULL); // Should still work
    ASSERT_GPU_SUCCESS(err);
    
    // Cleanup
    vsla_gpu_tensor_free(gpu_tensor);
    vsla_gpu_tensor_free(gpu_1d);
    vsla_free(cpu_tensor);
    vsla_free(cpu_1d);
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Test GPU vs CPU result consistency
static int test_gpu_cpu_consistency(void) {
    printf("    Testing GPU vs CPU result consistency...\n");
    
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("      Skipping - no GPU available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    ASSERT_GPU_NOT_NULL(ctx);
    
    // Test tensor addition consistency
    uint64_t shape[] = {8, 12};
    vsla_tensor_t* cpu_a = create_test_tensor_f32(2, shape, 1.0f, 0.1f);
    vsla_tensor_t* cpu_b = create_test_tensor_f32(2, shape, 2.0f, 0.2f);
    vsla_tensor_t* cpu_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* gpu_computed_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    // CPU computation
    vsla_error_t err = vsla_add(cpu_result, cpu_a, cpu_b);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    // GPU computation
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(cpu_a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(cpu_b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(gpu_computed_result, ctx);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_a, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_b, ctx));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_alloc(gpu_result, ctx));
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_a, cpu_a->data, false));
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_gpu(gpu_b, cpu_b->data, false));
    
    err = vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
    ASSERT_GPU_SUCCESS(err);
    
    ASSERT_GPU_SUCCESS(vsla_gpu_tensor_copy_to_cpu(gpu_result, gpu_computed_result->data, false));
    
    // Compare results
    float* cpu_data = (float*)cpu_result->data;
    float* gpu_data = (float*)gpu_computed_result->data;
    
    for (size_t i = 0; i < 96; i++) {
        ASSERT_TRUE(gpu_values_close_f32(cpu_data[i], gpu_data[i], GPU_TEST_TOLERANCE_F32));
    }
    
    // Cleanup
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    vsla_free(cpu_a);
    vsla_free(cpu_b);
    vsla_free(cpu_result);
    vsla_free(gpu_computed_result);
    vsla_gpu_destroy(ctx);
    
    return 1;
}

// Register GPU test suite
void register_gpu_tests(void) {
    printf("Running GPU tests:\n");
    
    // Check if GPU is available first
    if (!vsla_has_gpu()) {
        printf("  GPU support not compiled - skipping all GPU tests\n");
        return;
    }
    
    if (!vsla_gpu_is_available()) {
        printf("  GPU hardware not available - skipping all GPU tests\n");
        return;
    }
    
    // Run all GPU tests
    RUN_TEST(test_gpu_device_detection);
    RUN_TEST(test_gpu_context_management);
    RUN_TEST(test_gpu_tensor_memory);
    RUN_TEST(test_gpu_tensor_addition);
    RUN_TEST(test_gpu_tensor_scaling);
    RUN_TEST(test_gpu_matrix_multiplication);
    RUN_TEST(test_gpu_error_handling);
    RUN_TEST(test_gpu_cpu_consistency);
}

