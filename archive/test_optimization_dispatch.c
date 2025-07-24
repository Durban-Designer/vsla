/**
 * @file test_optimization_dispatch.c
 * @brief Test the unified optimization dispatch system
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

static void test_equal_shapes_optimization() {
    printf("Testing equal shapes optimization...\n");
    
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return;
    }
    
    // Create equal-shaped tensors for optimal vectorization
    uint64_t shape[] = {1000, 1000}; // Large enough to benefit from vectorization
    vsla_tensor_t* A = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* B = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* C = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!A || !B || !C) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill with test data
    size_t data_size;
    double* A_ptr = (double*)vsla_tensor_data(A, &data_size);
    double* B_ptr = (double*)vsla_tensor_data(B, &data_size);
    
    for (size_t i = 0; i < 1000 * 1000; i++) {
        A_ptr[i] = 1.0 + (double)i * 0.001;
        B_ptr[i] = 2.0 + (double)i * 0.002;
    }
    
    // Time the optimized addition
    clock_t start = clock();
    vsla_error_t result = vsla_add(ctx, C, A, B);
    clock_t end = clock();
    
    if (result != VSLA_SUCCESS) {
        printf("ERROR: Addition failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Equal shapes addition (1M elements): %.2f ms\n", time_ms);
    
    // Verify correctness
    double* C_ptr = (double*)vsla_tensor_data(C, &data_size);
    bool correct = true;
    for (int i = 0; i < 100; i++) { // Check first 100 elements
        double expected = A_ptr[i] + B_ptr[i];
        if (fabs(C_ptr[i] - expected) > 1e-12) {
            correct = false;
            printf("Mismatch at %d: got %.6f, expected %.6f\n", i, C_ptr[i], expected);
            break;
        }
    }
    
    printf("Equal shapes optimization test: %s\n", correct ? "PASSED" : "FAILED");
    
    vsla_tensor_free(A);
    vsla_tensor_free(B);
    vsla_tensor_free(C);
    vsla_cleanup(ctx);
}

static void test_scalar_broadcasting_optimization() {
    printf("\nTesting scalar broadcasting optimization...\n");
    
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return;
    }
    
    // Create tensor + scalar broadcasting case (same rank for VSLA compliance)
    uint64_t tensor_shape[] = {500, 500};
    uint64_t scalar_shape[] = {1, 1};     // Same rank as tensor but all dims = 1
    uint64_t output_shape[] = {500, 500};
    
    vsla_tensor_t* tensor = vsla_tensor_create(ctx, 2, tensor_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* scalar = vsla_tensor_create(ctx, 2, scalar_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 2, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!tensor || !scalar || !output) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill with test data
    size_t data_size;
    double* tensor_ptr = (double*)vsla_tensor_data(tensor, &data_size);
    double* scalar_ptr = (double*)vsla_tensor_data(scalar, &data_size);
    
    for (size_t i = 0; i < 500 * 500; i++) {
        tensor_ptr[i] = (double)i;
    }
    scalar_ptr[0] = 42.0;
    
    // Time the scalar broadcasting
    clock_t start = clock();
    vsla_error_t result = vsla_add(ctx, output, tensor, scalar);
    clock_t end = clock();
    
    if (result != VSLA_SUCCESS) {
        printf("ERROR: Scalar broadcasting failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Scalar broadcasting (250K elements): %.2f ms\n", time_ms);
    
    // Verify correctness
    double* output_ptr = (double*)vsla_tensor_data(output, &data_size);
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        double expected = tensor_ptr[i] + 42.0;
        if (fabs(output_ptr[i] - expected) > 1e-12) {
            correct = false;
            printf("Mismatch at %d: got %.6f, expected %.6f\n", i, output_ptr[i], expected);
            break;
        }
    }
    
    printf("Scalar broadcasting optimization test: %s\n", correct ? "PASSED" : "FAILED");
    
    vsla_tensor_free(tensor);
    vsla_tensor_free(scalar);
    vsla_tensor_free(output);
    vsla_cleanup(ctx);
}

static void test_2d_row_broadcasting_optimization() {
    printf("\nTesting 2D row broadcasting optimization...\n");
    
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return;
    }
    
    // Create matrix + row vector broadcasting: [N,M] + [1,M]
    uint64_t matrix_shape[] = {100, 1000};
    uint64_t row_shape[] = {1, 1000};
    uint64_t output_shape[] = {100, 1000};
    
    vsla_tensor_t* matrix = vsla_tensor_create(ctx, 2, matrix_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* row = vsla_tensor_create(ctx, 2, row_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 2, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!matrix || !row || !output) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill with test data
    size_t data_size;
    double* matrix_ptr = (double*)vsla_tensor_data(matrix, &data_size);
    double* row_ptr = (double*)vsla_tensor_data(row, &data_size);
    
    for (size_t i = 0; i < 100 * 1000; i++) {
        matrix_ptr[i] = (double)i * 0.1;
    }
    for (size_t j = 0; j < 1000; j++) {
        row_ptr[j] = (double)j * 0.01;
    }
    
    // Time the row broadcasting
    clock_t start = clock();
    vsla_error_t result = vsla_add(ctx, output, matrix, row);
    clock_t end = clock();
    
    if (result != VSLA_SUCCESS) {
        printf("ERROR: Row broadcasting failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("2D row broadcasting (100K elements): %.2f ms\n", time_ms);
    
    // Verify correctness - check a few elements
    double* output_ptr = (double*)vsla_tensor_data(output, &data_size);
    bool correct = true;
    
    // Check first row
    for (int j = 0; j < 10; j++) {
        double expected = matrix_ptr[j] + row_ptr[j];
        if (fabs(output_ptr[j] - expected) > 1e-12) {
            correct = false;
            printf("Mismatch at (0,%d): got %.6f, expected %.6f\n", j, output_ptr[j], expected);
            break;
        }
    }
    
    // Check second row 
    for (int j = 0; j < 10; j++) {
        int idx = 1000 + j; // Second row
        double expected = matrix_ptr[idx] + row_ptr[j];
        if (fabs(output_ptr[idx] - expected) > 1e-12) {
            correct = false;
            printf("Mismatch at (1,%d): got %.6f, expected %.6f\n", j, output_ptr[idx], expected);
            break;
        }
    }
    
    printf("2D row broadcasting optimization test: %s\n", correct ? "PASSED" : "FAILED");
    
    vsla_tensor_free(matrix);
    vsla_tensor_free(row);
    vsla_tensor_free(output);
    vsla_cleanup(ctx);
}

static void test_general_ambient_promotion() {
    printf("\nTesting general ambient promotion (complex broadcasting)...\n");
    
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return;
    }
    
    // Create complex broadcasting case that falls back to general ambient promotion
    uint64_t a_shape[] = {5, 7};
    uint64_t b_shape[] = {3, 7};
    uint64_t output_shape[] = {5, 7}; // max(5,3) = 5, max(7,7) = 7
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 2, a_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 2, b_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 2, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !output) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill with test data
    size_t data_size;
    double* a_ptr = (double*)vsla_tensor_data(a, &data_size);
    double* b_ptr = (double*)vsla_tensor_data(b, &data_size);
    
    for (size_t i = 0; i < 5 * 7; i++) {
        a_ptr[i] = (double)i + 1.0;
    }
    for (size_t i = 0; i < 3 * 7; i++) {
        b_ptr[i] = (double)i + 10.0;
    }
    
    // Time the general ambient promotion
    clock_t start = clock();
    vsla_error_t result = vsla_add(ctx, output, a, b);
    clock_t end = clock();
    
    if (result != VSLA_SUCCESS) {
        printf("ERROR: General ambient promotion failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("General ambient promotion (35 elements): %.2f ms\n", time_ms);
    
    // Verify correctness with VSLA ambient promotion semantics
    double* output_ptr = (double*)vsla_tensor_data(output, &data_size);
    bool correct = true;
    
    // Check overlapping region (first 3 rows)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 7; j++) {
            double a_val = a_ptr[i * 7 + j];
            double b_val = b_ptr[i * 7 + j];
            double expected = a_val + b_val;
            double actual = output_ptr[i * 7 + j];
            
            if (fabs(actual - expected) > 1e-12) {
                correct = false;
                printf("Mismatch at (%d,%d): got %.6f, expected %.6f (a=%.6f, b=%.6f)\n", 
                       i, j, actual, expected, a_val, b_val);
                break;
            }
        }
        if (!correct) break;
    }
    
    // Check extended region (rows 3-4, b is zero-extended)
    for (int i = 3; i < 5 && correct; i++) {
        for (int j = 0; j < 7; j++) {
            double a_val = a_ptr[i * 7 + j];
            double b_val = 0.0; // b is zero-extended
            double expected = a_val + b_val;
            double actual = output_ptr[i * 7 + j];
            
            if (fabs(actual - expected) > 1e-12) {
                correct = false;
                printf("Mismatch at (%d,%d): got %.6f, expected %.6f (a=%.6f, b=0.0 zero-extended)\n", 
                       i, j, actual, expected, a_val);
                break;
            }
        }
    }
    
    printf("General ambient promotion test: %s\n", correct ? "PASSED" : "FAILED");
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(output);
    vsla_cleanup(ctx);
}

int main() {
    printf("=== VSLA Optimization Dispatch Tests ===\n");
    
    test_equal_shapes_optimization();     // Should use vectorized equal-shapes kernel
    test_scalar_broadcasting_optimization(); // Should use vectorized scalar broadcasting
    test_2d_row_broadcasting_optimization(); // Should use specialized 2D row kernel
    test_general_ambient_promotion();        // Should fall back to general ambient promotion
    
    printf("\n=== Optimization Tests Complete ===\n");
    return 0;
}