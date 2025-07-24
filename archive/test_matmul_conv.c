/**
 * @file test_matmul_conv.c
 * @brief Simple test for matrix multiplication and convolution
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

static void test_matmul_2x2() {
    printf("Testing 2x2 matrix multiplication...\n");
    
    // Initialize VSLA
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
    
    // Create matrices: A[2,2] @ B[2,2] = C[2,2]
    uint64_t shape[] = {2, 2};
    vsla_tensor_t* A = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* B = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* C = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!A || !B || !C) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill matrices with test data
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    double B_data[] = {5.0, 6.0, 7.0, 8.0};
    
    size_t data_size;
    double* A_ptr = (double*)vsla_tensor_data(A, &data_size);
    double* B_ptr = (double*)vsla_tensor_data(B, &data_size);
    
    for (int i = 0; i < 4; i++) {
        A_ptr[i] = A_data[i];
        B_ptr[i] = B_data[i];
    }
    
    // Perform matrix multiplication
    vsla_error_t result = vsla_matmul(ctx, C, A, B);
    if (result != VSLA_SUCCESS) {
        printf("ERROR: Matrix multiplication failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    // Check result
    // Expected: C = [[19, 22], [43, 50]]
    double* C_ptr = (double*)vsla_tensor_data(C, &data_size);
    double expected[] = {19.0, 22.0, 43.0, 50.0};
    
    printf("Result matrix C:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("  %.1f", C_ptr[i * 2 + j]);
        }
        printf("\n");
    }
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(C_ptr[i] - expected[i]) > 1e-9) {
            correct = false;
            break;
        }
    }
    
    printf("Matrix multiplication test: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    vsla_tensor_free(A);
    vsla_tensor_free(B);
    vsla_tensor_free(C);
    vsla_cleanup(ctx);
}

static void test_conv_1d() {
    printf("\nTesting 1D convolution...\n");
    
    // Initialize VSLA
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
    
    // Create 1D signals for convolution
    uint64_t signal_shape[] = {4};  // Signal length 4
    uint64_t kernel_shape[] = {3};  // Kernel length 3
    uint64_t output_shape[] = {6};  // Output length = 4 + 3 - 1 = 6
    
    vsla_tensor_t* signal = vsla_tensor_create(ctx, 1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_tensor_create(ctx, 1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 1, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel || !output) {
        printf("ERROR: Failed to create tensors\n");
        vsla_cleanup(ctx);
        return;
    }
    
    // Fill with test data
    // Signal = [1, 2, 3, 4]
    // Kernel = [0.5, 1, 0.5]
    size_t data_size;
    double* signal_ptr = (double*)vsla_tensor_data(signal, &data_size);
    double* kernel_ptr = (double*)vsla_tensor_data(kernel, &data_size);
    
    signal_ptr[0] = 1.0; signal_ptr[1] = 2.0; signal_ptr[2] = 3.0; signal_ptr[3] = 4.0;
    kernel_ptr[0] = 0.5; kernel_ptr[1] = 1.0; kernel_ptr[2] = 0.5;
    
    // Perform convolution
    vsla_error_t result = vsla_conv(ctx, output, signal, kernel);
    if (result != VSLA_SUCCESS) {
        printf("ERROR: Convolution failed with error %d\n", result);
        vsla_cleanup(ctx);
        return;
    }
    
    // Check result
    double* output_ptr = (double*)vsla_tensor_data(output, &data_size);
    
    printf("Convolution result:\n");
    for (int i = 0; i < 6; i++) {
        printf("  %.2f", output_ptr[i]);
    }
    printf("\n");
    
    printf("1D convolution test: COMPLETED\n");
    
    // Cleanup
    vsla_tensor_free(signal);
    vsla_tensor_free(kernel);
    vsla_tensor_free(output);
    vsla_cleanup(ctx);
}

int main() {
    printf("=== VSLA Matrix Multiplication and Convolution Tests ===\n");
    
    test_matmul_2x2();
    test_conv_1d();
    
    printf("\n=== Tests Complete ===\n");
    return 0;
}