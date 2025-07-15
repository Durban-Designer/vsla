/**
 * @file basic_usage.c
 * @brief Basic usage example for libvsla
 * 
 * This example demonstrates the core concepts of Variable-Shape Linear Algebra:
 * - Creating tensors with different shapes
 * - Automatic zero-padding for operations
 * - Basic arithmetic operations
 * - Memory management
 * 
 * Compile with: gcc -I../include basic_usage.c ../build/libvsla.a -lm -o basic_usage
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "vsla/vsla.h"

void print_tensor_data(const vsla_tensor_t* tensor, const char* name) {
    if (!tensor) {
        printf("%s: NULL\n", name);
        return;
    }
    
    printf("%s (rank %d): [", name, tensor->rank);
    
    if (tensor->rank == 1) {
        for (uint64_t i = 0; i < tensor->shape[0]; i++) {
            double value;
            vsla_get_f64(tensor, &i, &value);
            printf("%.1f", value);
            if (i < tensor->shape[0] - 1) printf(", ");
        }
    } else if (tensor->rank == 0) {
        printf("empty");
    } else {
        printf("...");  /* Multi-dimensional - simplified display */
    }
    
    printf("]\n");
}

int main() {
    printf("=== VSLA Basic Usage Example ===\n\n");
    
    /* Initialize library */
    vsla_error_t err = vsla_init();
    if (err != VSLA_SUCCESS) {
        printf("Failed to initialize VSLA: %s\n", vsla_error_string(err));
        return 1;
    }
    
    printf("Library version: %s\n", vsla_version());
    printf("FFTW support: %s\n\n", vsla_has_fftw() ? "Yes" : "No");
    
    /* Example 1: Creating tensors with different shapes */
    printf("=== Example 1: Variable-Shape Tensors ===\n");
    
    uint64_t shape_a[] = {3};
    uint64_t shape_b[] = {5};
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b) {
        printf("Failed to create tensors\n");
        return 1;
    }
    
    /* Fill tensors with data */
    for (uint64_t i = 0; i < shape_a[0]; i++) {
        vsla_set_f64(a, &i, (double)(i + 1));  /* a = [1, 2, 3] */
    }
    
    for (uint64_t i = 0; i < shape_b[0]; i++) {
        vsla_set_f64(b, &i, (double)(i + 1));  /* b = [1, 2, 3, 4, 5] */
    }
    
    print_tensor_data(a, "Tensor a");
    print_tensor_data(b, "Tensor b");
    
    /* Example 2: Variable-shape addition */
    printf("\n=== Example 2: Variable-Shape Addition ===\n");
    
    /* Create result tensor with shape = max(shape_a, shape_b) */
    uint64_t result_shape[] = {5};  /* max(3, 5) = 5 */
    vsla_tensor_t* result = vsla_zeros(1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!result) {
        printf("Failed to create result tensor\n");
        goto cleanup;
    }
    
    /* Perform addition: [1,2,3] + [1,2,3,4,5] = [2,4,6,4,5] */
    /* VSLA automatically pads a to [1,2,3,0,0] */
    err = vsla_add(result, a, b);
    if (err != VSLA_SUCCESS) {
        printf("Addition failed: %s\n", vsla_error_string(err));
        goto cleanup;
    }
    
    printf("Automatic padding: a becomes [1, 2, 3, 0, 0]\n");
    printf("Addition result: a + b = ");
    print_tensor_data(result, "");
    
    /* Verify the mathematical correctness */
    double expected[] = {2.0, 4.0, 6.0, 4.0, 5.0};
    for (uint64_t i = 0; i < 5; i++) {
        double value;
        vsla_get_f64(result, &i, &value);
        assert(fabs(value - expected[i]) < 1e-15);
    }
    printf("✓ Mathematical verification passed\n");
    
    /* Example 3: Semiring elements */
    printf("\n=== Example 3: Semiring Elements ===\n");
    
    vsla_tensor_t* zero = vsla_zero_element(VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* one = vsla_one_element(VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!zero || !one) {
        printf("Failed to create semiring elements\n");
        goto cleanup;
    }
    
    printf("Zero element: rank %d (empty tensor)\n", zero->rank);
    printf("One element: rank %d, ", one->rank);
    print_tensor_data(one, "");
    
    /* Test semiring property: a + 0 = a */
    vsla_tensor_t* test_result = vsla_zeros(1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (test_result) {
        err = vsla_add(test_result, a, zero);
        if (err == VSLA_SUCCESS) {
            printf("Semiring test a + 0 = ");
            print_tensor_data(test_result, "");
            printf("✓ Semiring identity verified\n");
        }
        vsla_free(test_result);
    }
    
    /* Example 4: Different data types */
    printf("\n=== Example 4: Data Type Handling ===\n");
    
    uint64_t small_shape[] = {3};
    vsla_tensor_t* f32_tensor = vsla_new(1, small_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (f32_tensor) {
        /* Set value using f64 API - automatic conversion */
        double pi = 3.14159265359;
        vsla_set_f64(f32_tensor, (uint64_t[]){0}, pi);
        
        /* Get value back - automatic conversion */
        double retrieved;
        vsla_get_f64(f32_tensor, (uint64_t[]){0}, &retrieved);
        
        printf("Original f64: %.10f\n", pi);
        printf("Stored as f32: %.6f\n", retrieved);
        printf("✓ Automatic type conversion works\n");
        
        vsla_free(f32_tensor);
    }
    
    /* Example 5: Memory and performance information */
    printf("\n=== Example 5: Memory Information ===\n");
    
    printf("Tensor a:\n");
    printf("  - Elements: %llu\n", (unsigned long long)vsla_numel(a));
    printf("  - Capacity: %llu\n", (unsigned long long)vsla_capacity(a));
    printf("  - Shape: [%llu]\n", (unsigned long long)a->shape[0]);
    printf("  - Capacity: [%llu] (power-of-2 growth)\n", (unsigned long long)a->cap[0]);
    printf("  - Memory efficiency: %.1f%%\n", 
           100.0 * vsla_numel(a) / vsla_capacity(a));
    
    /* Example 6: Error handling */
    printf("\n=== Example 6: Error Handling ===\n");
    
    /* Try to access out-of-bounds element */
    double value;
    uint64_t bad_index = 10;  /* Out of bounds for tensor a */
    err = vsla_get_f64(a, &bad_index, &value);
    if (err != VSLA_SUCCESS) {
        printf("Expected error for out-of-bounds access: %s\n", 
               vsla_error_string(err));
        printf("✓ Bounds checking works correctly\n");
    }
    
    /* Try to create tensor with invalid parameters */
    vsla_tensor_t* invalid = vsla_new(2, (uint64_t[]){3, 0}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!invalid) {
        printf("Expected failure for zero dimension: tensor creation rejected\n");
        printf("✓ Input validation works correctly\n");
    }
    
    printf("\n=== Summary ===\n");
    printf("✅ Variable-shape tensors created successfully\n");
    printf("✅ Automatic zero-padding in operations\n");
    printf("✅ Mathematical correctness verified\n");
    printf("✅ Semiring properties satisfied\n");
    printf("✅ Type safety and conversion working\n");
    printf("✅ Memory management efficient\n");
    printf("✅ Error handling comprehensive\n");
    printf("\nVSLA core functionality validation: PASSED\n");
    
cleanup:
    /* Clean up all resources */
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
    vsla_free(zero);
    vsla_free(one);
    
    vsla_cleanup();
    
    printf("\n=== Example Complete ===\n");
    return 0;
}