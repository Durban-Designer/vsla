/**
 * @file test_arithmetic_unified.c
 * @brief Comprehensive tests for arithmetic operations using unified interface
 * 
 * Tests cover: success cases, failure cases, and edge cases for all backends
 * 
 * @copyright MIT License
 */

#include "test_unified_framework.h"

/* Test tensor addition - success case */
void test_add_success(void) {
    UNIFIED_TEST_CASE("Addition - Success Case");
    
    /* Create test tensors */
    vsla_tensor_t* a = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Fill test data: a = [1, 2, 3, 4], b = [5, 6, 7, 8] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, a, &idx, (double)(i + 1)));
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, b, &idx, (double)(i + 5)));
    }

    
    
    /* Perform addition */
    ASSERT_SUCCESS(vsla_add(g_test_ctx, result, a, b));
    
    /* Verify results: result = [6, 8, 10, 12] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        double val;
        ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, result, &idx, &val));
        ASSERT_DOUBLE_EQ(val, (double)(i + 6));
    }
    
    /* Cleanup */
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test tensor addition - failure case (incompatible shapes) */
void test_add_incompatible_shapes(void) {
    UNIFIED_TEST_CASE("Addition - Incompatible Shapes");
    
    /* Create tensors with different shapes */
    vsla_tensor_t* a = unified_test_create_tensor_1d(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Addition should fail with dimension mismatch */
    ASSERT_ERROR(vsla_add(g_test_ctx, result, a, b), VSLA_ERROR_DIMENSION_MISMATCH);
    
    /* Cleanup */
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test tensor addition - null pointer */
void test_add_null_pointers(void) {
    UNIFIED_TEST_CASE("Addition - Null Pointers");
    
    vsla_tensor_t* a = unified_test_create_tensor_1d(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(a);
    
    /* Test various null pointer combinations */
    ASSERT_ERROR(vsla_add(g_test_ctx, NULL, a, a), VSLA_ERROR_NULL_POINTER);
    ASSERT_ERROR(vsla_add(g_test_ctx, a, NULL, a), VSLA_ERROR_NULL_POINTER);
    ASSERT_ERROR(vsla_add(g_test_ctx, a, a, NULL), VSLA_ERROR_NULL_POINTER);
    ASSERT_ERROR(vsla_add(NULL, a, a, a), VSLA_ERROR_NULL_POINTER);
    
    vsla_tensor_free(a);
    UNIFIED_TEST_END();
}

/* Test tensor subtraction - success case */
void test_sub_success(void) {
    UNIFIED_TEST_CASE("Subtraction - Success Case");
    
    /* Create test tensors */
    vsla_tensor_t* a = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Fill test data: a = [10, 20, 30, 40], b = [1, 2, 3, 4] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, a, &idx, (double)((i + 1) * 10)));
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, b, &idx, (double)(i + 1)));
    }
    
    /* Perform subtraction */
    ASSERT_SUCCESS(vsla_sub(g_test_ctx, result, a, b));
    
    /* Verify results: result = [9, 18, 27, 36] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        double val;
        ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, result, &idx, &val));
        ASSERT_DOUBLE_EQ(val, (double)((i + 1) * 10 - (i + 1)));
    }
    
    /* Cleanup */
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test tensor scaling - success case */
void test_scale_success(void) {
    UNIFIED_TEST_CASE("Scaling - Success Case");
    
    /* Create test tensors */
    vsla_tensor_t* a = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(result);
    
    /* Fill test data: a = [1, 2, 3, 4] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, a, &idx, (double)(i + 1)));
    }
    
    /* Perform scaling by 2.5 */
    ASSERT_SUCCESS(vsla_scale(g_test_ctx, result, a, 2.5));
    
    /* Verify results: result = [2.5, 5.0, 7.5, 10.0] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        double val;
        ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, result, &idx, &val));
        ASSERT_DOUBLE_EQ(val, (double)(i + 1) * 2.5);
    }
    
    /* Cleanup */
    vsla_tensor_free(a);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test hadamard product - success case */
void test_hadamard_success(void) {
    UNIFIED_TEST_CASE("Hadamard Product - Success Case");
    
    /* Create test tensors */
    vsla_tensor_t* a = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_1d(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Fill test data: a = [2, 3, 4, 5], b = [1, 2, 3, 4] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, a, &idx, (double)(i + 2)));
        ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, b, &idx, (double)(i + 1)));
    }
    
    /* Perform hadamard product */
    ASSERT_SUCCESS(vsla_hadamard(g_test_ctx, result, a, b));
    
    /* Verify results: result = [2, 6, 12, 20] */
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t idx = i;
        double val;
        ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, result, &idx, &val));
        ASSERT_DOUBLE_EQ(val, (double)((i + 2) * (i + 1)));
    }
    
    /* Cleanup */
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test fill operation - success case */
void test_fill_success(void) {
    UNIFIED_TEST_CASE("Fill - Success Case");
    
    /* Create test tensor */
    vsla_tensor_t* tensor = unified_test_create_tensor_1d(5, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    /* Fill with value 3.14 */
    ASSERT_SUCCESS(vsla_fill(g_test_ctx, tensor, 3.14));
    
    /* Verify all elements are 3.14 */
    for (uint64_t i = 0; i < 5; i++) {
        uint64_t idx = i;
        double val;
        ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, tensor, &idx, &val));
        ASSERT_DOUBLE_EQ(val, 3.14);
    }
    
    vsla_tensor_free(tensor);
    UNIFIED_TEST_END();
}

/* Test edge case: zero-sized tensor */
void test_zero_size_tensor(void) {
    UNIFIED_TEST_CASE("Zero Size Tensor Operations");
    
    /* Create zero-sized tensor (scalar) */
    uint64_t shape[] = {0};
    vsla_tensor_t* zero_tensor = vsla_tensor_create(g_test_ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (zero_tensor) {
        /* Fill should succeed but do nothing */
        ASSERT_SUCCESS(vsla_fill(g_test_ctx, zero_tensor, 1.0));
        vsla_tensor_free(zero_tensor);
    }
    
    UNIFIED_TEST_END();
}

/* Test mixed data types - should fail */
void test_mixed_data_types(void) {
    UNIFIED_TEST_CASE("Mixed Data Types - Should Fail");
    
    /* Create tensors with different data types */
    vsla_tensor_t* a = unified_test_create_tensor_1d(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_1d(3, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = unified_test_create_tensor_1d(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Operations should fail with type mismatch */
    ASSERT_ERROR(vsla_add(g_test_ctx, result, a, b), VSLA_ERROR_INVALID_ARGUMENT);
    ASSERT_ERROR(vsla_hadamard(g_test_ctx, result, a, b), VSLA_ERROR_INVALID_ARGUMENT);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Test 2D tensor operations */
void test_2d_operations(void) {
    UNIFIED_TEST_CASE("2D Tensor Operations");
    
    /* Create 2x3 tensors */
    vsla_tensor_t* a = unified_test_create_tensor_2d(2, 3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = unified_test_create_tensor_2d(2, 3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = unified_test_create_tensor_2d(2, 3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    /* Fill test data */
    for (uint64_t i = 0; i < 2; i++) {
        for (uint64_t j = 0; j < 3; j++) {
            uint64_t indices[] = {i, j};
            ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, a, indices, (double)(i * 3 + j + 1)));
            ASSERT_SUCCESS(vsla_set_f64(g_test_ctx, b, indices, (double)(i * 3 + j + 10)));
        }
    }

    
    
    /* Test addition */
    ASSERT_SUCCESS(vsla_add(g_test_ctx, result, a, b));
    
    /* Verify results */
    
    for (uint64_t i = 0; i < 2; i++) {
        for (uint64_t j = 0; j < 3; j++) {
            uint64_t indices[] = {i, j};
            double val;
            ASSERT_SUCCESS(vsla_get_f64(g_test_ctx, result, indices, &val));
            ASSERT_DOUBLE_EQ(val, (double)((i * 3 + j + 1) + (i * 3 + j + 10)));
        }
    }
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    
    UNIFIED_TEST_END();
}

/* Main test runner for arithmetic operations */
int run_arithmetic_tests(vsla_backend_t backend) {
    printf("=== Arithmetic Operations Tests ===\n");
    
    if (!UNIFIED_TEST_INIT(backend)) {
        printf("Failed to initialize test framework\n");
        return 1;
    }
    
    /* Run all tests */
    test_add_success();
    test_add_incompatible_shapes();
    test_add_null_pointers();
    test_sub_success();
    test_scale_success();
    test_hadamard_success();
    test_fill_success();
    test_zero_size_tensor();
    test_mixed_data_types();
    test_2d_operations();
    
    unified_test_print_summary();
    UNIFIED_TEST_CLEANUP();
    
    return g_test_results.tests_failed;
}