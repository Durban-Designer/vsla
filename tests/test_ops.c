/**
 * @file test_ops.c
 * @brief Tests for basic tensor operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <math.h>

static vsla_tensor_t *test_tensors[10];
static size_t tensor_count = 0;

static void ops_test_setup(void) {
    tensor_count = 0;
    for (size_t i = 0; i < 10; i++) {
        test_tensors[i] = NULL;
    }
}

static void ops_test_teardown(void) {
    for (size_t i = 0; i < tensor_count; i++) {
        if (test_tensors[i]) {
            vsla_free(test_tensors[i]);
            test_tensors[i] = NULL;
        }
    }
    tensor_count = 0;
}

static vsla_tensor_t* create_test_tensor(size_t size, vsla_model_t model, vsla_dtype_t dtype) {
    if (tensor_count >= 10) return NULL;
    
    vsla_tensor_t* tensor = vsla_new(1, &size, model, dtype);
    if (tensor) {
        test_tensors[tensor_count++] = tensor;
    }
    return tensor;
}

static int test_tensor_addition(void) {
    // Test 1: Same size tensors
    vsla_tensor_t* a = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    // Fill test data: a = [1, 2, 3], b = [4, 5, 6]
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(a, &idx, (double)(i + 1));
        vsla_set_f64(b, &idx, (double)(i + 4));
    }
    
    // Perform addition
    ASSERT_EQ(VSLA_SUCCESS, vsla_add(result, a, b));
    
    // Check results: should be [5, 7, 9]
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double expected = (double)(i + 1) + (double)(i + 4);
        double actual;
        ASSERT_EQ(VSLA_SUCCESS, vsla_get_f64(result, &idx, &actual));
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    
    // Test 2: Different size tensors (shape promotion)
    vsla_tensor_t* c = create_test_tensor(2, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* d = create_test_tensor(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result2 = create_test_tensor(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(c);
    ASSERT_NOT_NULL(d);
    ASSERT_NOT_NULL(result2);
    
    // Fill test data: c = [1, 2], d = [3, 4, 5, 6]
    for (size_t i = 0; i < 2; i++) {
        idx = i;
        vsla_set_f64(c, &idx, (double)(i + 1));
    }
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        vsla_set_f64(d, &idx, (double)(i + 3));
    }
    
    ASSERT_EQ(VSLA_SUCCESS, vsla_add(result2, c, d));
    
    // Check results: should be [4, 6, 5, 6] (c padded to [1, 2, 0, 0])
    double expected_vals[] = {4.0, 6.0, 5.0, 6.0};
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        double actual = vsla_get_f64(result2, &idx);
        ASSERT_FLOAT_EQ(expected_vals[i], actual, 1e-12);
    }
    return 1;
}

static int test_tensor_subtraction(void) {
    vsla_tensor_t* a = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    // Fill test data: a = [5, 7, 9], b = [1, 2, 3]
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(a, &idx, (double)(i * 2 + 5));
        vsla_set_f64(b, &idx, (double)(i + 1));
    }
    
    ASSERT_EQ(VSLA_SUCCESS, vsla_sub(result, a, b));
    
    // Check results: should be [4, 5, 6]
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double expected = (double)(i * 2 + 5) - (double)(i + 1);
        double actual;
        ASSERT_EQ(VSLA_SUCCESS, vsla_get_f64(result, &idx, &actual));
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    return 1;
}

static int test_tensor_scaling(void) {
    vsla_tensor_t* a = create_test_tensor(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = create_test_tensor(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(result);
    
    // Fill test data: a = [1, 2, 3, 4]
    uint64_t idx;
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        vsla_set_f64(a, &idx, (double)(i + 1));
    }
    
    // Scale by 2.5
    double scalar = 2.5;
    ASSERT_EQ(VSLA_SUCCESS, vsla_scale(result, a, scalar));
    
    // Check results: should be [2.5, 5.0, 7.5, 10.0]
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        double expected = (double)(i + 1) * scalar;
        double actual;
        ASSERT_EQ(VSLA_SUCCESS, vsla_get_f64(result, &idx, &actual));
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    
    // Test in-place scaling
    ASSERT_EQ(VSLA_SUCCESS, vsla_scale(a, a, 0.5));
    
    // Check results: original values should now be halved
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        double expected = (double)(i + 1) * 0.5;
        double actual = vsla_get_f64(a, &idx);
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    return 1;
}

static int test_hadamard_product(void) {
    vsla_tensor_t* a = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(result);
    
    // Fill test data: a = [2, 3, 4], b = [5, 6, 7]
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(a, &idx, (double)(i + 2));
        vsla_set_f64(b, &idx, (double)(i + 5));
    }
    
    ASSERT_EQ(VSLA_SUCCESS, vsla_hadamard(result, a, b));
    
    // Check results: should be [10, 18, 28]
    double expected_vals[] = {10.0, 18.0, 28.0};
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double actual;
        ASSERT_EQ(VSLA_SUCCESS, vsla_get_f64(result, &idx, &actual));
        ASSERT_FLOAT_EQ(expected_vals[i], actual, 1e-12);
    }
    return 1;
}

static int test_matrix_transpose(void) {
    // Create 2x3 matrix
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* matrix = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(matrix);
    test_tensors[tensor_count++] = matrix;
    
    // Create 3x2 result matrix
    uint64_t result_shape[] = {3, 2};
    vsla_tensor_t* result = vsla_new(2, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(result);
    test_tensors[tensor_count++] = result;
    
    // Fill matrix with values:
    // [1 2 3]
    // [4 5 6]
    uint64_t idx[2];
    double val = 1.0;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            idx[0] = i;
            idx[1] = j;
            vsla_set_f64(matrix, idx, val);
            val += 1.0;
        }
    }
    
    ASSERT_EQ(VSLA_SUCCESS, vsla_transpose(result, matrix));
    
    // Check result should be:
    // [1 4]
    // [2 5]
    // [3 6]
    double expected[][2] = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            idx[0] = i;
            idx[1] = j;
            double actual = vsla_get_f64(result, idx);
            ASSERT_FLOAT_EQ(expected[i][j], actual, 1e-12);
        }
    }
    return 1;
}

static int test_tensor_reshape(void) {
    // Create 1D tensor with 6 elements
    vsla_tensor_t* tensor = create_test_tensor(6, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with values [1, 2, 3, 4, 5, 6]
    uint64_t idx;
    for (size_t i = 0; i < 6; i++) {
        idx = i;
        vsla_set_f64(tensor, &idx, (double)(i + 1));
    }
    
    // Reshape to 2x3 matrix
    uint64_t new_shape[] = {2, 3};
    ASSERT_EQ(VSLA_SUCCESS, vsla_reshape(tensor, 2, new_shape));
    
    // Verify shape changed
    ASSERT_EQ(2, tensor->ndim);
    ASSERT_EQ(2, tensor->shape[0]);
    ASSERT_EQ(3, tensor->shape[1]);
    
    // Verify data integrity (row-major order)
    uint64_t matrix_idx[2];
    double expected = 1.0;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            matrix_idx[0] = i;
            matrix_idx[1] = j;
            double actual = vsla_get_f64(tensor, matrix_idx);
            ASSERT_FLOAT_EQ(expected, actual, 1e-12);
            expected += 1.0;
        }
    }
    return 1;
}

static int test_tensor_norm(void) {
    vsla_tensor_t* tensor = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with values [3, 4, 0] (should give norm = 5)
    uint64_t idx;
    vsla_set_f64(tensor, &(idx = 0), 3.0);
    vsla_set_f64(tensor, &(idx = 1), 4.0);
    vsla_set_f64(tensor, &(idx = 2), 0.0);
    
    double norm;
    ASSERT_EQ(VSLA_SUCCESS, vsla_norm(tensor, &norm));
    ASSERT_FLOAT_EQ(5.0, norm, 1e-12);
    return 1;
}

static int test_tensor_sum(void) {
    vsla_tensor_t* tensor = create_test_tensor(4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with values [1, 2, 3, 4] (sum = 10)
    uint64_t idx;
    for (size_t i = 0; i < 4; i++) {
        idx = i;
        vsla_set_f64(tensor, &idx, (double)(i + 1));
    }
    
    double sum;
    ASSERT_EQ(VSLA_SUCCESS, vsla_sum(tensor, &sum));
    ASSERT_FLOAT_EQ(10.0, sum, 1e-12);
    return 1;
}

static int test_tensor_max_min(void) {
    vsla_tensor_t* tensor = create_test_tensor(5, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with values [-2, 5, 1, -10, 3]
    double vals[] = {-2.0, 5.0, 1.0, -10.0, 3.0};
    uint64_t idx;
    for (size_t i = 0; i < 5; i++) {
        idx = i;
        vsla_set_f64(tensor, &idx, vals[i]);
    }
    
    double max_val, min_val;
    ASSERT_EQ(VSLA_SUCCESS, vsla_max(tensor, &max_val));
    ASSERT_EQ(VSLA_SUCCESS, vsla_min(tensor, &min_val));
    
    ASSERT_FLOAT_EQ(5.0, max_val, 1e-12);
    ASSERT_FLOAT_EQ(-10.0, min_val, 1e-12);
    return 1;
}

static int test_tensor_slice(void) {
    // Create 1D tensor with 5 elements
    vsla_tensor_t* tensor = create_test_tensor(5, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with values [1, 2, 3, 4, 5]
    uint64_t idx;
    for (size_t i = 0; i < 5; i++) {
        idx = i;
        vsla_set_f64(tensor, &idx, (double)(i + 1));
    }
    
    // Create slice [1:4] (should get elements 2, 3, 4)
    uint64_t start[] = {1};
    uint64_t end[] = {4};
    vsla_tensor_t* slice = vsla_slice(tensor, start, end);
    ASSERT_NOT_NULL(slice);
    test_tensors[tensor_count++] = slice;
    
    // Verify slice size and content
    ASSERT_EQ(3, slice->shape[0]);
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double expected = (double)(i + 2);  // Values 2, 3, 4
        double actual = vsla_get_f64(slice, &idx);
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    return 1;
}

static int test_rank_padding(void) {
    // Create 1D tensor
    vsla_tensor_t* tensor = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill with test data
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(tensor, &idx, (double)(i + 1));
    }
    
    // Pad to rank 3
    uint64_t target_cap[] = {3, 2, 4};
    ASSERT_EQ(VSLA_SUCCESS, vsla_pad_rank(tensor, 3, target_cap));
    
    // Verify new rank and shape
    ASSERT_EQ(3, tensor->ndim);
    ASSERT_EQ(3, tensor->shape[0]);
    ASSERT_EQ(1, tensor->shape[1]); // Default padding
    ASSERT_EQ(1, tensor->shape[2]); // Default padding
    
    // Verify original data is preserved
    uint64_t multi_idx[3] = {0, 0, 0};
    for (size_t i = 0; i < 3; i++) {
        multi_idx[0] = i;
        double expected = (double)(i + 1);
        double actual = vsla_get_f64(tensor, multi_idx);
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    return 1;
}

static int test_error_conditions(void) {
    // Test NULL pointer handling
    vsla_tensor_t* tensor = create_test_tensor(3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    double dummy;
    ASSERT_EQ(VSLA_ERROR_NULL_POINTER, vsla_add(NULL, tensor, tensor));
    ASSERT_EQ(VSLA_ERROR_NULL_POINTER, vsla_add(tensor, NULL, tensor));
    ASSERT_EQ(VSLA_ERROR_NULL_POINTER, vsla_sum(NULL, &dummy));
    ASSERT_EQ(VSLA_ERROR_NULL_POINTER, vsla_sum(tensor, NULL));
    
    // Test incompatible shapes for operations that require them
    uint64_t shape2d[] = {2, 2};
    vsla_tensor_t* matrix = vsla_new(2, shape2d, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(matrix);
    test_tensors[tensor_count++] = matrix;
    
    // Transpose on 1D tensor should fail
    ASSERT_EQ(VSLA_ERROR_INVALID_ARGUMENT, vsla_transpose(matrix, tensor));
    return 1;
}

static void run_ops_tests(void) {
    TEST_CASE("Tensor Addition", test_tensor_addition);
    TEST_CASE("Tensor Subtraction", test_tensor_subtraction);
    TEST_CASE("Tensor Scaling", test_tensor_scaling);
    TEST_CASE("Hadamard Product", test_hadamard_product);
    TEST_CASE("Matrix Transpose", test_matrix_transpose);
    TEST_CASE("Tensor Reshape", test_tensor_reshape);
    TEST_CASE("Tensor Norm", test_tensor_norm);
    TEST_CASE("Tensor Sum", test_tensor_sum);
    TEST_CASE("Max/Min Operations", test_tensor_max_min);
    TEST_CASE("Tensor Slice", test_tensor_slice);
    TEST_CASE("Rank Padding", test_rank_padding);
    TEST_CASE("Error Conditions", test_error_conditions);
}

static const test_suite_t ops_suite = {
    .name = "ops",
    .setup = ops_test_setup,
    .teardown = ops_test_teardown,
    .run_tests = run_ops_tests
};

void register_ops_tests(void) {
    register_test_suite(&ops_suite);
}