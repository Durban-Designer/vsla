/**
 * @file test_tensor.c
 * @brief Comprehensive tests for tensor module
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <float.h>

/* Test helper functions */
static int tensors_equal(const vsla_tensor_t* a, const vsla_tensor_t* b, double eps) {
    if (!a || !b) return a == b;
    if (a->rank != b->rank || a->model != b->model || a->dtype != b->dtype) {
        return 0;
    }
    
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    
    uint64_t n = vsla_numel(a);
    uint64_t* indices = calloc(a->rank, sizeof(uint64_t));
    if (!indices) return 0;
    
    int equal = 1;
    for (uint64_t idx = 0; idx < n && equal; idx++) {
        double val_a, val_b;
        if (vsla_get_f64(a, indices, &val_a) != VSLA_SUCCESS ||
            vsla_get_f64(b, indices, &val_b) != VSLA_SUCCESS) {
            equal = 0;
            break;
        }
        
        if (fabs(val_a - val_b) > eps) {
            equal = 0;
            break;
        }
        
        /* Increment indices */
        int carry = 1;
        for (int i = a->rank - 1; i >= 0 && carry; i--) {
            indices[i]++;
            if (indices[i] < a->shape[i]) {
                carry = 0;
            } else {
                indices[i] = 0;
            }
        }
    }
    
    free(indices);
    return equal;
}

/* Test cases */
DECLARE_TEST(tensor_creation_basic) {
    uint64_t shape[] = {3, 4};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_NOT_NULL(tensor);
    ASSERT_EQ(tensor->rank, 2);
    ASSERT_EQ(tensor->model, VSLA_MODEL_A);
    ASSERT_EQ(tensor->dtype, VSLA_DTYPE_F64);
    ASSERT_EQ(tensor->shape[0], 3);
    ASSERT_EQ(tensor->shape[1], 4);
    ASSERT_EQ(vsla_numel(tensor), 12);
    
    /* Check capacity is power of 2 */
    ASSERT_TRUE(vsla_is_pow2(tensor->cap[0]));
    ASSERT_TRUE(vsla_is_pow2(tensor->cap[1]));
    ASSERT_TRUE(tensor->cap[0] >= tensor->shape[0]);
    ASSERT_TRUE(tensor->cap[1] >= tensor->shape[1]);
    
    vsla_free(tensor);
    return 1;
}

DECLARE_TEST(tensor_creation_edge_cases) {
    /* Zero rank tensor */
    vsla_tensor_t* empty = vsla_new(0, NULL, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(empty);
    ASSERT_EQ(empty->rank, 0);
    ASSERT_EQ(vsla_numel(empty), 0);
    vsla_free(empty);
    
    /* 1D tensor */
    uint64_t shape1d = 10;
    vsla_tensor_t* vec = vsla_new(1, &shape1d, VSLA_MODEL_B, VSLA_DTYPE_F32);
    ASSERT_NOT_NULL(vec);
    ASSERT_EQ(vec->rank, 1);
    ASSERT_EQ(vec->shape[0], 10);
    ASSERT_EQ(vsla_numel(vec), 10);
    vsla_free(vec);
    
    /* Large dimensions */
    uint64_t large_shape[] = {1000, 1000};
    vsla_tensor_t* large = vsla_new(2, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(large);
    vsla_free(large);
    
    return 1;
}

DECLARE_TEST(tensor_creation_invalid) {
    uint64_t shape[] = {3, 4};
    
    /* Invalid model */
    ASSERT_NULL(vsla_new(2, shape, 99, VSLA_DTYPE_F64));
    
    /* Invalid dtype */
    ASSERT_NULL(vsla_new(2, shape, VSLA_MODEL_A, 99));
    
    /* NULL shape with non-zero rank */
    ASSERT_NULL(vsla_new(2, NULL, VSLA_MODEL_A, VSLA_DTYPE_F64));
    
    /* Zero dimension in shape */
    uint64_t bad_shape[] = {3, 0};
    ASSERT_NULL(vsla_new(2, bad_shape, VSLA_MODEL_A, VSLA_DTYPE_F64));
    
    /* Extremely large dimension */
    uint64_t huge_shape[] = {UINT64_MAX};
    ASSERT_NULL(vsla_new(1, huge_shape, VSLA_MODEL_A, VSLA_DTYPE_F64));
    
    return 1;
}

DECLARE_TEST(tensor_copy) {
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* orig = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(orig);
    
    /* Fill with test data */
    uint64_t indices[] = {1, 2};
    vsla_set_f64(orig, indices, 42.0);
    
    /* Copy tensor */
    vsla_tensor_t* copy = vsla_copy_basic(orig);
    ASSERT_NOT_NULL(copy);
    
    /* Verify copy */
    ASSERT_TRUE(tensors_equal(orig, copy, 1e-15));
    
    /* Modify original, copy should be unchanged */
    vsla_set_f64(orig, indices, 99.0);
    double orig_val, copy_val;
    vsla_get_f64(orig, indices, &orig_val);
    vsla_get_f64(copy, indices, &copy_val);
    ASSERT_DOUBLE_EQ(orig_val, 99.0, 1e-15);
    ASSERT_DOUBLE_EQ(copy_val, 42.0, 1e-15);
    
    vsla_free(orig);
    vsla_free(copy);
    return 1;
}

DECLARE_TEST(tensor_zeros_ones) {
    uint64_t shape[] = {2, 3};
    
    /* Test zeros */
    vsla_tensor_t* zeros = vsla_zeros(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(zeros);
    
    uint64_t indices[] = {1, 1};
    double val;
    vsla_get_f64(zeros, indices, &val);
    ASSERT_DOUBLE_EQ(val, 0.0, 1e-15);
    
    /* Test ones */
    vsla_tensor_t* ones = vsla_ones(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(ones);
    
    vsla_get_f64(ones, indices, &val);
    ASSERT_DOUBLE_EQ(val, 1.0, 1e-15);
    
    vsla_free(zeros);
    vsla_free(ones);
    return 1;
}

DECLARE_TEST(tensor_get_set) {
    uint64_t shape[] = {3, 4, 5};
    vsla_tensor_t* tensor = vsla_new(3, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    /* Test setting and getting values */
    uint64_t indices1[] = {0, 0, 0};
    uint64_t indices2[] = {2, 3, 4};
    uint64_t indices3[] = {1, 2, 3};
    
    ASSERT_EQ(vsla_set_f64(tensor, indices1, 1.23), VSLA_SUCCESS);
    ASSERT_EQ(vsla_set_f64(tensor, indices2, -4.56), VSLA_SUCCESS);
    ASSERT_EQ(vsla_set_f64(tensor, indices3, 7.89), VSLA_SUCCESS);
    
    double val;
    ASSERT_EQ(vsla_get_f64(tensor, indices1, &val), VSLA_SUCCESS);
    ASSERT_DOUBLE_EQ(val, 1.23, 1e-15);
    
    ASSERT_EQ(vsla_get_f64(tensor, indices2, &val), VSLA_SUCCESS);
    ASSERT_DOUBLE_EQ(val, -4.56, 1e-15);
    
    ASSERT_EQ(vsla_get_f64(tensor, indices3, &val), VSLA_SUCCESS);
    ASSERT_DOUBLE_EQ(val, 7.89, 1e-15);
    
    /* Test out of bounds */
    uint64_t bad_indices[] = {3, 0, 0};
    ASSERT_NE(vsla_get_f64(tensor, bad_indices, &val), VSLA_SUCCESS);
    ASSERT_NE(vsla_set_f64(tensor, bad_indices, 99.0), VSLA_SUCCESS);
    
    vsla_free(tensor);
    return 1;
}

DECLARE_TEST(tensor_fill) {
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    /* Fill with specific value */
    ASSERT_EQ(vsla_fill_basic(tensor, 3.14), VSLA_SUCCESS);
    
    /* Check all elements */
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            uint64_t indices[] = {i, j};
            double val;
            ASSERT_EQ(vsla_get_f64(tensor, indices, &val), VSLA_SUCCESS);
            ASSERT_DOUBLE_EQ(val, 3.14, 1e-15);
        }
    }
    
    /* Test invalid values */
    ASSERT_NE(vsla_fill_basic(tensor, NAN), VSLA_SUCCESS);
    ASSERT_NE(vsla_fill_basic(tensor, INFINITY), VSLA_SUCCESS);
    ASSERT_NE(vsla_fill_basic(NULL, 1.0), VSLA_SUCCESS);
    
    vsla_free(tensor);
    return 1;
}

DECLARE_TEST(tensor_dtype_conversion) {
    uint64_t shape[] = {2, 2};
    
    /* Test with f32 */
    vsla_tensor_t* f32_tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    ASSERT_NOT_NULL(f32_tensor);
    
    uint64_t indices[] = {0, 0};
    ASSERT_EQ(vsla_set_f64(f32_tensor, indices, 1.5), VSLA_SUCCESS);
    
    double val;
    ASSERT_EQ(vsla_get_f64(f32_tensor, indices, &val), VSLA_SUCCESS);
    ASSERT_DOUBLE_EQ(val, 1.5, 1e-6);  /* f32 precision */
    
    vsla_free(f32_tensor);
    return 1;
}

DECLARE_TEST(tensor_shape_equal) {
    uint64_t shape1[] = {3, 4};
    uint64_t shape2[] = {3, 4};
    uint64_t shape3[] = {3, 5};
    uint64_t shape4[] = {3, 4, 1};
    
    vsla_tensor_t* t1 = vsla_new(2, shape1, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t2 = vsla_new(2, shape2, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t3 = vsla_new(2, shape3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t4 = vsla_new(3, shape4, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_TRUE(vsla_shape_equal(t1, t2));
    ASSERT_FALSE(vsla_shape_equal(t1, t3));
    ASSERT_FALSE(vsla_shape_equal(t1, t4));
    ASSERT_FALSE(vsla_shape_equal(t1, NULL));
    ASSERT_FALSE(vsla_shape_equal(NULL, t1));
    ASSERT_FALSE(vsla_shape_equal(NULL, NULL));  /* NULL tensors have no meaningful shape */
    
    vsla_free(t1);
    vsla_free(t2);
    vsla_free(t3);
    vsla_free(t4);
    return 1;
}

DECLARE_TEST(tensor_semiring_elements) {
    /* Test zero element */
    vsla_tensor_t* zero = vsla_zero_element(VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(zero);
    ASSERT_EQ(zero->rank, 0);
    ASSERT_EQ(zero->model, VSLA_MODEL_A);
    ASSERT_EQ(zero->dtype, VSLA_DTYPE_F64);
    vsla_free(zero);
    
    /* Test one element */
    vsla_tensor_t* one = vsla_one_element(VSLA_MODEL_B, VSLA_DTYPE_F32);
    ASSERT_NOT_NULL(one);
    ASSERT_EQ(one->rank, 1);
    ASSERT_EQ(one->shape[0], 1);
    ASSERT_EQ(one->model, VSLA_MODEL_B);
    ASSERT_EQ(one->dtype, VSLA_DTYPE_F32);
    
    uint64_t idx = 0;
    double val;
    ASSERT_EQ(vsla_get_f64(one, &idx, &val), VSLA_SUCCESS);
    ASSERT_DOUBLE_EQ(val, 1.0, 1e-6);
    
    vsla_free(one);
    return 1;
}

DECLARE_TEST(tensor_memory_management) {
    /* Test that vsla_free handles NULL gracefully */
    vsla_free(NULL);
    
    /* Test large tensor allocation and deallocation */
    uint64_t large_shape[] = {100, 100};
    vsla_tensor_t* large = vsla_new(2, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (large) {  /* May fail on systems with limited memory */
        ASSERT_EQ(vsla_numel(large), 10000);
        vsla_free(large);
    }
    
    return 1;
}

DECLARE_TEST(tensor_capacity_management) {
    uint64_t shape[] = {5, 7};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    /* Capacity should be next power of 2 */
    ASSERT_EQ(tensor->cap[0], 8);  /* next_pow2(5) = 8 */
    ASSERT_EQ(tensor->cap[1], 8);  /* next_pow2(7) = 8 */
    
    /* Total capacity */
    ASSERT_EQ(vsla_capacity(tensor), 64);
    
    vsla_free(tensor);
    return 1;
}

/* Test suite setup */
static void tensor_test_setup(void) {
    /* Any setup needed before tests */
}

static void tensor_test_teardown(void) {
    /* Any cleanup needed after tests */
}

static void run_tensor_tests(void) {
    RUN_TEST(tensor_creation_basic);
    RUN_TEST(tensor_creation_edge_cases);
    RUN_TEST(tensor_creation_invalid);
    RUN_TEST(tensor_copy);
    RUN_TEST(tensor_zeros_ones);
    RUN_TEST(tensor_get_set);
    RUN_TEST(tensor_fill);
    RUN_TEST(tensor_dtype_conversion);
    RUN_TEST(tensor_shape_equal);
    RUN_TEST(tensor_semiring_elements);
    RUN_TEST(tensor_memory_management);
    RUN_TEST(tensor_capacity_management);
}

static const test_suite_t tensor_suite = {
    .name = "tensor",
    .setup = tensor_test_setup,
    .teardown = tensor_test_teardown,
    .run_tests = run_tensor_tests
};

void register_tensor_tests(void) {
    register_test_suite(&tensor_suite);
}