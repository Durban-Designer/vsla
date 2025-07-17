/**
 * @file test_ops.c
 * @brief Tests for basic tensor operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"

static vsla_context_t* ctx = NULL;
static vsla_tensor_t *test_tensors[10];
static size_t tensor_count = 0;

static int ops_test_setup(void) {
    tensor_count = 0;
    for (size_t i = 0; i < 10; i++) {
        test_tensors[i] = NULL;
    }
    ctx = vsla_init(NULL);
    ASSERT_NOT_NULL(ctx);
    return 1;
}

static void ops_test_teardown(void) {
    for (size_t i = 0; i < tensor_count; i++) {
        if (test_tensors[i]) {
            vsla_free(test_tensors[i]);
            test_tensors[i] = NULL;
        }
    }
    tensor_count = 0;
    vsla_cleanup(ctx);
}

static vsla_tensor_t* create_test_tensor(uint8_t rank, const uint64_t* shape, vsla_model_t model, vsla_dtype_t dtype) {
    if (tensor_count >= 10) return NULL;
    
    vsla_tensor_t* tensor = vsla_new(rank, shape, model, dtype);
    if (tensor) {
        test_tensors[tensor_count++] = tensor;
    }
    return tensor;
}

static int test_tensor_addition(void) {
    uint64_t shape[] = {3};
    vsla_tensor_t* a = create_test_tensor(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = create_test_tensor(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = create_test_tensor(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(a, &idx, (double)(i + 1));
        vsla_set_f64(b, &idx, (double)(i + 4));
    }
    
    ASSERT_EQ(VSLA_SUCCESS, vsla_add(ctx, result, a, b));
    
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double expected = (double)(i + 1) + (double)(i + 4);
        double actual;
        ASSERT_EQ(VSLA_SUCCESS, vsla_get_f64(result, &idx, &actual));
        ASSERT_FLOAT_EQ(expected, actual, 1e-12);
    }
    return 1;
}

static void run_ops_tests(void) {
    TEST_CASE("Tensor Addition", test_tensor_addition);
}

static const test_suite_t ops_suite = {
    .name = "ops",
    .setup = (int (*)(void))ops_test_setup,
    .teardown = ops_test_teardown,
    .run_tests = run_ops_tests
};

void register_ops_tests(void) {
    register_test_suite(&ops_suite);
}