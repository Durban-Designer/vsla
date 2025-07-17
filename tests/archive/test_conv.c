/**
 * @file test_conv.c
 * @brief Tests for convolution operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"

static vsla_context_t* ctx = NULL;

static int conv_test_setup(void) {
    ctx = vsla_init(NULL);
    ASSERT_NOT_NULL(ctx);
    return 1;
}

static void conv_test_teardown(void) {
    vsla_cleanup(ctx);
}

static int test_conv_1d_simple(void) {
    uint64_t a_shape[] = {3};
    uint64_t b_shape[] = {2};
    uint64_t out_shape[] = {4};
    vsla_tensor_t* a = vsla_new(1, a_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, b_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

    double a_data[] = {1, 2, 3};
    double b_data[] = {4, 5};
    memcpy(a->data, a_data, sizeof(a_data));
    memcpy(b->data, b_data, sizeof(b_data));

    ASSERT_EQ(VSLA_SUCCESS, vsla_conv(ctx, out, a, b));

    double expected[] = {4, 13, 22, 15};
    for (uint64_t i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(expected[i], ((double*)out->data)[i], 1e-9);
    }

    vsla_free(a);
    vsla_free(b);
    vsla_free(out);
    return 1;
}

static void run_conv_tests(void) {
    TEST_CASE("1D Simple Convolution", test_conv_1d_simple);
}

static const test_suite_t conv_suite = {
    .name = "conv",
    .setup = (int (*)(void))conv_test_setup,
    .teardown = conv_test_teardown,
    .run_tests = run_conv_tests
};

void register_conv_tests(void) {
    register_test_suite(&conv_suite);
}