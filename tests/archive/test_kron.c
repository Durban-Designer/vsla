/**
 * @file test_kron.c
 * @brief Tests for Kronecker product operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"

static vsla_context_t* ctx = NULL;

static int kron_test_setup(void) {
    ctx = vsla_init(NULL);
    ASSERT_NOT_NULL(ctx);
    return 1;
}

static void kron_test_teardown(void) {
    vsla_cleanup(ctx);
}

static int test_kron_1d_simple(void) {
    uint64_t a_shape[] = {2};
    uint64_t b_shape[] = {3};
    uint64_t out_shape[] = {6};
    vsla_tensor_t* a = vsla_new(1, a_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, b_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, out_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);

    double a_data[] = {1, 2};
    double b_data[] = {3, 4, 5};
    memcpy(a->data, a_data, sizeof(a_data));
    memcpy(b->data, b_data, sizeof(b_data));

    ASSERT_EQ(VSLA_SUCCESS, vsla_kron(ctx, out, a, b));

    double expected[] = {3, 4, 5, 6, 8, 10};
    for (uint64_t i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(expected[i], ((double*)out->data)[i], 1e-9);
    }

    vsla_free(a);
    vsla_free(b);
    vsla_free(out);
    return 1;
}

static void run_kron_tests(void) {
    TEST_CASE("1D Simple Kronecker Product", test_kron_1d_simple);
}

static const test_suite_t kron_suite = {
    .name = "kron",
    .setup = (int (*)(void))kron_test_setup,
    .teardown = kron_test_teardown,
    .run_tests = run_kron_tests
};

void register_kron_tests(void) {
    register_test_suite(&kron_suite);
}