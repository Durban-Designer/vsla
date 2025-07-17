/**
 * @file test_autograd.c
 * @brief Tests for autograd operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"

static vsla_context_t* ctx = NULL;

static int autograd_test_setup(void) {
    ctx = vsla_init(NULL);
    ASSERT_NOT_NULL(ctx);
    return 1;
}

static void autograd_test_teardown(void) {
    vsla_cleanup(ctx);
}

static int test_addition_backward(void) {
    vsla_tape_t* tape = vsla_tape_new();
    ASSERT_NOT_NULL(tape);

    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

    vsla_tensor_t* inputs[] = {a, b};
    vsla_tape_record(tape, VSLA_OP_ADD, inputs, 2, c, NULL, 0);

    vsla_tensor_t* grad_c = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    double grad_c_data[] = {1, 1, 1};
    memcpy(grad_c->data, grad_c_data, sizeof(grad_c_data));
    vsla_set_gradient(tape, c, grad_c);

    vsla_backward(tape);

    vsla_tensor_t* grad_a = vsla_get_gradient(tape, a);
    vsla_tensor_t* grad_b = vsla_get_gradient(tape, b);
    ASSERT_NOT_NULL(grad_a);
    ASSERT_NOT_NULL(grad_b);

    for (uint64_t i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(1.0, ((double*)grad_a->data)[i], 1e-9);
        ASSERT_FLOAT_EQ(1.0, ((double*)grad_b->data)[i], 1e-9);
    }

    vsla_tape_free(tape);
    vsla_free(a);
    vsla_free(b);
    vsla_free(c);
    vsla_free(grad_c);
    return 1;
}

static void run_autograd_tests(void) {
    TEST_CASE("Addition Backward", test_addition_backward);
}

static const test_suite_t autograd_suite = {
    .name = "autograd",
    .setup = (int (*)(void))autograd_test_setup,
    .teardown = autograd_test_teardown,
    .run_tests = run_autograd_tests
};

void register_autograd_tests(void) {
    register_test_suite(&autograd_suite);
}