/**
 * @file test_gpu.c
 * @brief Tests for GPU operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

static vsla_context_t* ctx = NULL;

static int gpu_test_setup(void) {
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CUDA,
    };
    ctx = vsla_init(&config);
    if (!ctx) return 0; // Skip tests if no CUDA context
    return 1;
}

static void gpu_test_teardown(void) {
    if (ctx) {
        vsla_cleanup(ctx);
    }
}

static int test_gpu_addition(void) {
    if (!ctx) return 1; // Skip if no CUDA context
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    double a_data[] = {1, 2, 3};
    double b_data[] = {4, 5, 6};
    memcpy(vsla_tensor_data_mut(a, NULL), a_data, sizeof(a_data));
    memcpy(vsla_tensor_data_mut(b, NULL), b_data, sizeof(b_data));

    ASSERT_EQ(VSLA_SUCCESS, vsla_add(ctx, result, a, b));

    const double* result_data = vsla_tensor_data(result, NULL);
    double expected[] = {5, 7, 9};
    for (uint64_t i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(expected[i], result_data[i], 1e-9);
    }

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    return 1;
}

static void run_gpu_tests(void) {
    TEST_CASE("GPU Addition", test_gpu_addition);
}

static const test_suite_t gpu_suite = {
    .name = "gpu",
    .setup = (int (*)(void))gpu_test_setup,
    .teardown = gpu_test_teardown,
    .run_tests = run_gpu_tests
};

void register_gpu_tests(void) {
    register_test_suite(&gpu_suite);
}