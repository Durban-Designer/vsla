/**
 * @file test_unified_backend.c
 * @brief Modern unified backend test using clean API
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_DOUBLE_EQ(a, b, eps) assert(fabs((a) - (b)) < (eps))

int test_backend_creation(void) {
    printf("ENTERING: test_backend_creation\n");

    printf("Testing backend creation...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    ASSERT_EQ(ctx != NULL, 1);
    
    vsla_backend_t backend;
    char device_name[64];
    double memory_gb;
    vsla_error_t err = vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    printf("  ✅ Backend: %s\n", device_name);
    
    vsla_cleanup(ctx);
    return 1;
}

int test_tensor_operations(void) {
    printf("ENTERING: test_tensor_operations\n");
    printf("Testing tensor operations...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    ASSERT_EQ(ctx != NULL, 1);
    
    // Create test tensors
    uint64_t shape[] = {5};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_EQ(a != NULL, 1);
    ASSERT_EQ(b != NULL, 1);
    ASSERT_EQ(out != NULL, 1);
    
    // Fill tensors
    vsla_error_t err = vsla_fill(ctx, a, 2.0);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    err = vsla_fill(ctx, b, 3.0);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    // Test addition
    err = vsla_add(ctx, out, a, b);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    // Verify result using sum
    double sum_result = 0.0;
    err = vsla_sum(ctx, out, &sum_result);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    double expected_sum = 5.0 * 5.0; // (2.0 + 3.0) * 5 elements
    ASSERT_DOUBLE_EQ(sum_result, expected_sum, 1e-10);
    
    printf("  ✅ Addition: %.1f (expected %.1f)\n", sum_result, expected_sum);
    
    // Test scaling
    err = vsla_scale(ctx, out, a, 2.5);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    err = vsla_sum(ctx, out, &sum_result);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    expected_sum = 2.0 * 2.5 * 5.0; // 2.0 * 2.5 * 5 elements
    ASSERT_DOUBLE_EQ(sum_result, expected_sum, 1e-10);
    
    printf("  ✅ Scaling: %.1f (expected %.1f)\n", sum_result, expected_sum);
    
    // Test norm
    double norm_result = 0.0;
    err = vsla_norm(ctx, a, &norm_result);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    double expected_norm = sqrt(2.0 * 2.0 * 5.0); // sqrt(sum of squares)
    ASSERT_DOUBLE_EQ(norm_result, expected_norm, 1e-10);
    
    printf("  ✅ Norm: %.3f (expected %.3f)\n", norm_result, expected_norm);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_cleanup(ctx);
    
    return 1;
}

int test_backend_selection(void) {
    printf("ENTERING: test_backend_selection\n");
    printf("Testing backend selection...\n");
    
    // Test CPU backend explicitly
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = -1,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_NONE,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    ASSERT_EQ(ctx != NULL, 1);
    
    vsla_backend_t backend;
    char device_name[64];
    double memory_gb;
    vsla_error_t err = vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    ASSERT_EQ(err, VSLA_SUCCESS);
    ASSERT_EQ(backend, VSLA_BACKEND_CPU);
    
    printf("  ✅ Explicit CPU backend: %s\n", device_name);
    
    vsla_cleanup(ctx);
    return 1;
}

int main(void) {
    printf("🧪 VSLA Unified Backend Test Suite\n");
    printf("===================================\n");
    
    int tests_passed = 0;
    int total_tests = 0;
    
    total_tests++;
    if (test_backend_creation()) tests_passed++;
    
    total_tests++;
    if (test_tensor_operations()) tests_passed++;
    
    total_tests++;
    if (test_backend_selection()) tests_passed++;
    
    printf("\n📊 Results: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("✅ All tests PASSED!\n");
        return 0;
    } else {
        printf("❌ Some tests FAILED!\n");
        return 1;
    }
}