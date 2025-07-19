/**
 * @file test_unified_framework.c
 * @brief Implementation of unified test framework for VSLA operations
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L
#include "test_unified_framework.h"
#include <stdarg.h>
#include <time.h>

/* Global test state */
test_results_t g_test_results = {0, 0, 0, NULL};
test_config_t g_test_config = {VSLA_BACKEND_CPU, 0, false, 1e-9};
vsla_context_t* g_test_ctx = NULL;

bool unified_test_framework_init(vsla_backend_t backend) {
    printf("Initializing VSLA unified test framework...\n");
    printf("Backend: %s\n", backend == VSLA_BACKEND_CPU ? "CPU" : 
                          backend == VSLA_BACKEND_CUDA ? "CUDA" : 
                          backend == VSLA_BACKEND_AUTO ? "AUTO" : "UNKNOWN");
    
    /* Initialize test configuration */
    g_test_config.backend = backend;
    g_test_config.device_id = 0;
    g_test_config.verbose = false;
    g_test_config.tolerance = 1e-9;
    
    /* Initialize test results */
    g_test_results.tests_run = 0;
    g_test_results.tests_passed = 0;
    g_test_results.tests_failed = 0;
    g_test_results.current_test_name = NULL;
    
    /* Initialize VSLA context */
    vsla_config_t config = {
        .backend = backend,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_NONE,
        .enable_profiling = false,
        .verbose = g_test_config.verbose
    };
    
    g_test_ctx = vsla_init(&config);
    if (!g_test_ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return false;
    }
    
    printf("Framework initialized successfully\n\n");
    return true;
}

void unified_test_framework_cleanup(void) {
    if (g_test_ctx) {
        vsla_cleanup(g_test_ctx);
        g_test_ctx = NULL;
    }
    
    if (g_test_results.current_test_name) {
        free(g_test_results.current_test_name);
        g_test_results.current_test_name = NULL;
    }
    
    printf("\nTest framework cleanup complete\n");
}

void unified_test_case_begin(const char* name) {
    if (g_test_results.current_test_name) {
        free(g_test_results.current_test_name);
    }
    g_test_results.current_test_name = strdup(name);
    
    printf("Running test: %s... ", name);
    fflush(stdout);
    g_test_results.tests_run++;
}

void unified_test_case_end(void) {
    printf("PASSED\n");
    g_test_results.tests_passed++;
}

void unified_test_print_summary(void) {
    printf("\n=== Test Summary ===\n");
    printf("Tests run:    %d\n", g_test_results.tests_run);
    printf("Tests passed: %d\n", g_test_results.tests_passed);
    printf("Tests failed: %d\n", g_test_results.tests_failed);
    printf("Success rate: %.1f%%\n", 
           g_test_results.tests_run > 0 ? 
           (100.0 * g_test_results.tests_passed) / g_test_results.tests_run : 0.0);
    
    if (g_test_results.tests_failed == 0) {
        printf("🎉 All tests passed!\n");
    } else {
        printf("❌ %d test(s) failed\n", g_test_results.tests_failed);
    }
}

static void test_failure(const char* expr, const char* file, int line, const char* format, ...) {
    printf("FAILED\n");
    printf("    ASSERTION FAILED: %s at %s:%d\n", expr, file, line);
    
    va_list args;
    va_start(args, format);
    printf("    ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
    
    g_test_results.tests_failed++;
}

bool unified_test_assert_success(vsla_error_t result, const char* expr, const char* file, int line) {
    if (result != VSLA_SUCCESS) {
        test_failure(expr, file, line, "Expected VSLA_SUCCESS, got error: %s", vsla_error_string(result));
        return false;
    }
    return true;
}

bool unified_test_assert_error(vsla_error_t result, vsla_error_t expected, const char* expr, const char* file, int line) {
    if (result != expected) {
        test_failure(expr, file, line, "Expected error %s, got %s", 
                    vsla_error_string(expected), vsla_error_string(result));
        return false;
    }
    return true;
}

bool unified_test_assert_null(const void* ptr, const char* expr, const char* file, int line) {
    if (ptr != NULL) {
        test_failure(expr, file, line, "Expected NULL pointer");
        return false;
    }
    return true;
}

bool unified_test_assert_not_null(const void* ptr, const char* expr, const char* file, int line) {
    if (ptr == NULL) {
        test_failure(expr, file, line, "Expected non-NULL pointer");
        return false;
    }
    return true;
}

bool unified_test_assert_double_eq(double a, double b, const char* expr, const char* file, int line) {
    double diff = fabs(a - b);
    if (diff > g_test_config.tolerance) {
        test_failure(expr, file, line, "%.10f != %.10f (diff=%.2e > %.2e)", 
                    a, b, diff, g_test_config.tolerance);
        return false;
    }
    return true;
}

bool unified_test_assert_tensor_eq(const vsla_tensor_t* a, const vsla_tensor_t* b, const char* expr, const char* file, int line) {
    if (!a || !b) {
        test_failure(expr, file, line, "One or both tensors are NULL");
        return false;
    }
    
    /* Check metadata */
    if (vsla_get_rank(a) != vsla_get_rank(b)) {
        test_failure(expr, file, line, "Tensor ranks differ: %d != %d", 
                    vsla_get_rank(a), vsla_get_rank(b));
        return false;
    }
    
    if (vsla_get_model(a) != vsla_get_model(b)) {
        test_failure(expr, file, line, "Tensor models differ");
        return false;
    }
    
    if (vsla_get_dtype(a) != vsla_get_dtype(b)) {
        test_failure(expr, file, line, "Tensor data types differ");
        return false;
    }
    
    if (!vsla_shape_equal(a, b)) {
        test_failure(expr, file, line, "Tensor shapes differ");
        return false;
    }
    
    /* Check data equality */
    uint64_t n = vsla_numel(a);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t indices[8]; /* Support up to 8D tensors */
        uint64_t temp = i;
        uint8_t rank = vsla_get_rank(a);
        
        /* Convert linear index to multi-dimensional indices */
        uint64_t shape[8];
        vsla_get_shape(a, shape);
        
        for (int d = rank - 1; d >= 0; d--) {
            indices[d] = temp % shape[d];
            temp /= shape[d];
        }
        
        double val_a, val_b;
        if (vsla_get_f64(g_test_ctx, a, indices, &val_a) != VSLA_SUCCESS ||
            vsla_get_f64(g_test_ctx, b, indices, &val_b) != VSLA_SUCCESS) {
            test_failure(expr, file, line, "Failed to read tensor values at index %llu", (unsigned long long)i);
            return false;
        }
        
        if (fabs(val_a - val_b) > g_test_config.tolerance) {
            test_failure(expr, file, line, "Tensor values differ at index %llu: %.10f != %.10f", 
                        (unsigned long long)i, val_a, val_b);
            return false;
        }
    }
    
    return true;
}

bool unified_test_assert_shape_eq(const vsla_tensor_t* tensor, const uint64_t* expected_shape, uint8_t rank, const char* file, int line) {
    if (!tensor || !expected_shape) {
        test_failure("shape check", file, line, "NULL tensor or expected shape");
        return false;
    }
    
    if (vsla_get_rank(tensor) != rank) {
        test_failure("shape check", file, line, "Rank mismatch: %d != %d", 
                    vsla_get_rank(tensor), rank);
        return false;
    }
    
    uint64_t actual_shape[8];
    vsla_get_shape(tensor, actual_shape);
    
    for (uint8_t i = 0; i < rank; i++) {
        if (actual_shape[i] != expected_shape[i]) {
            test_failure("shape check", file, line, "Shape mismatch at dimension %d: %llu != %llu", 
                        i, (unsigned long long)actual_shape[i], (unsigned long long)expected_shape[i]);
            return false;
        }
    }
    
    return true;
}

/* Utility functions */

vsla_tensor_t* unified_test_create_tensor_1d(uint64_t size, vsla_model_t model, vsla_dtype_t dtype) {
    uint64_t shape[] = {size};
    return vsla_tensor_create(g_test_ctx, 1, shape, model, dtype);
}

vsla_tensor_t* unified_test_create_tensor_2d(uint64_t rows, uint64_t cols, vsla_model_t model, vsla_dtype_t dtype) {
    uint64_t shape[] = {rows, cols};
    return vsla_tensor_create(g_test_ctx, 2, shape, model, dtype);
}

void unified_test_fill_tensor_sequential(vsla_tensor_t* tensor, double start_value) {
    uint64_t n = vsla_numel(tensor);
    uint8_t rank = vsla_get_rank(tensor);
    uint64_t shape[8];
    vsla_get_shape(tensor, shape);
    
    for (uint64_t i = 0; i < n; i++) {
        uint64_t indices[8];
        uint64_t temp = i;
        
        /* Convert linear index to multi-dimensional indices */
        for (int d = rank - 1; d >= 0; d--) {
            indices[d] = temp % shape[d];
            temp /= shape[d];
        }
        
        vsla_set_f64(g_test_ctx, tensor, indices, start_value + i);
    }
}

void unified_test_fill_tensor_random(vsla_tensor_t* tensor, double min_val, double max_val) {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }
    
    uint64_t n = vsla_numel(tensor);
    uint8_t rank = vsla_get_rank(tensor);
    uint64_t shape[8];
    vsla_get_shape(tensor, shape);
    
    for (uint64_t i = 0; i < n; i++) {
        uint64_t indices[8];
        uint64_t temp = i;
        
        /* Convert linear index to multi-dimensional indices */
        for (int d = rank - 1; d >= 0; d--) {
            indices[d] = temp % shape[d];
            temp /= shape[d];
        }
        
        double val = min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
        vsla_set_f64(g_test_ctx, tensor, indices, val);
    }
}

void unified_test_print_tensor(const vsla_tensor_t* tensor, const char* name) {
    if (!g_test_config.verbose) return;
    
    printf("Tensor %s:\n", name);
    printf("  Rank: %d\n", vsla_get_rank(tensor));
    printf("  Model: %d\n", vsla_get_model(tensor));
    printf("  Dtype: %d\n", vsla_get_dtype(tensor));
    printf("  Elements: %llu\n", (unsigned long long)vsla_numel(tensor));
    
    uint8_t rank = vsla_get_rank(tensor);
    uint64_t shape[8];
    vsla_get_shape(tensor, shape);
    
    printf("  Shape: [");
    for (uint8_t i = 0; i < rank; i++) {
        printf("%llu", (unsigned long long)shape[i]);
        if (i < rank - 1) printf(", ");
    }
    printf("]\n");
}