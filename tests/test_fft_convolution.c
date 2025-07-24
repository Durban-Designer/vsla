/**
 * @file test_fft_convolution.c
 * @brief Test FFT convolution implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

#define EPSILON 1e-10

static int test_fft_convolution_sizes() {
    printf("Testing FFT convolution for various sizes...\n");
    
    vsla_error_t err;
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize context\n");
        return 1;
    }
    
    // Test sizes - both small (direct) and large (FFT)
    size_t test_cases[][2] = {
        {3, 4},      // 3 * 4 = 12 ops (direct)
        {10, 10},    // 10 * 10 = 100 ops (direct)
        {32, 32},    // 32 * 32 = 1024 ops (at threshold)
        {50, 50},    // 50 * 50 = 2500 ops (FFT)
        {100, 100},  // 100 * 100 = 10000 ops (FFT)
        {256, 256}   // Large size (FFT)
    };
    
    for (size_t tc = 0; tc < sizeof(test_cases)/sizeof(test_cases[0]); tc++) {
        size_t m = test_cases[tc][0];
        size_t n = test_cases[tc][1];
        size_t out_size = m + n - 1;
        
        printf("  Testing m=%zu, n=%zu (m*n=%zu)...\n", m, n, m*n);
        
        // Create test vectors
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){m}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){n}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!a || !b || !out) {
            printf("    Failed to create tensors\n");
            return 1;
        }
        
        // Fill tensors with zeros
        vsla_fill(ctx, a, 0.0);
        vsla_fill(ctx, b, 0.0);
        vsla_fill(ctx, out, 0.0);
        
        // Fill with test data
        for (size_t i = 0; i < m; i++) {
            uint64_t idx[] = {i};
            double val = sin(2.0 * M_PI * i / m);
            vsla_set_f64(ctx, a, idx, val);
        }
        for (size_t i = 0; i < n; i++) {
            uint64_t idx[] = {i};
            double val = cos(2.0 * M_PI * i / n);
            vsla_set_f64(ctx, b, idx, val);
        }
        
        err = vsla_conv(ctx, out, a, b);
        if (err != VSLA_SUCCESS) {
            printf("    Convolution failed: %d\n", err);
            return 1;
        }
        
        // Basic verification - check output is not all zeros
        double sum = 0.0;
        double max_val = 0.0;
        
        for (size_t i = 0; i < out_size; i++) {
            uint64_t idx[] = {i};
            double val;
            vsla_get_f64(ctx, out, idx, &val);
            sum += fabs(val);
            if (fabs(val) > max_val) {
                max_val = fabs(val);
            }
        }
        
        if (sum < EPSILON) {
            printf("    Output is all zeros!\n");
            return 1;
        }
        
        printf("    PASSED (sum=%.3f, max=%.3f)\n", sum, max_val);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
    }
    
    vsla_cleanup(ctx);
    return 0;
}

static int test_fft_correctness() {
    printf("Testing FFT convolution correctness...\n");
    
    vsla_error_t err;
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize context\n");
        return 1;
    }
    
    // Test 1: Convolve with identity
    {
        printf("  Testing convolution with identity...\n");
        size_t m = 100;
        size_t n = 1;
        size_t out_size = m + n - 1;
        
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){m}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){n}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!a || !b || !out) {
            printf("    Failed to create tensors\n");
            return 1;
        }
        
        // Fill with zeros
        vsla_fill(ctx, a, 0.0);
        vsla_fill(ctx, b, 0.0);
        vsla_fill(ctx, out, 0.0);
        
        // Fill a with pattern, b with identity
        for (size_t i = 0; i < m; i++) {
            uint64_t idx[] = {i};
            vsla_set_f64(ctx, a, idx, (double)i);
        }
        uint64_t idx_zero[] = {0};
        vsla_set_f64(ctx, b, idx_zero, 1.0);
        
        err = vsla_conv(ctx, out, a, b);
        if (err != VSLA_SUCCESS) {
            printf("    Convolution failed: %d\n", err);
            return 1;
        }
        
        // Result should equal input a
        for (size_t i = 0; i < m; i++) {
            uint64_t idx[] = {i};
            double out_val, a_val;
            vsla_get_f64(ctx, out, idx, &out_val);
            vsla_get_f64(ctx, a, idx, &a_val);
            if (fabs(out_val - a_val) > EPSILON) {
                printf("    Identity convolution failed at index %zu: expected %g, got %g\n",
                       i, a_val, out_val);
                return 1;
            }
        }
        
        printf("    PASSED\n");
        
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
    }
    
    // Test 2: Known convolution result
    {
        printf("  Testing known convolution result...\n");
        size_t m = 3;
        size_t n = 3;
        size_t out_size = m + n - 1;
        
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){m}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){n}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!a || !b || !out) {
            printf("    Failed to create tensors\n");
            return 1;
        }
        
        // Fill with zeros
        vsla_fill(ctx, a, 0.0);
        vsla_fill(ctx, b, 0.0);
        vsla_fill(ctx, out, 0.0);
        
        // a = [1, 2, 3], b = [1, 0, -1]
        uint64_t idx0[] = {0};
        uint64_t idx1[] = {1};
        uint64_t idx2[] = {2};
        
        vsla_set_f64(ctx, a, idx0, 1.0);
        vsla_set_f64(ctx, a, idx1, 2.0);
        vsla_set_f64(ctx, a, idx2, 3.0);
        
        vsla_set_f64(ctx, b, idx0, 1.0);
        vsla_set_f64(ctx, b, idx1, 0.0);
        vsla_set_f64(ctx, b, idx2, -1.0);
        
        err = vsla_conv(ctx, out, a, b);
        if (err != VSLA_SUCCESS) {
            printf("    Convolution failed: %d\n", err);
            return 1;
        }
        
        // Expected result: [1, 2, 2, -2, -3]
        double expected[] = {1.0, 2.0, 2.0, -2.0, -3.0};
        
        for (size_t i = 0; i < out_size; i++) {
            uint64_t idx[] = {i};
            double val;
            vsla_get_f64(ctx, out, idx, &val);
            if (fabs(val - expected[i]) > EPSILON) {
                printf("    Mismatch at index %zu: expected %g, got %g\n",
                       i, expected[i], val);
                return 1;
            }
        }
        
        printf("    PASSED\n");
        
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
    }
    
    vsla_cleanup(ctx);
    return 0;
}

static int test_fft_edge_cases() {
    printf("Testing FFT convolution edge cases...\n");
    
    vsla_error_t err;
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize context\n");
        return 1;
    }
    
    // Test with power-of-2 sizes
    size_t sizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
    
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        size_t size = sizes[i];
        size_t out_size = size + size - 1;
        
        printf("  Testing size=%zu (power of 2)...\n", size);
        
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!a || !b || !out) {
            printf("    Failed to create tensors\n");
            return 1;
        }
        
        // Fill with zeros
        vsla_fill(ctx, a, 0.0);
        vsla_fill(ctx, b, 0.0);
        vsla_fill(ctx, out, 0.0);
        
        // Simple test pattern
        for (size_t j = 0; j < size; j++) {
            uint64_t idx[] = {j};
            double a_val = (j == 0) ? 1.0 : 0.0; // Impulse
            double b_val = 1.0 / size;           // Uniform
            vsla_set_f64(ctx, a, idx, a_val);
            vsla_set_f64(ctx, b, idx, b_val);
        }
        
        err = vsla_conv(ctx, out, a, b);
        if (err != VSLA_SUCCESS) {
            printf("    Convolution failed: %d\n", err);
            return 1;
        }
        
        // Verify result
        double sum = 0.0;
        for (size_t j = 0; j < out_size; j++) {
            uint64_t idx[] = {j};
            double val;
            vsla_get_f64(ctx, out, idx, &val);
            sum += val;
        }
        
        // Sum should be approximately 1.0
        if (fabs(sum - 1.0) > EPSILON * size) {
            printf("    Sum check failed: expected 1.0, got %g\n", sum);
            return 1;
        }
        
        printf("    PASSED\n");
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
    }
    
    vsla_cleanup(ctx);
    return 0;
}

static int test_fft_performance() {
    printf("Testing FFT convolution performance (timing)...\n");
    
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize context\n");
        return 1;
    }
    
    // Large size to ensure FFT path
    size_t m = 10000;
    size_t n = 10000;
    size_t out_size = m + n - 1;
    
    printf("  Creating large vectors (m=%zu, n=%zu)...\n", m, n);
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){m}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){n}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        printf("    Failed to create tensors\n");
        return 1;
    }
    
    // Fill with zeros first
    vsla_fill(ctx, a, 0.0);
    vsla_fill(ctx, b, 0.0);
    vsla_fill(ctx, out, 0.0);
    
    // Fill with random data
    for (size_t i = 0; i < m; i++) {
        uint64_t idx[] = {i};
        double val = (double)rand() / RAND_MAX - 0.5;
        vsla_set_f64(ctx, a, idx, val);
    }
    for (size_t i = 0; i < n; i++) {
        uint64_t idx[] = {i};
        double val = (double)rand() / RAND_MAX - 0.5;
        vsla_set_f64(ctx, b, idx, val);
    }
    
    printf("  Running FFT convolution...\n");
    
    // Time the convolution
    clock_t start = clock();
    vsla_error_t err = vsla_conv(ctx, out, a, b);
    clock_t end = clock();
    
    if (err != VSLA_SUCCESS) {
        printf("    Convolution failed: %d\n", err);
        return 1;
    }
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("    PASSED (time: %.3f seconds)\n", time_taken);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_cleanup(ctx);
    
    return 0;
}

int main() {
    printf("=== VSLA FFT Convolution Tests ===\n\n");
    
    int failed = 0;
    
    failed += test_fft_convolution_sizes();
    printf("\n");
    
    failed += test_fft_correctness();
    printf("\n");
    
    failed += test_fft_edge_cases();
    printf("\n");
    
    failed += test_fft_performance();
    printf("\n");
    
    if (failed == 0) {
        printf("All FFT convolution tests PASSED!\n");
    } else {
        printf("Some tests FAILED!\n");
    }
    
    return failed;
}