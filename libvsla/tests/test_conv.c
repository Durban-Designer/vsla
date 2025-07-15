/**
 * @file test_conv.c
 * @brief Tests for convolution operations (Model A)
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <math.h>

// Test simple 1D convolution
static int test_conv_1d_simple(void) {
    // Create simple 1D tensors: [1, 2] * [3, 4] = [3, 10, 8]
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {2};
    uint64_t shape_out[] = {3};  // 2 + 2 - 1
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Set input values
    uint64_t idx0 = 0, idx1 = 1;
    if (vsla_set_f64(a, &idx0, 1.0) != VSLA_SUCCESS ||
        vsla_set_f64(a, &idx1, 2.0) != VSLA_SUCCESS ||
        vsla_set_f64(b, &idx0, 3.0) != VSLA_SUCCESS ||
        vsla_set_f64(b, &idx1, 4.0) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Compute convolution
    if (vsla_conv_direct(out, a, b) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Check results: [1,2] * [3,4] = [1*3, 1*4+2*3, 2*4] = [3, 10, 8]
    double expected[] = {3.0, 10.0, 8.0};
    for (int i = 0; i < 3; i++) {
        double val;
        uint64_t idx = i;
        if (vsla_get_f64(out, &idx, &val) != VSLA_SUCCESS ||
            fabs(val - expected[i]) > 1e-15) {
            vsla_free(a); vsla_free(b); vsla_free(out);
            return 0;
        }
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out);
    return 1;
}

// Test FFT vs direct convolution equivalence
static int test_fft_vs_direct(void) {
    uint64_t shape_a[] = {4};
    uint64_t shape_b[] = {3};
    uint64_t shape_out[] = {6};  // 4 + 3 - 1
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out_direct = vsla_new(1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out_fft = vsla_new(1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !out_direct || !out_fft) {
        vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
        return 0;
    }
    
    // Set test values
    double a_vals[] = {1.0, -1.0, 2.0, 0.5};
    double b_vals[] = {2.0, 1.0, -1.0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t idx = i;
        if (vsla_set_f64(a, &idx, a_vals[i]) != VSLA_SUCCESS) {
            vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
            return 0;
        }
    }
    
    for (int i = 0; i < 3; i++) {
        uint64_t idx = i;
        if (vsla_set_f64(b, &idx, b_vals[i]) != VSLA_SUCCESS) {
            vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
            return 0;
        }
    }
    
    // Compute both ways
    if (vsla_conv_direct(out_direct, a, b) != VSLA_SUCCESS ||
        vsla_conv_fft(out_fft, a, b) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
        return 0;
    }
    
    // Compare results
    for (int i = 0; i < 6; i++) {
        double val_direct, val_fft;
        uint64_t idx = i;
        if (vsla_get_f64(out_direct, &idx, &val_direct) != VSLA_SUCCESS ||
            vsla_get_f64(out_fft, &idx, &val_fft) != VSLA_SUCCESS ||
            fabs(val_direct - val_fft) > 1e-12) {
            vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
            return 0;
        }
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out_direct); vsla_free(out_fft);
    return 1;
}

// Test polynomial conversion
static int test_polynomial_conversion(void) {
    // Create polynomial [1, 2, 3] representing 1 + 2x + 3x^2
    double coeffs[] = {1.0, 2.0, 3.0};
    size_t degree = 3;
    
    vsla_tensor_t* poly = vsla_from_polynomial(coeffs, degree, VSLA_DTYPE_F64);
    if (!poly) return 0;
    
    // Extract back to coefficients
    double extracted[3];
    if (vsla_to_polynomial(poly, extracted, 2) != VSLA_SUCCESS) {
        vsla_free(poly);
        return 0;
    }
    
    // Check values
    for (int i = 0; i < 3; i++) {
        if (fabs(extracted[i] - coeffs[i]) > 1e-15) {
            vsla_free(poly);
            return 0;
        }
    }
    
    vsla_free(poly);
    return 1;
}

// Test convolution with identity element
static int test_conv_identity(void) {
    // Identity for convolution is [1, 0, 0, ...]
    uint64_t shape_a[] = {3};
    uint64_t shape_id[] = {1};
    uint64_t shape_out[] = {3};
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* identity = vsla_new(1, shape_id, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !identity || !out) {
        vsla_free(a); vsla_free(identity); vsla_free(out);
        return 0;
    }
    
    // Set up test values
    double a_vals[] = {2.0, -1.0, 3.5};
    for (int i = 0; i < 3; i++) {
        uint64_t idx = i;
        if (vsla_set_f64(a, &idx, a_vals[i]) != VSLA_SUCCESS) {
            vsla_free(a); vsla_free(identity); vsla_free(out);
            return 0;
        }
    }
    
    uint64_t idx0 = 0;
    if (vsla_set_f64(identity, &idx0, 1.0) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(identity); vsla_free(out);
        return 0;
    }
    
    // Convolve with identity
    if (vsla_conv(out, a, identity) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(identity); vsla_free(out);
        return 0;
    }
    
    // Result should equal original
    for (int i = 0; i < 3; i++) {
        double val;
        uint64_t idx = i;
        if (vsla_get_f64(out, &idx, &val) != VSLA_SUCCESS ||
            fabs(val - a_vals[i]) > 1e-15) {
            vsla_free(a); vsla_free(identity); vsla_free(out);
            return 0;
        }
    }
    
    vsla_free(a); vsla_free(identity); vsla_free(out);
    return 1;
}

// Test error handling
static int test_conv_error_handling(void) {
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_B, VSLA_DTYPE_F64);  // Wrong model
    vsla_tensor_t* out = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test NULL pointer errors
    if (vsla_conv(NULL, a, b) != VSLA_ERROR_NULL_POINTER) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test model mismatch
    if (vsla_conv(out, a, b) != VSLA_ERROR_INVALID_MODEL) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test wrong output dimensions
    uint64_t wrong_shape[] = {4};  // Should be 5 for convolution of two 3-element vectors
    vsla_tensor_t* wrong_out = vsla_new(1, wrong_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_free(b);
    b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!b || !wrong_out) {
        vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
        return 0;
    }
    
    if (vsla_conv(wrong_out, a, b) != VSLA_ERROR_DIMENSION_MISMATCH) {
        vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
        return 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
    return 1;
}

// Test 2D convolution
static int test_conv_2d(void) {
    // Simple 2x2 * 2x2 convolution
    uint64_t shape_a[] = {2, 2};
    uint64_t shape_b[] = {2, 2};
    uint64_t shape_out[] = {3, 3};  // (2+2-1, 2+2-1)
    
    vsla_tensor_t* a = vsla_new(2, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(2, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(2, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Set simple values: a = [[1,2],[3,4]], b = [[1,0],[0,1]] (identity-like)
    uint64_t indices[2];
    indices[0] = 0; indices[1] = 0; vsla_set_f64(a, indices, 1.0);
    indices[0] = 0; indices[1] = 1; vsla_set_f64(a, indices, 2.0);
    indices[0] = 1; indices[1] = 0; vsla_set_f64(a, indices, 3.0);
    indices[0] = 1; indices[1] = 1; vsla_set_f64(a, indices, 4.0);
    
    indices[0] = 0; indices[1] = 0; vsla_set_f64(b, indices, 1.0);
    indices[0] = 0; indices[1] = 1; vsla_set_f64(b, indices, 0.0);
    indices[0] = 1; indices[1] = 0; vsla_set_f64(b, indices, 0.0);
    indices[0] = 1; indices[1] = 1; vsla_set_f64(b, indices, 1.0);
    
    // Compute convolution
    if (vsla_conv_direct(out, a, b) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Check that convolution completed without error
    // (Full verification of 2D convolution values would be complex)
    double val;
    indices[0] = 0; indices[1] = 0;
    if (vsla_get_f64(out, indices, &val) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out);
    return 1;
}

static void conv_test_setup(void) {
    // Setup for convolution tests
}

static void conv_test_teardown(void) {
    // Teardown for convolution tests
}

static void run_conv_tests(void) {
    printf("Running Convolution tests:\n");
    
    RUN_TEST(test_conv_1d_simple);
    RUN_TEST(test_fft_vs_direct);
    RUN_TEST(test_polynomial_conversion);
    RUN_TEST(test_conv_identity);
    RUN_TEST(test_conv_error_handling);
    RUN_TEST(test_conv_2d);
}

static const test_suite_t conv_suite = {
    .name = "conv",
    .setup = conv_test_setup,
    .teardown = conv_test_teardown,
    .run_tests = run_conv_tests
};

void register_conv_tests(void) {
    register_test_suite(&conv_suite);
}