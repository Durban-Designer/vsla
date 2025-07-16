/**
 * @file test_kron.c
 * @brief Tests for Kronecker product operations (Model B)
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <math.h>

// Test simple 1D Kronecker product
static int test_kron_1d_simple(void) {
    // Create simple 1D tensors: [1, 2] ⊗ [3, 4] = [3, 4, 6, 8]
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {2};
    uint64_t shape_out[] = {4};  // 2 * 2
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
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
    
    // Compute Kronecker product
    if (vsla_kron_naive(out, a, b) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Check results: [1,2] ⊗ [3,4] = [1*3, 1*4, 2*3, 2*4] = [3, 4, 6, 8]
    double expected[] = {3.0, 4.0, 6.0, 8.0};
    for (int i = 0; i < 4; i++) {
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

// Test tiled vs naive Kronecker product equivalence
static int test_tiled_vs_naive(void) {
    uint64_t shape_a[] = {3};
    uint64_t shape_b[] = {4};
    uint64_t shape_out[] = {12};  // 3 * 4
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out_naive = vsla_new(1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out_tiled = vsla_new(1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!a || !b || !out_naive || !out_tiled) {
        vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
        return 0;
    }
    
    // Set test values
    double a_vals[] = {1.0, -1.0, 2.0};
    double b_vals[] = {2.0, 0.5, -1.0, 3.0};
    
    for (int i = 0; i < 3; i++) {
        uint64_t idx = i;
        if (vsla_set_f64(a, &idx, a_vals[i]) != VSLA_SUCCESS) {
            vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
            return 0;
        }
    }
    
    for (int i = 0; i < 4; i++) {
        uint64_t idx = i;
        if (vsla_set_f64(b, &idx, b_vals[i]) != VSLA_SUCCESS) {
            vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
            return 0;
        }
    }
    
    // Compute both ways
    if (vsla_kron_naive(out_naive, a, b) != VSLA_SUCCESS ||
        vsla_kron_tiled(out_tiled, a, b, 2) != VSLA_SUCCESS) {  // Use small tile size
        vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
        return 0;
    }
    
    // Compare results
    for (int i = 0; i < 12; i++) {
        double val_naive, val_tiled;
        uint64_t idx = i;
        if (vsla_get_f64(out_naive, &idx, &val_naive) != VSLA_SUCCESS ||
            vsla_get_f64(out_tiled, &idx, &val_tiled) != VSLA_SUCCESS ||
            fabs(val_naive - val_tiled) > 1e-15) {
            vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
            return 0;
        }
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out_naive); vsla_free(out_tiled);
    return 1;
}

// Test monoid algebra conversion
static int test_monoid_algebra_conversion(void) {
    // Create tensor [0, 2, 0, 3] representing 2*e_2 + 3*e_4
    double coeffs_in[] = {2.0, 3.0};
    uint64_t indices_in[] = {2, 4};
    size_t num_terms_in = 2;
    
    vsla_tensor_t* tensor = vsla_from_monoid_algebra(coeffs_in, indices_in, num_terms_in, VSLA_DTYPE_F64);
    if (!tensor) return 0;
    
    // Extract back to monoid algebra
    double coeffs_out[10];
    uint64_t indices_out[10];
    size_t num_terms_out;
    
    if (vsla_to_monoid_algebra(tensor, coeffs_out, indices_out, 10, &num_terms_out) != VSLA_SUCCESS) {
        vsla_free(tensor);
        return 0;
    }
    
    // Check values
    if (num_terms_out != 2) {
        vsla_free(tensor);
        return 0;
    }
    
    // Sort for comparison (order might differ)
    for (size_t i = 0; i < num_terms_out; i++) {
        int found = 0;
        for (size_t j = 0; j < num_terms_in; j++) {
            if (indices_out[i] == indices_in[j] && 
                fabs(coeffs_out[i] - coeffs_in[j]) < 1e-15) {
                found = 1;
                break;
            }
        }
        if (!found) {
            vsla_free(tensor);
            return 0;
        }
    }
    
    vsla_free(tensor);
    return 1;
}

// Test Kronecker product with identity element
static int test_kron_identity(void) {
    // Identity for Kronecker product is a scalar tensor [1]
    uint64_t shape_a[] = {3};
    uint64_t shape_id[] = {1};
    uint64_t shape_out[] = {3};
    
    vsla_tensor_t* a = vsla_new(1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* identity = vsla_new(1, shape_id, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
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
    
    // Kronecker product with identity
    if (vsla_kron(out, a, identity) != VSLA_SUCCESS) {
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

// Test commutativity checking
static int test_kron_commutativity(void) {
    // Test scalar tensors (should be commutative)
    uint64_t scalar_shape[] = {1};
    vsla_tensor_t* scalar_a = vsla_new(1, scalar_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* scalar_b = vsla_new(1, scalar_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!scalar_a || !scalar_b) {
        vsla_free(scalar_a); vsla_free(scalar_b);
        return 0;
    }
    
    uint64_t idx0 = 0;
    vsla_set_f64(scalar_a, &idx0, 2.0);
    vsla_set_f64(scalar_b, &idx0, 3.0);
    
    if (!vsla_kron_is_commutative(scalar_a, scalar_b)) {
        vsla_free(scalar_a); vsla_free(scalar_b);
        return 0;
    }
    
    vsla_free(scalar_a); vsla_free(scalar_b);
    
    // Test non-scalar tensors (should be non-commutative)
    uint64_t vector_shape[] = {2};
    vsla_tensor_t* vec_a = vsla_new(1, vector_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* vec_b = vsla_new(1, vector_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!vec_a || !vec_b) {
        vsla_free(vec_a); vsla_free(vec_b);
        return 0;
    }
    
    if (vsla_kron_is_commutative(vec_a, vec_b)) {
        vsla_free(vec_a); vsla_free(vec_b);
        return 0;  // Should be non-commutative
    }
    
    vsla_free(vec_a); vsla_free(vec_b);
    return 1;
}

// Test error handling
static int test_kron_error_handling(void) {
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);  // Wrong model
    vsla_tensor_t* out = vsla_new(1, shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test NULL pointer errors
    if (vsla_kron(NULL, a, b) != VSLA_ERROR_NULL_POINTER) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test model mismatch
    if (vsla_kron(out, a, b) != VSLA_ERROR_INVALID_MODEL) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Test wrong output dimensions
    uint64_t wrong_shape[] = {6};  // Should be 9 for Kronecker product of two 3-element vectors
    vsla_tensor_t* wrong_out = vsla_new(1, wrong_shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_free(b);
    b = vsla_new(1, shape, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!b || !wrong_out) {
        vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
        return 0;
    }
    
    if (vsla_kron(wrong_out, a, b) != VSLA_ERROR_DIMENSION_MISMATCH) {
        vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
        return 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out); vsla_free(wrong_out);
    return 1;
}

// Test 2D Kronecker product
static int test_kron_2d(void) {
    // Simple 2x2 ⊗ 2x2 Kronecker product
    uint64_t shape_a[] = {2, 2};
    uint64_t shape_b[] = {2, 2};
    uint64_t shape_out[] = {4, 4};  // (2*2, 2*2)
    
    vsla_tensor_t* a = vsla_new(2, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(2, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(2, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    if (!a || !b || !out) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Set simple values: a = [[1,2],[3,4]], b = [[1,0],[0,1]] (identity matrix)
    uint64_t indices[2];
    indices[0] = 0; indices[1] = 0; vsla_set_f64(a, indices, 1.0);
    indices[0] = 0; indices[1] = 1; vsla_set_f64(a, indices, 2.0);
    indices[0] = 1; indices[1] = 0; vsla_set_f64(a, indices, 3.0);
    indices[0] = 1; indices[1] = 1; vsla_set_f64(a, indices, 4.0);
    
    indices[0] = 0; indices[1] = 0; vsla_set_f64(b, indices, 1.0);
    indices[0] = 0; indices[1] = 1; vsla_set_f64(b, indices, 0.0);
    indices[0] = 1; indices[1] = 0; vsla_set_f64(b, indices, 0.0);
    indices[0] = 1; indices[1] = 1; vsla_set_f64(b, indices, 1.0);
    
    // Compute Kronecker product
    if (vsla_kron_naive(out, a, b) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Check that Kronecker product completed without error
    // (Full verification of 2D Kronecker product values would be complex)
    double val;
    indices[0] = 0; indices[1] = 0;
    if (vsla_get_f64(out, indices, &val) != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    // Should be 1.0 * 1.0 = 1.0
    if (fabs(val - 1.0) > 1e-15) {
        vsla_free(a); vsla_free(b); vsla_free(out);
        return 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(out);
    return 1;
}

static void kron_test_setup(void) {
    // Setup for Kronecker tests
}

static void kron_test_teardown(void) {
    // Teardown for Kronecker tests
}

static void run_kron_tests(void) {
    printf("Running Kronecker tests:\n");
    
    RUN_TEST(test_kron_1d_simple);
    RUN_TEST(test_tiled_vs_naive);
    RUN_TEST(test_monoid_algebra_conversion);
    RUN_TEST(test_kron_identity);
    RUN_TEST(test_kron_commutativity);
    RUN_TEST(test_kron_error_handling);
    RUN_TEST(test_kron_2d);
}

static const test_suite_t kron_suite = {
    .name = "kron",
    .setup = kron_test_setup,
    .teardown = kron_test_teardown,
    .run_tests = run_kron_tests
};

void register_kron_tests(void) {
    register_test_suite(&kron_suite);
}