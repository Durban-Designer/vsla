/**
 * @file test_vsla_spec_validation.c
 * @brief Validates CPU backend implementation against VSLA v3.1 specification
 * 
 * Tests focus on:
 * - Ambient promotion semantics (Section 4.1)
 * - Shrinking to minimal representative (Section 6)
 * - Model A (convolution) correctness (Section 4.3)
 * - Model B (Kronecker) correctness (Section 4.4)
 * - Mathematical properties (associativity, distributivity, identity)
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-12
#define ASSERT_DOUBLE_EQ(a, b) assert(fabs((a) - (b)) < EPSILON)

// Test context
static vsla_context_t* ctx = NULL;
static int test_count = 0;
static int pass_count = 0;

// Helper: print test result
static void report_test(const char* name, int passed) {
    test_count++;
    if (passed) {
        pass_count++;
        printf("✅ %s\n", name);
    } else {
        printf("❌ %s\n", name);
    }
}


// Test 1: Ambient promotion in addition (Section 4.1)
static void test_ambient_promotion_add(void) {
    printf("\n=== Testing Ambient Promotion (Addition) ===\n");
    
    // Create tensors with different shapes
    // a = [1, 2, 3], b = [4, 5]
    // Expected: result = [5, 7, 3] (ambient shape [3])
    uint64_t shape_a[] = {3};
    uint64_t shape_b[] = {2};
    uint64_t shape_out[] = {3}; // max(3, 2) = 3
    
    printf("Creating tensor a with shape [3]...\n");
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    printf("Created tensor a: %p\n", (void*)a);
    
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill data
    for (uint64_t i = 0; i < 3; i++) {
        vsla_set_f64(ctx, a, &i, (double)(i + 1));
    }
    for (uint64_t i = 0; i < 2; i++) {
        vsla_set_f64(ctx, b, &i, (double)(i + 4));
    }
    
    // Perform addition
    vsla_error_t err = vsla_add(ctx, out, a, b);
    report_test("Ambient promotion add operation", err == VSLA_SUCCESS);
    
    // Verify results
    double expected[] = {5.0, 7.0, 3.0};
    int correct = 1;
    for (uint64_t i = 0; i < 3; i++) {
        double val;
        vsla_get_f64(ctx, out, &i, &val);
        if (fabs(val - expected[i]) >= EPSILON) {
            correct = 0;
            printf("  Index %lu: got %.6f, expected %.6f\n", i, val, expected[i]);
        }
    }
    report_test("Ambient promotion add values", correct);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
}

// Test 2: Shrinking to minimal representative (Section 6)
static void test_shrinking(void) {
    printf("\n=== Testing Shrinking to Minimal Representative ===\n");
    
    // Create tensor that will have trailing zeros after subtraction
    // a = [5, 3, 2], b = [2, 1, 2]
    // a - b = [3, 2, 0] -> should shrink to [3, 2]
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    double a_vals[] = {5.0, 3.0, 2.0};
    double b_vals[] = {2.0, 1.0, 2.0};
    
    for (uint64_t i = 0; i < 3; i++) {
        vsla_set_f64(ctx, a, &i, a_vals[i]);
        vsla_set_f64(ctx, b, &i, b_vals[i]);
    }
    
    // Perform subtraction
    vsla_error_t err = vsla_sub(ctx, out, a, b);
    report_test("Subtraction for shrinking test", err == VSLA_SUCCESS);
    
    // Shrink the result
    err = vsla_shrink(ctx, out);
    report_test("Shrink operation", err == VSLA_SUCCESS);
    
    // Check if shape was reduced
    uint64_t new_shape[VSLA_MAX_RANK];
    vsla_get_shape(out, new_shape);
    report_test("Shape reduced after shrinking", new_shape[0] == 2);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
}

// Test 3: Convolution (Model A) correctness
static void test_convolution_model_a(void) {
    printf("\n=== Testing Convolution (Model A) ===\n");
    
    // Simple test: [1, 2] * [3, 4] = [3, 10, 8]
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {2};
    uint64_t shape_out[] = {3}; // m + n - 1 = 2 + 2 - 1 = 3
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill data
    double a_vals[] = {1.0, 2.0};
    double b_vals[] = {3.0, 4.0};
    
    for (uint64_t i = 0; i < 2; i++) {
        vsla_set_f64(ctx, a, &i, a_vals[i]);
        vsla_set_f64(ctx, b, &i, b_vals[i]);
    }
    
    // Perform convolution
    vsla_error_t err = vsla_conv(ctx, out, a, b);
    report_test("Convolution operation", err == VSLA_SUCCESS);
    
    // Expected: [1*3, 1*4+2*3, 2*4] = [3, 10, 8]
    double expected[] = {3.0, 10.0, 8.0};
    int correct = 1;
    for (uint64_t i = 0; i < 3; i++) {
        double val;
        vsla_get_f64(ctx, out, &i, &val);
        if (fabs(val - expected[i]) >= EPSILON) {
            correct = 0;
            printf("  Index %lu: got %.6f, expected %.6f\n", i, val, expected[i]);
        }
    }
    report_test("Convolution values", correct);
    
    // Test scalar identity: [a] * [1] = [a]
    uint64_t scalar_shape[] = {1};
    vsla_tensor_t* scalar = vsla_tensor_create(ctx, 1, scalar_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_set_f64(ctx, scalar, &(uint64_t){0}, 1.0);
    
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    err = vsla_conv(ctx, result, a, scalar);
    
    correct = 1;
    for (uint64_t i = 0; i < 2; i++) {
        double val, orig;
        vsla_get_f64(ctx, result, &i, &val);
        vsla_get_f64(ctx, a, &i, &orig);
        if (fabs(val - orig) >= EPSILON) {
            correct = 0;
        }
    }
    report_test("Convolution scalar identity", correct);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_tensor_free(scalar);
    vsla_tensor_free(result);
}

// Test 4: Kronecker product (Model B) correctness
static void test_kronecker_model_b(void) {
    printf("\n=== Testing Kronecker Product (Model B) ===\n");
    
    // Test: [1, 2] ⊗ [3, 4] = [3, 4, 6, 8]
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {2};
    uint64_t shape_out[] = {4}; // m * n = 2 * 2 = 4
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    // Fill data
    double a_vals[] = {1.0, 2.0};
    double b_vals[] = {3.0, 4.0};
    
    for (uint64_t i = 0; i < 2; i++) {
        vsla_set_f64(ctx, a, &i, a_vals[i]);
        vsla_set_f64(ctx, b, &i, b_vals[i]);
    }
    
    // Perform Kronecker product
    vsla_error_t err = vsla_kron(ctx, out, a, b);
    report_test("Kronecker product operation", err == VSLA_SUCCESS);
    
    // Expected: [1*3, 1*4, 2*3, 2*4] = [3, 4, 6, 8]
    double expected[] = {3.0, 4.0, 6.0, 8.0};
    int correct = 1;
    for (uint64_t i = 0; i < 4; i++) {
        double val;
        vsla_get_f64(ctx, out, &i, &val);
        if (fabs(val - expected[i]) >= EPSILON) {
            correct = 0;
            printf("  Index %lu: got %.6f, expected %.6f\n", i, val, expected[i]);
        }
    }
    report_test("Kronecker product values", correct);
    
    // Test non-commutativity: a ⊗ b ≠ b ⊗ a
    vsla_tensor_t* out2 = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    err = vsla_kron(ctx, out2, b, a);
    
    // b ⊗ a should give [3, 6, 4, 8]
    int different = 0;
    for (uint64_t i = 0; i < 4; i++) {
        double val1, val2;
        vsla_get_f64(ctx, out, &i, &val1);
        vsla_get_f64(ctx, out2, &i, &val2);
        if (fabs(val1 - val2) >= EPSILON && i != 0 && i != 3) {
            different = 1;
        }
    }
    report_test("Kronecker non-commutativity", different);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_tensor_free(out2);
}

// Test 5: Associativity of addition
static void test_addition_associativity(void) {
    printf("\n=== Testing Addition Associativity ===\n");
    
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* temp1 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* temp2 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result1 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result2 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with test data
    double a_vals[] = {1.5, 2.7, 3.1};
    double b_vals[] = {4.2, 5.8, 6.3};
    double c_vals[] = {7.9, 8.4, 9.6};
    
    for (uint64_t i = 0; i < 3; i++) {
        vsla_set_f64(ctx, a, &i, a_vals[i]);
        vsla_set_f64(ctx, b, &i, b_vals[i]);
        vsla_set_f64(ctx, c, &i, c_vals[i]);
    }
    
    // Compute (a + b) + c
    vsla_add(ctx, temp1, a, b);
    vsla_add(ctx, result1, temp1, c);
    
    // Compute a + (b + c)
    vsla_add(ctx, temp2, b, c);
    vsla_add(ctx, result2, a, temp2);
    
    // Compare results
    int correct = 1;
    for (uint64_t i = 0; i < 3; i++) {
        double val1, val2;
        vsla_get_f64(ctx, result1, &i, &val1);
        vsla_get_f64(ctx, result2, &i, &val2);
        if (fabs(val1 - val2) >= EPSILON) {
            correct = 0;
            printf("  Index %lu: (a+b)+c = %.6f, a+(b+c) = %.6f\n", i, val1, val2);
        }
    }
    report_test("Addition associativity", correct);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(c);
    vsla_tensor_free(temp1);
    vsla_tensor_free(temp2);
    vsla_tensor_free(result1);
    vsla_tensor_free(result2);
}

// Test 6: Hadamard product with ambient promotion
static void test_hadamard_ambient(void) {
    printf("\n=== Testing Hadamard Product with Ambient Promotion ===\n");
    
    // a = [2, 3], b = [4, 5, 6]
    // Expected: [8, 15, 0] with ambient shape [3]
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {3};
    uint64_t shape_out[] = {3};
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill data
    double a_vals[] = {2.0, 3.0};
    double b_vals[] = {4.0, 5.0, 6.0};
    
    for (uint64_t i = 0; i < 2; i++) {
        vsla_set_f64(ctx, a, &i, a_vals[i]);
    }
    for (uint64_t i = 0; i < 3; i++) {
        vsla_set_f64(ctx, b, &i, b_vals[i]);
    }
    
    // Perform Hadamard product
    vsla_error_t err = vsla_hadamard(ctx, out, a, b);
    report_test("Hadamard with ambient promotion", err == VSLA_SUCCESS);
    
    // Expected: [2*4, 3*5, 0*6] = [8, 15, 0]
    double expected[] = {8.0, 15.0, 0.0};
    int correct = 1;
    for (uint64_t i = 0; i < 3; i++) {
        double val;
        vsla_get_f64(ctx, out, &i, &val);
        if (fabs(val - expected[i]) >= EPSILON) {
            correct = 0;
            printf("  Index %lu: got %.6f, expected %.6f\n", i, val, expected[i]);
        }
    }
    report_test("Hadamard ambient values", correct);
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
}

int main(void) {
    printf("🔬 VSLA Specification Validation Test Suite\n");
    printf("==========================================\n");
    
    // Initialize context with CPU backend
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU
    };
    
    printf("Initializing VSLA context...\n");
    fflush(stdout);
    ctx = vsla_init(&config);
    printf("vsla_init returned: %p\n", (void*)ctx);
    fflush(stdout);
    
    if (!ctx) {
        fprintf(stderr, "Failed to initialize VSLA context\n");
        return 1;
    }
    printf("Context initialized successfully\n");
    
    // Run all tests
    printf("About to run test_ambient_promotion_add...\n");
    fflush(stdout);
    test_ambient_promotion_add();
    test_shrinking();
    test_convolution_model_a();
    test_kronecker_model_b();
    test_addition_associativity();
    test_hadamard_ambient();
    
    // Summary
    printf("\n==========================================\n");
    printf("📊 Test Summary: %d/%d tests passed\n", pass_count, test_count);
    
    if (pass_count == test_count) {
        printf("✅ All tests passed! CPU backend is spec-compliant.\n");
    } else {
        printf("❌ Some tests failed. Backend needs fixes.\n");
    }
    
    vsla_cleanup(ctx);
    return (pass_count == test_count) ? 0 : 1;
}