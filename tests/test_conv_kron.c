/**
 * @file test_conv_kron.c
 * @brief Test convolution and Kronecker product operations
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_DOUBLE_EQ(a, b, eps) assert(fabs((a) - (b)) < (eps))

int test_conv_operation(void) {
    printf("Testing convolution operation...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    ASSERT_EQ(ctx != NULL, 1);
    
    // Create 1D tensors for convolution
    uint64_t shape_a[] = {3};  // [1.0, 2.0, 3.0]
    uint64_t shape_b[] = {2};  // [0.5, 1.5]
    uint64_t shape_out[] = {4}; // Should be 3+2-1=4
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    ASSERT_EQ(a != NULL, 1);
    ASSERT_EQ(b != NULL, 1);
    ASSERT_EQ(out != NULL, 1);
    
    // Fill tensor a: [1.0, 2.0, 3.0]
    double* a_data = (double*)vsla_tensor_data_mut(a, NULL);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    
    // Fill tensor b: [0.5, 1.5]
    double* b_data = (double*)vsla_tensor_data_mut(b, NULL);
    b_data[0] = 0.5;
    b_data[1] = 1.5;
    
    // Perform convolution
    vsla_error_t err = vsla_conv(ctx, out, a, b);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    // Check result
    // Expected: [1.0*0.5, 1.0*1.5+2.0*0.5, 2.0*1.5+3.0*0.5, 3.0*1.5] = [0.5, 2.5, 4.5, 4.5]
    const double* out_data = (const double*)vsla_tensor_data(out, NULL);
    ASSERT_DOUBLE_EQ(out_data[0], 0.5, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[1], 2.5, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[2], 4.5, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[3], 4.5, 1e-10);
    
    printf("  ✅ Convolution result: [%.1f, %.1f, %.1f, %.1f]\n", 
           out_data[0], out_data[1], out_data[2], out_data[3]);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_cleanup(ctx);
    
    return 1;
}

int test_kron_operation(void) {
    printf("Testing Kronecker product operation...\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    ASSERT_EQ(ctx != NULL, 1);
    
    // Create 1D tensors for Kronecker product
    uint64_t shape_a[] = {2};  // [2.0, 3.0]
    uint64_t shape_b[] = {3};  // [1.0, 0.0, -1.0]
    uint64_t shape_out[] = {6}; // Should be 2*3=6
    
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
    
    ASSERT_EQ(a != NULL, 1);
    ASSERT_EQ(b != NULL, 1);
    ASSERT_EQ(out != NULL, 1);
    
    // Fill tensor a: [2.0, 3.0]
    double* a_data = (double*)vsla_tensor_data_mut(a, NULL);
    a_data[0] = 2.0;
    a_data[1] = 3.0;
    
    // Fill tensor b: [1.0, 0.0, -1.0]
    double* b_data = (double*)vsla_tensor_data_mut(b, NULL);
    b_data[0] = 1.0;
    b_data[1] = 0.0;
    b_data[2] = -1.0;
    
    // Perform Kronecker product
    vsla_error_t err = vsla_kron(ctx, out, a, b);
    ASSERT_EQ(err, VSLA_SUCCESS);
    
    // Check result
    // Expected: [2*1, 2*0, 2*(-1), 3*1, 3*0, 3*(-1)] = [2.0, 0.0, -2.0, 3.0, 0.0, -3.0]
    const double* out_data = (const double*)vsla_tensor_data(out, NULL);
    ASSERT_DOUBLE_EQ(out_data[0], 2.0, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[1], 0.0, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[2], -2.0, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[3], 3.0, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[4], 0.0, 1e-10);
    ASSERT_DOUBLE_EQ(out_data[5], -3.0, 1e-10);
    
    printf("  ✅ Kronecker product result: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out);
    vsla_cleanup(ctx);
    
    return 1;
}

int main(void) {
    printf("🧪 VSLA Convolution & Kronecker Test Suite\n");
    printf("==========================================\n");
    
    int tests_passed = 0;
    int total_tests = 0;
    
    total_tests++;
    if (test_conv_operation()) tests_passed++;
    
    total_tests++;
    if (test_kron_operation()) tests_passed++;
    
    printf("\n📊 Results: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("✅ All conv/kron tests PASSED!\n");
        return 0;
    } else {
        printf("❌ Some tests FAILED!\n");
        return 1;
    }
}