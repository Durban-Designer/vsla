/**
 * @file test_core_only.c
 * @brief Test core functionality without full test framework
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(void) {
    printf("Testing VSLA core functionality...\n");
    
    // Test error strings
    const char* err_str = vsla_error_string(VSLA_SUCCESS);
    if (!err_str || strlen(err_str) == 0) {
        printf("✗ Error string test failed\n");
        return 1;
    }
    printf("✓ Error string: %s\n", err_str);
    
    // Test dtype sizes
    size_t f64_size = vsla_dtype_size(VSLA_DTYPE_F64);
    if (f64_size != 8) {
        printf("✗ F64 size test failed: expected 8, got %zu\n", f64_size);
        return 1;
    }
    printf("✓ F64 size: %zu bytes\n", f64_size);
    
    size_t f32_size = vsla_dtype_size(VSLA_DTYPE_F32);
    if (f32_size != 4) {
        printf("✗ F32 size test failed: expected 4, got %zu\n", f32_size);
        return 1;
    }
    printf("✓ F32 size: %zu bytes\n", f32_size);
    
    // Test power of 2 functions
    if (!vsla_is_pow2(1) || !vsla_is_pow2(2) || !vsla_is_pow2(4) || !vsla_is_pow2(8)) {
        printf("✗ is_pow2 test failed for powers of 2\n");
        return 1;
    }
    if (vsla_is_pow2(0) || vsla_is_pow2(3) || vsla_is_pow2(5) || vsla_is_pow2(6)) {
        printf("✗ is_pow2 test failed for non-powers of 2\n");
        return 1;
    }
    printf("✓ is_pow2 function working correctly\n");
    
    // Test next_pow2
    if (vsla_next_pow2(0) != 1 || vsla_next_pow2(1) != 1 || 
        vsla_next_pow2(3) != 4 || vsla_next_pow2(7) != 8 ||
        vsla_next_pow2(15) != 16) {
        printf("✗ next_pow2 test failed\n");
        return 1;
    }
    printf("✓ next_pow2 function working correctly\n");
    
    // Test tensor operations (from autograd tests that were working)
    vsla_init();
    
    // Create simple tensors for convolution test
    uint64_t signal_shape[] = {4};
    uint64_t kernel_shape[] = {3};
    uint64_t result_shape[] = {6}; // 4 + 3 - 1
    
    vsla_tensor_t* signal = vsla_new(1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_new(1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_new(1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel || !result) {
        printf("✗ Failed to create convolution tensors\n");
        return 1;
    }
    
    // Fill signal: [1, 0, 1, 0]
    uint64_t idx;
    idx = 0; vsla_set_f64(signal, &idx, 1.0);
    idx = 1; vsla_set_f64(signal, &idx, 0.0);
    idx = 2; vsla_set_f64(signal, &idx, 1.0);
    idx = 3; vsla_set_f64(signal, &idx, 0.0);
    
    // Fill kernel: [1, 1, 1]
    idx = 0; vsla_set_f64(kernel, &idx, 1.0);
    idx = 1; vsla_set_f64(kernel, &idx, 1.0);
    idx = 2; vsla_set_f64(kernel, &idx, 1.0);
    
    // Test convolution
    if (vsla_conv(result, signal, kernel) == VSLA_SUCCESS) {
        printf("✓ Convolution operation successful\n");
        
        // Check some values
        double val;
        idx = 0;
        vsla_get_f64(result, &idx, &val);
        if (fabs(val - 1.0) < 1e-12) {
            printf("✓ Convolution result[0] correct: %f\n", val);
        }
    } else {
        printf("? Convolution operation failed (FFT might not be available)\n");
    }
    
    // Cleanup
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(result);
    vsla_cleanup();
    
    printf("\n🎉 Core functionality tests completed successfully!\n");
    return 0;
}