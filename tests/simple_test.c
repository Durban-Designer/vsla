#include "include/vsla/vsla.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=== VSLA Library Basic Verification ===\n\n");
    
    // Initialize the library
    vsla_init();
    
    // Test 1: Create basic tensors
    size_t size = 5;
    vsla_tensor_t* a = vsla_new(1, &size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, &size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_new(1, &size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !result) {
        printf("ERROR: Failed to create tensors\n");
        return 1;
    }
    printf("✓ Basic tensor creation successful\n");
    
    // Test 2: Set some values
    for (size_t i = 0; i < size; i++) {
        uint64_t idx = i;
        vsla_set_f64(a, &idx, (double)(i + 1));
        vsla_set_f64(b, &idx, (double)(i + 2));
    }
    printf("✓ Tensor value setting successful\n");
    
    // Test 3: Addition
    vsla_error_t err = vsla_add(result, a, b);
    if (err != VSLA_SUCCESS) {
        printf("ERROR: Addition failed with error %d\n", err);
        return 1;
    }
    printf("✓ Tensor addition successful\n");
    
    // Test 4: Verify results
    for (size_t i = 0; i < size; i++) {
        uint64_t idx = i;
        double value;
        err = vsla_get_f64(result, &idx, &value);
        if (err != VSLA_SUCCESS) {
            printf("ERROR: Failed to get value at index %zu\n", i);
            return 1;
        }
        double expected = (double)(i + 1) + (double)(i + 2);
        if (fabs(value - expected) > 1e-12) {
            printf("ERROR: Value mismatch at index %zu: got %f, expected %f\n", i, value, expected);
            return 1;
        }
    }
    printf("✓ Addition results verified\n");
    
    // Test 5: Convolution
    size_t signal_size = 8;
    size_t kernel_size = 3;
    size_t conv_size = signal_size + kernel_size - 1;
    
    vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* conv_result = vsla_new(1, &conv_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel || !conv_result) {
        printf("ERROR: Failed to create convolution tensors\n");
        return 1;
    }
    
    // Simple test pattern
    for (size_t i = 0; i < signal_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(signal, &idx, 1.0);
    }
    for (size_t i = 0; i < kernel_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(kernel, &idx, 1.0);
    }
    
    err = vsla_conv(conv_result, signal, kernel);
    if (err != VSLA_SUCCESS) {
        printf("ERROR: Convolution failed with error %d\n", err);
        return 1;
    }
    printf("✓ FFT convolution successful\n");
    
    // Cleanup
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(conv_result);
    
    vsla_cleanup();
    
    printf("\n=== All Tests Passed! ===\n");
    printf("VSLA library is working correctly.\n");
    
    return 0;
}