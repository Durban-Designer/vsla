/**
 * @file test_simple.c
 * @brief Simple test to verify VSLA library works
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(void) {
    printf("Starting VSLA simple test...\n");
    
    // Initialize library
    if (vsla_init() != VSLA_SUCCESS) {
        printf("Failed to initialize VSLA\n");
        return 1;
    }
    
    printf("✓ Library initialization successful\n");
    
    // Test version info
    const char* version = vsla_version();
    if (!version || strlen(version) == 0) {
        printf("✗ Version string is invalid\n");
        return 1;
    }
    printf("✓ Version: %s\n", version);
    
    // Test basic tensor creation
    uint64_t shape[] = {3};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!tensor) {
        printf("✗ Failed to create tensor\n");
        return 1;
    }
    printf("✓ Tensor creation successful\n");
    
    // Test data access
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        if (vsla_set_f64(tensor, &idx, (double)(i + 1)) != VSLA_SUCCESS) {
            printf("✗ Failed to set tensor value at index %zu\n", i);
            return 1;
        }
    }
    printf("✓ Tensor data setting successful\n");
    
    // Test data retrieval
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double val;
        if (vsla_get_f64(tensor, &idx, &val) != VSLA_SUCCESS) {
            printf("✗ Failed to get tensor value at index %zu\n", i);
            return 1;
        }
        double expected = (double)(i + 1);
        if (fabs(val - expected) > 1e-12) {
            printf("✗ Tensor value mismatch at index %zu: expected %f, got %f\n", i, expected, val);
            return 1;
        }
    }
    printf("✓ Tensor data retrieval successful\n");
    
    // Test basic operations
    vsla_tensor_t* tensor2 = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!tensor2 || !result) {
        printf("✗ Failed to create additional tensors\n");
        return 1;
    }
    
    // Fill second tensor: [2, 3, 4]
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        vsla_set_f64(tensor2, &idx, (double)(i + 2));
    }
    
    // Test addition: [1,2,3] + [2,3,4] = [3,5,7]
    if (vsla_add(result, tensor, tensor2) != VSLA_SUCCESS) {
        printf("✗ Failed to add tensors\n");
        return 1;
    }
    
    // Verify addition result
    double expected_sum[] = {3.0, 5.0, 7.0};
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double val;
        if (vsla_get_f64(result, &idx, &val) != VSLA_SUCCESS) {
            printf("✗ Failed to get addition result at index %zu\n", i);
            return 1;
        }
        if (fabs(val - expected_sum[i]) > 1e-12) {
            printf("✗ Addition result mismatch at index %zu: expected %f, got %f\n", 
                   i, expected_sum[i], val);
            return 1;
        }
    }
    printf("✓ Tensor addition successful\n");
    
    // Test scaling
    if (vsla_scale(result, tensor, 2.0) != VSLA_SUCCESS) {
        printf("✗ Failed to scale tensor\n");
        return 1;
    }
    
    // Verify scaling result: [1,2,3] * 2 = [2,4,6]
    double expected_scale[] = {2.0, 4.0, 6.0};
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double val;
        if (vsla_get_f64(result, &idx, &val) != VSLA_SUCCESS) {
            printf("✗ Failed to get scaling result at index %zu\n", i);
            return 1;
        }
        if (fabs(val - expected_scale[i]) > 1e-12) {
            printf("✗ Scaling result mismatch at index %zu: expected %f, got %f\n", 
                   i, expected_scale[i], val);
            return 1;
        }
    }
    printf("✓ Tensor scaling successful\n");
    
    // Test utility functions
    if (vsla_dtype_size(VSLA_DTYPE_F64) != 8) {
        printf("✗ Data type size incorrect\n");
        return 1;
    }
    printf("✓ Data type utilities working\n");
    
    if (!vsla_is_pow2(16) || vsla_is_pow2(15)) {
        printf("✗ Power of 2 utilities incorrect\n");
        return 1;
    }
    printf("✓ Power of 2 utilities working\n");
    
    if (vsla_next_pow2(15) != 16) {
        printf("✗ Next power of 2 incorrect\n");
        return 1;
    }
    printf("✓ Next power of 2 utility working\n");
    
    // Cleanup
    vsla_free(tensor);
    vsla_free(tensor2);
    vsla_free(result);
    
    if (vsla_cleanup() != VSLA_SUCCESS) {
        printf("✗ Failed to cleanup library\n");
        return 1;
    }
    printf("✓ Library cleanup successful\n");
    
    printf("\n🎉 All tests passed! VSLA library is working correctly.\n");
    return 0;
}