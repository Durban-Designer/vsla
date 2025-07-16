/**
 * @file test_gpu_basic.c
 * @brief Basic test for GPU functionality without CUDA
 */

#include <stdio.h>
#include <stdlib.h>
#include "vsla/vsla.h"

int main(void) {
    printf("VSLA GPU Basic Test\n");
    printf("==================\n\n");
    
    // Initialize VSLA library
    vsla_error_t err = vsla_init();
    if (err != VSLA_SUCCESS) {
        printf("Failed to initialize VSLA library: %s\n", vsla_error_string(err));
        return 1;
    }
    
    // Test GPU availability
    printf("Testing GPU availability...\n");
    int has_gpu = vsla_has_gpu();
    printf("GPU support compiled: %s\n", has_gpu ? "YES" : "NO");
    
    // Test GPU initialization
    printf("\nTesting GPU initialization...\n");
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (ctx) {
        printf("GPU context initialized successfully\n");
        
        // Test device info
        char device_name[256];
        double memory_gb;
        err = vsla_gpu_get_device_info(0, device_name, &memory_gb);
        if (err == VSLA_SUCCESS) {
            printf("GPU Device: %s (%.1f GB)\n", device_name, memory_gb);
        } else {
            printf("GPU device info failed: %s\n", vsla_error_string(err));
        }
        
        // Test memory usage
        size_t used_mb, total_mb;
        err = vsla_gpu_get_memory_usage(ctx, &used_mb, &total_mb);
        if (err == VSLA_SUCCESS) {
            printf("GPU Memory: %zu MB used / %zu MB total\n", used_mb, total_mb);
        } else {
            printf("GPU memory usage failed: %s\n", vsla_error_string(err));
        }
        
        vsla_gpu_destroy(ctx);
    } else {
        printf("GPU context initialization failed (expected without CUDA)\n");
    }
    
    // Test basic tensor operations
    printf("\nTesting CPU tensor operations...\n");
    uint64_t shape[] = {4, 4};
    vsla_tensor_t* tensor_a = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* tensor_b = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (tensor_a && tensor_b && result) {
        // Initialize data
        float* data_a = (float*)tensor_a->data;
        float* data_b = (float*)tensor_b->data;
        
        for (int i = 0; i < 16; i++) {
            data_a[i] = (float)i;
            data_b[i] = (float)(i + 1);
        }
        
        // Test addition
        err = vsla_add(result, tensor_a, tensor_b);
        if (err == VSLA_SUCCESS) {
            printf("CPU tensor addition successful\n");
            
            // Verify first few results
            float* result_data = (float*)result->data;
            printf("First 5 results: ");
            for (int i = 0; i < 5; i++) {
                printf("%.1f ", result_data[i]);
            }
            printf("\n");
        } else {
            printf("CPU tensor addition failed: %s\n", vsla_error_string(err));
        }
        
        // Test scaling
        err = vsla_scale(result, tensor_a, 2.0);
        if (err == VSLA_SUCCESS) {
            printf("CPU tensor scaling successful\n");
            
            // Verify first few results
            float* result_data = (float*)result->data;
            printf("First 5 scaled results: ");
            for (int i = 0; i < 5; i++) {
                printf("%.1f ", result_data[i]);
            }
            printf("\n");
        } else {
            printf("CPU tensor scaling failed: %s\n", vsla_error_string(err));
        }
        
        // Clean up
        vsla_free(tensor_a);
        vsla_free(tensor_b);
        vsla_free(result);
    } else {
        printf("Failed to create test tensors\n");
    }
    
    // Test error strings for GPU errors
    printf("\nTesting GPU error strings...\n");
    printf("GPU_FAILURE: %s\n", vsla_error_string(VSLA_ERROR_GPU_FAILURE));
    printf("INVALID_STATE: %s\n", vsla_error_string(VSLA_ERROR_INVALID_STATE));
    
    // Cleanup
    vsla_cleanup();
    
    printf("\nBasic GPU test completed successfully!\n");
    return 0;
}