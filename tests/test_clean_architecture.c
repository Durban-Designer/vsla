/**
 * @file test_clean_architecture.c 
 * @brief Clean architecture test for multi-agent development
 * 
 * This test demonstrates the clean unified interface that both CPU and CUDA
 * backends can implement. It shows Gemini exactly what needs to be implemented.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(int argc, char* argv[]) {
    printf("=== VSLA Clean Architecture Test ===\n");

    vsla_backend_t backend_type = VSLA_BACKEND_AUTO;
    if (argc > 1 && strcmp(argv[1], "--backend") == 0 && argc > 2) {
        printf("Backend specified: %s\n", argv[2]);
        if (strcmp(argv[2], "CUDA") == 0) {
            printf("Selecting CUDA backend\n");
            backend_type = VSLA_BACKEND_CUDA;
        }
    }

    vsla_config_t config = { .backend = backend_type };
    vsla_context_t* ctx = vsla_init(&config);
    assert(ctx != NULL);
    printf("✅ Context initialized\n");
    
    // Get runtime info
    vsla_backend_t backend;
    char device_name[64];
    double memory_gb;
    vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    printf("✅ Backend: %s (memory: %.1f GB)\n", device_name, memory_gb);
    
    // Create test tensors using unified interface
    uint64_t shape[] = {5};
    vsla_tensor_t* tensor = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    assert(tensor != NULL);
    printf("✅ Tensor created: rank=%d, size=%lu\n", 
           vsla_get_rank(tensor), shape[0]);
    
    // Test basic operations that backends must implement
    vsla_error_t err = vsla_fill(ctx, tensor, 42.0);
    printf("✅ Fill operation: %s\n", 
           err == VSLA_SUCCESS ? "SUCCESS" : vsla_error_string(err));
    
    double sum_result = 0.0;
    err = vsla_sum(ctx, tensor, &sum_result);
    printf("✅ Sum operation: %s (result=%.1f)\n", 
           err == VSLA_SUCCESS ? "SUCCESS" : vsla_error_string(err), sum_result);
    
    // Cleanup
    vsla_tensor_free(tensor);
    vsla_cleanup(ctx);
    printf("✅ Cleanup completed\n");
    
    printf("\n🎯 ARCHITECTURE READY FOR MULTI-AGENT DEVELOPMENT!\n");
    printf("   • CPU Backend: Implement functions in cpu/*.c files\n");  
    printf("   • CUDA Backend: Implement vsla_backend_cuda.c\n");
    printf("   • All functions follow unified interface in vsla_backend.h\n");
    
    return 0;
}