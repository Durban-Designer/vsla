#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

int main() {
    printf("Testing vsla_matmul for hang issue...\n");
    
    // Initialize VSLA
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return 1;
    }
    
    // Test with a small matrix first
    printf("\nTest 1: Small matrix (32x64 @ 64x32)\n");
    uint64_t a_shape[] = {32, 64};
    uint64_t b_shape[] = {64, 32};
    uint64_t c_shape[] = {32, 32};
    
    vsla_tensor_t* A = vsla_tensor_create(ctx, 2, a_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* B = vsla_tensor_create(ctx, 2, b_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* C = vsla_tensor_create(ctx, 2, c_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!A || !B || !C) {
        printf("ERROR: Failed to create tensors\n");
        return 1;
    }
    
    vsla_fill(ctx, A, 1.0);
    vsla_fill(ctx, B, 0.5);
    vsla_fill(ctx, C, 0.0);
    
    printf("Starting matmul...\n");
    fflush(stdout);
    
    clock_t start = clock();
    vsla_error_t result = vsla_matmul(ctx, C, A, B);
    clock_t end = clock();
    
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Matmul completed in %.3f seconds with result: %d\n", time_spent, result);
    
    // Test with a larger matrix
    printf("\nTest 2: Larger matrix (512x512 @ 512x512)\n");
    vsla_tensor_free(A);
    vsla_tensor_free(B);
    vsla_tensor_free(C);
    
    uint64_t large_shape[] = {512, 512};
    A = vsla_tensor_create(ctx, 2, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    B = vsla_tensor_create(ctx, 2, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    C = vsla_tensor_create(ctx, 2, large_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!A || !B || !C) {
        printf("ERROR: Failed to create large tensors\n");
        return 1;
    }
    
    vsla_fill(ctx, A, 1.0);
    vsla_fill(ctx, B, 0.5);
    vsla_fill(ctx, C, 0.0);
    
    printf("Starting large matmul...\n");
    fflush(stdout);
    
    start = clock();
    result = vsla_matmul(ctx, C, A, B);
    end = clock();
    
    time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Large matmul completed in %.3f seconds with result: %d\n", time_spent, result);
    
    // Cleanup
    vsla_tensor_free(A);
    vsla_tensor_free(B);
    vsla_tensor_free(C);
    vsla_cleanup(ctx);
    
    printf("\nTest completed successfully!\n");
    return 0;
}