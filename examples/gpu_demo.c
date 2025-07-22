/**
 * @file gpu_demo.c
 * @brief Demonstration of VSLA GPU acceleration capabilities with the new API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vsla/vsla.h"

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void demo_performance_comparison(vsla_context_t* cpu_ctx, vsla_context_t* gpu_ctx) {
    printf("=== Performance Comparison Demo ===\n");

    // Test different sizes
    uint64_t sizes[] = {64, 256, 1024, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        uint64_t size = sizes[s];
        uint64_t shape[] = {size, size};

        printf("Testing size %lux%lu...\n", size, size);

        // Create tensors for CPU
        vsla_tensor_t* cpu_a = vsla_tensor_create(cpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* cpu_b = vsla_tensor_create(cpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* cpu_result = vsla_tensor_create(cpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

        // Create tensors for GPU
        vsla_tensor_t* gpu_a = vsla_tensor_create(gpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* gpu_b = vsla_tensor_create(gpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* gpu_result = vsla_tensor_create(gpu_ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

        if (!cpu_a || !cpu_b || !cpu_result || !gpu_a || !gpu_b || !gpu_result) {
            printf("Failed to create tensors for size %lu\n", size);
            continue;
        }

        // Initialize data
        vsla_fill(cpu_ctx, cpu_a, 1.0f);
        vsla_fill(cpu_ctx, cpu_b, 2.0f);

        // CPU operation timing
        double start_time = get_time();
        vsla_add(cpu_ctx, cpu_result, cpu_a, cpu_b);
        double cpu_time = get_time() - start_time;

        // GPU operation timing
        // First, copy data to GPU tensors (this would typically be done with a copy function)
        // For this example, we'll just fill them directly on the GPU.
        vsla_fill(gpu_ctx, gpu_a, 1.0f);
        vsla_fill(gpu_ctx, gpu_b, 2.0f);

        start_time = get_time();
        vsla_add(gpu_ctx, gpu_result, gpu_a, gpu_b);
        // vsla_context_synchronize(gpu_ctx); // Assuming a sync function exists
        double gpu_time = get_time() - start_time;

        printf("  CPU: %.6f seconds\n", cpu_time);
        printf("  GPU: %.6f seconds\n", gpu_time);
        printf("  Speedup: %.2fx\n", cpu_time / gpu_time);

        vsla_tensor_free(cpu_a);
        vsla_tensor_free(cpu_b);
        vsla_tensor_free(cpu_result);
        vsla_tensor_free(gpu_a);
        vsla_tensor_free(gpu_b);
        vsla_tensor_free(gpu_result);

        printf("\n");
    }
}

int main(void) {
    printf("VSLA GPU Acceleration Demo\n");
    printf("==========================\n\n");

    // Initialize CPU context
    vsla_config_t cpu_config = { .backend_selection = VSLA_BACKEND_CPU_ONLY };
    vsla_context_t* cpu_ctx = vsla_init(&cpu_config);
    if (!cpu_ctx) {
        printf("Failed to initialize CPU context\n");
        return 1;
    }

    // Initialize GPU context
    vsla_config_t gpu_config = { .backend_selection = VSLA_BACKEND_GPU_ONLY };
    vsla_context_t* gpu_ctx = vsla_init(&gpu_config);
    if (!gpu_ctx) {
        printf("Failed to initialize GPU context. Make sure a CUDA-enabled GPU is available.\n");
        vsla_cleanup(cpu_ctx);
        return 1;
    }

    // Run demonstrations
    demo_performance_comparison(cpu_ctx, gpu_ctx);

    // Cleanup
    vsla_cleanup(cpu_ctx);
    vsla_cleanup(gpu_ctx);

    printf("Demo completed successfully!\n");
    return 0;
}
