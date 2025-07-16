/**
 * @file gpu_demo.c
 * @brief Demonstration of VSLA GPU acceleration capabilities
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

static void print_tensor_info(const char* name, vsla_tensor_t* tensor) {
    printf("%s: rank=%d, shape=[", name, tensor->rank);
    for (int i = 0; i < tensor->rank; i++) {
        printf("%lu", tensor->shape[i]);
        if (i < tensor->rank - 1) printf(", ");
    }
    printf("], dtype=%s\n", tensor->dtype == VSLA_DTYPE_F32 ? "float32" : "float64");
}

static void demo_gpu_availability(void) {
    printf("=== GPU Availability Demo ===\n");
    
    int has_gpu = vsla_has_gpu();
    printf("GPU support compiled: %s\n", has_gpu ? "YES" : "NO");
    
    if (!has_gpu) {
        printf("To enable GPU support, compile with -DVSLA_ENABLE_CUDA=ON\n");
        return;
    }
    
    // Try to initialize GPU context
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("GPU initialization failed - no CUDA device available\n");
        return;
    }
    
    // Get device information
    char device_name[256];
    double memory_gb;
    vsla_error_t err = vsla_gpu_get_device_info(0, device_name, &memory_gb);
    if (err == VSLA_SUCCESS) {
        printf("GPU Device: %s\n", device_name);
        printf("GPU Memory: %.1f GB\n", memory_gb);
    }
    
    // Get memory usage
    size_t used_mb, total_mb;
    err = vsla_gpu_get_memory_usage(ctx, &used_mb, &total_mb);
    if (err == VSLA_SUCCESS) {
        printf("Memory Usage: %zu MB used / %zu MB total\n", used_mb, total_mb);
    }
    
    vsla_gpu_destroy(ctx);
    printf("\n");
}

static void demo_gpu_tensor_operations(void) {
    printf("=== GPU Tensor Operations Demo ===\n");
    
    if (!vsla_has_gpu()) {
        printf("Skipping GPU tensor demo - CUDA not available\n\n");
        return;
    }
    
    // Initialize GPU context
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("GPU context initialization failed\n\n");
        return;
    }
    
    // Create test tensors
    uint64_t shape[] = {8, 8};
    vsla_tensor_t* cpu_a = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* cpu_b = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* cpu_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (!cpu_a || !cpu_b || !cpu_result) {
        printf("Failed to create CPU tensors\n\n");
        return;
    }
    
    print_tensor_info("Tensor A", cpu_a);
    print_tensor_info("Tensor B", cpu_b);
    
    // Initialize test data
    float* data_a = (float*)cpu_a->data;
    float* data_b = (float*)cpu_b->data;
    
    printf("Initializing test data...\n");
    for (int i = 0; i < 64; i++) {
        data_a[i] = (float)i * 0.1f;
        data_b[i] = (float)(i + 1) * 0.2f;
    }
    
    // Create GPU tensors
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(cpu_a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(cpu_b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(cpu_result, ctx);
    
    if (!gpu_a || !gpu_b || !gpu_result) {
        printf("Failed to create GPU tensors\n\n");
        goto cleanup_cpu;
    }
    
    // Allocate GPU memory
    printf("Allocating GPU memory...\n");
    vsla_error_t err = vsla_gpu_tensor_alloc(gpu_a, ctx);
    if (err != VSLA_SUCCESS) {
        printf("GPU memory allocation failed for tensor A\n\n");
        goto cleanup_gpu;
    }
    
    err = vsla_gpu_tensor_alloc(gpu_b, ctx);
    if (err != VSLA_SUCCESS) {
        printf("GPU memory allocation failed for tensor B\n\n");
        goto cleanup_gpu;
    }
    
    err = vsla_gpu_tensor_alloc(gpu_result, ctx);
    if (err != VSLA_SUCCESS) {
        printf("GPU memory allocation failed for result tensor\n\n");
        goto cleanup_gpu;
    }
    
    // Copy data to GPU
    printf("Copying data to GPU...\n");
    err = vsla_gpu_tensor_copy_to_gpu(gpu_a, cpu_a->data, false);
    if (err != VSLA_SUCCESS) {
        printf("Failed to copy tensor A to GPU\n\n");
        goto cleanup_gpu;
    }
    
    err = vsla_gpu_tensor_copy_to_gpu(gpu_b, cpu_b->data, false);
    if (err != VSLA_SUCCESS) {
        printf("Failed to copy tensor B to GPU\n\n");
        goto cleanup_gpu;
    }
    
    // Perform GPU addition
    printf("Performing GPU addition...\n");
    double start_time = get_time();
    err = vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
    vsla_gpu_tensor_sync(gpu_result);
    double end_time = get_time();
    
    if (err == VSLA_SUCCESS) {
        printf("GPU addition completed in %.6f seconds\n", end_time - start_time);
        
        // Copy result back to CPU
        float result_data[64];
        err = vsla_gpu_tensor_copy_to_cpu(gpu_result, result_data, false);
        if (err == VSLA_SUCCESS) {
            printf("First 10 results: ");
            for (int i = 0; i < 10; i++) {
                printf("%.2f ", result_data[i]);
            }
            printf("\n");
        }
    } else {
        printf("GPU addition failed\n");
    }
    
    // Perform GPU scaling
    printf("Performing GPU scaling (scale by 2.0)...\n");
    start_time = get_time();
    err = vsla_gpu_scale(gpu_result, gpu_a, 2.0, ctx);
    vsla_gpu_tensor_sync(gpu_result);
    end_time = get_time();
    
    if (err == VSLA_SUCCESS) {
        printf("GPU scaling completed in %.6f seconds\n", end_time - start_time);
        
        // Copy result back to CPU
        float result_data[64];
        err = vsla_gpu_tensor_copy_to_cpu(gpu_result, result_data, false);
        if (err == VSLA_SUCCESS) {
            printf("First 10 scaled results: ");
            for (int i = 0; i < 10; i++) {
                printf("%.2f ", result_data[i]);
            }
            printf("\n");
        }
    } else {
        printf("GPU scaling failed\n");
    }
    
cleanup_gpu:
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    
cleanup_cpu:
    vsla_free(cpu_a);
    vsla_free(cpu_b);
    vsla_free(cpu_result);
    vsla_gpu_destroy(ctx);
    
    printf("\n");
}

static void demo_performance_comparison(void) {
    printf("=== Performance Comparison Demo ===\n");
    
    if (!vsla_has_gpu()) {
        printf("Skipping performance comparison - CUDA not available\n\n");
        return;
    }
    
    // Initialize GPU context
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("GPU context initialization failed\n\n");
        return;
    }
    
    // Test different sizes
    uint64_t sizes[] = {64, 256, 1024, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        uint64_t size = sizes[s];
        uint64_t shape[] = {size, size};
        
        printf("Testing size %lux%lu...\n", size, size);
        
        // Create CPU tensors
        vsla_tensor_t* cpu_a = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* cpu_b = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_tensor_t* cpu_result = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
        
        if (!cpu_a || !cpu_b || !cpu_result) {
            printf("Failed to create CPU tensors for size %lu\n", size);
            continue;
        }
        
        // Initialize data
        float* data_a = (float*)cpu_a->data;
        float* data_b = (float*)cpu_b->data;
        
        for (uint64_t i = 0; i < size * size; i++) {
            data_a[i] = (float)i * 0.001f;
            data_b[i] = (float)(i + 1) * 0.001f;
        }
        
        // CPU operation timing
        double start_time = get_time();
        vsla_add(cpu_result, cpu_a, cpu_b);
        double cpu_time = get_time() - start_time;
        
        // GPU operation timing
        vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(cpu_a, ctx);
        vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(cpu_b, ctx);
        vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(cpu_result, ctx);
        
        if (gpu_a && gpu_b && gpu_result) {
            if (vsla_gpu_tensor_alloc(gpu_a, ctx) == VSLA_SUCCESS &&
                vsla_gpu_tensor_alloc(gpu_b, ctx) == VSLA_SUCCESS &&
                vsla_gpu_tensor_alloc(gpu_result, ctx) == VSLA_SUCCESS) {
                
                // Copy to GPU
                vsla_gpu_tensor_copy_to_gpu(gpu_a, cpu_a->data, false);
                vsla_gpu_tensor_copy_to_gpu(gpu_b, cpu_b->data, false);
                
                // Time GPU operation
                start_time = get_time();
                vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
                vsla_gpu_tensor_sync(gpu_result);
                double gpu_time = get_time() - start_time;
                
                printf("  CPU: %.6f seconds\n", cpu_time);
                printf("  GPU: %.6f seconds\n", gpu_time);
                printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
            }
            
            vsla_gpu_tensor_free(gpu_a);
            vsla_gpu_tensor_free(gpu_b);
            vsla_gpu_tensor_free(gpu_result);
        }
        
        vsla_free(cpu_a);
        vsla_free(cpu_b);
        vsla_free(cpu_result);
        
        printf("\n");
    }
    
    vsla_gpu_destroy(ctx);
}

int main(void) {
    printf("VSLA GPU Acceleration Demo\n");
    printf("==========================\n\n");
    
    // Initialize VSLA library
    vsla_error_t err = vsla_init();
    if (err != VSLA_SUCCESS) {
        printf("Failed to initialize VSLA library: %s\n", vsla_error_string(err));
        return 1;
    }
    
    // Run demonstrations
    demo_gpu_availability();
    demo_gpu_tensor_operations();
    demo_performance_comparison();
    
    // Cleanup
    vsla_cleanup();
    
    printf("Demo completed successfully!\n");
    return 0;
}