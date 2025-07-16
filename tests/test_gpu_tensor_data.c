#include <stdio.h>
#include "vsla/vsla.h"

int main() {
    printf("Testing GPU tensor with data initialization...\n");
    
    if (\!vsla_has_gpu()) {
        printf("GPU not available\n");
        return 1;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (\!ctx) {
        printf("Failed to initialize GPU context\n");
        return 1;
    }
    
    // Create a simple CPU tensor
    uint64_t shape[] = {10, 20};
    vsla_tensor_t* cpu_tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (\!cpu_tensor) {
        printf("Failed to create CPU tensor\n");
        vsla_gpu_destroy(ctx);
        return 1;
    }
    
    printf("CPU tensor created successfully\n");
    
    // Initialize the tensor data similar to create_test_tensor_f32
    size_t total_elements = 1;
    for (uint8_t i = 0; i < cpu_tensor->rank; i++) {
        total_elements *= cpu_tensor->shape[i];
    }
    
    float* data = (float*)cpu_tensor->data;
    for (size_t i = 0; i < total_elements; i++) {
        data[i] = 1.0f + i * 0.5f;
    }
    
    printf("CPU tensor data initialized successfully (%zu elements)\n", total_elements);
    
    // Create GPU tensor from CPU tensor
    vsla_gpu_tensor_t* gpu_tensor = vsla_gpu_tensor_from_cpu(cpu_tensor, ctx);
    if (\!gpu_tensor) {
        printf("Failed to create GPU tensor\n");
        vsla_free(cpu_tensor);
        vsla_gpu_destroy(ctx);
        return 1;
    }
    
    printf("GPU tensor created successfully\n");
    
    // Test GPU tensor allocation
    vsla_error_t err = vsla_gpu_tensor_alloc(gpu_tensor, ctx);
    if (err \!= VSLA_SUCCESS) {
        printf("Failed to allocate GPU tensor memory: %d\n", err);
        vsla_gpu_tensor_free(gpu_tensor);
        vsla_free(cpu_tensor);
        vsla_gpu_destroy(ctx);
        return 1;
    }
    
    printf("GPU tensor memory allocated successfully\n");
    
    // Clean up
    vsla_gpu_tensor_free(gpu_tensor);
    vsla_free(cpu_tensor);
    vsla_gpu_destroy(ctx);
    
    printf("Test completed successfully\n");
    return 0;
}
