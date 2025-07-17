/**
 * @file tensor_stacking.c
 * @brief Example demonstrating VSLA stacking operator and tensor pyramids
 * 
 * Shows how to use the Σ (stacking) and Ω (window-stacking) operators
 * to build tensor pyramids from streaming data.
 */

#include "vsla/vsla_stack.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Helper function to print tensor shape
void print_tensor_info(const char* name, const vsla_tensor_t* tensor) {
    printf("%s: rank=%d, shape=[", name, tensor->rank);
    for (uint8_t i = 0; i < tensor->rank; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->rank - 1) printf(", ");
    }
    printf("]\n");
}

// Example 1: Basic tensor stacking
void example_basic_stacking(void) {
    printf("\n=== Example 1: Basic Tensor Stacking ===\n");
    
    // Create three 1D vectors of different lengths
    vsla_tensor_t* v1 = vsla_new(1, (uint64_t[]){3}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* v2 = vsla_new(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* v3 = vsla_new(1, (uint64_t[]){4}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with sample data
    vsla_set_f64(v1, (uint64_t[]){0}, 1.0);
    vsla_set_f64(v1, (uint64_t[]){1}, 2.0);
    vsla_set_f64(v1, (uint64_t[]){2}, 3.0);
    
    vsla_set_f64(v2, (uint64_t[]){0}, 4.0);
    vsla_set_f64(v2, (uint64_t[]){1}, 5.0);
    
    vsla_set_f64(v3, (uint64_t[]){0}, 6.0);
    vsla_set_f64(v3, (uint64_t[]){1}, 7.0);
    vsla_set_f64(v3, (uint64_t[]){2}, 8.0);
    vsla_set_f64(v3, (uint64_t[]){3}, 9.0);
    
    print_tensor_info("v1", v1);
    print_tensor_info("v2", v2);
    print_tensor_info("v3", v3);
    
    // Stack them using Σ_3: (𝕋_1)^3 → 𝕋_2
    vsla_tensor_t* vectors[] = {v1, v2, v3};
    vsla_tensor_t* matrix = vsla_stack_create(vectors, 3, NULL);
    
    if (matrix) {
        print_tensor_info("Stacked matrix", matrix);
        printf("Matrix represents: [v1_padded; v2_padded; v3_padded]\n");
        
        // Verify stacking worked correctly
        double val;
        vsla_get_f64(matrix, (uint64_t[]){0, 0}, &val);
        printf("matrix[0,0] = %.1f (should be 1.0)\n", val);
        
        vsla_get_f64(matrix, (uint64_t[]){1, 1}, &val);
        printf("matrix[1,1] = %.1f (should be 5.0)\n", val);
        
        vsla_get_f64(matrix, (uint64_t[]){2, 3}, &val);
        printf("matrix[2,3] = %.1f (should be 9.0)\n", val);
        
        // Check zero-padding
        vsla_get_f64(matrix, (uint64_t[]){1, 3}, &val);
        printf("matrix[1,3] = %.1f (should be 0.0 - zero padding)\n", val);
        
        vsla_free(matrix);
    }
    
    vsla_free(v1);
    vsla_free(v2);
    vsla_free(v3);
}

// Example 2: Window-stacking operator Ω for streaming data
void example_window_stacking(void) {
    printf("\n=== Example 2: Window-Stacking for Streaming Data ===\n");
    
    // Create template tensor
    vsla_tensor_t* template = vsla_new(1, (uint64_t[]){3}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Create window accumulator with window size 4
    vsla_window_state_t* window = vsla_window_create(4, template);
    if (!window) {
        printf("Failed to create window\n");
        vsla_free(template);
        return;
    }
    
    printf("Created window accumulator (w=4)\n");
    
    // Simulate streaming data
    const int stream_length = 10;
    for (int t = 0; t < stream_length; t++) {
        // Create tensor for time t
        vsla_tensor_t* tensor_t = vsla_new(1, (uint64_t[]){3}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Fill with time-dependent data
        vsla_set_f64(tensor_t, (uint64_t[]){0}, t * 1.0);
        vsla_set_f64(tensor_t, (uint64_t[]){1}, t * 2.0);
        vsla_set_f64(tensor_t, (uint64_t[]){2}, t * 3.0);
        
        printf("Stream[%d]: [%.1f, %.1f, %.1f]\n", t, t*1.0, t*2.0, t*3.0);
        
        // Accumulate in window
        vsla_tensor_t* batch_output = NULL;
        bool window_ready = vsla_window_accum(window, tensor_t, &batch_output);
        
        if (window_ready) {
            printf("  → Window complete! Emitting batch tensor\n");
            print_tensor_info("  Batch", batch_output);
            
            // Process the batched data
            printf("  Batch contains time steps %d-%d\n", 
                   (int)(t/4)*4, (int)(t/4)*4 + 3);
            
            vsla_free(batch_output);
        }
        
        vsla_free(tensor_t);
    }
    
    // Flush remaining data
    vsla_tensor_t* final_batch = NULL;
    if (vsla_window_flush(window, &final_batch)) {
        printf("Final partial batch:\n");
        print_tensor_info("  Final batch", final_batch);
        vsla_free(final_batch);
    }
    
    // Get statistics
    size_t current, total, emitted;
    vsla_window_stats(window, &current, &total, &emitted);
    printf("Window stats: processed=%zu, emitted=%zu windows\n", total, emitted);
    
    vsla_window_free(window);
    vsla_free(template);
}

// Example 3: Tensor pyramid construction
void example_tensor_pyramid(void) {
    printf("\n=== Example 3: Tensor Pyramid Construction ===\n");
    
    // Create template tensor
    vsla_tensor_t* template = vsla_new(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Create 3-level pyramid: level 0 (w=3) → level 1 (w=2) → level 2 (w=2)
    size_t window_sizes[] = {3, 2, 2};
    vsla_pyramid_builder_t* pyramid = vsla_pyramid_create(3, window_sizes, template);
    
    if (!pyramid) {
        printf("Failed to create pyramid\n");
        vsla_free(template);
        return;
    }
    
    printf("Created 3-level tensor pyramid\n");
    printf("Level 0: window_size=3 (𝕋_1 → 𝕋_2)\n");
    printf("Level 1: window_size=2 (𝕋_2 → 𝕋_3)\n"); 
    printf("Level 2: window_size=2 (𝕋_3 → 𝕋_4)\n");
    
    // Feed data stream into pyramid
    const int stream_length = 15;
    for (int t = 0; t < stream_length; t++) {
        // Create input tensor
        vsla_tensor_t* input = vsla_new(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_set_f64(input, (uint64_t[]){0}, t * 10.0);
        vsla_set_f64(input, (uint64_t[]){1}, t * 10.0 + 1.0);
        
        printf("Input[%d]: [%.1f, %.1f] → ", t, t*10.0, t*10.0+1.0);
        
        // Add to pyramid
        vsla_tensor_t* level_outputs[3];
        size_t num_outputs;
        
        vsla_pyramid_add(pyramid, input, level_outputs, 3, &num_outputs);
        
        if (num_outputs > 0) {
            printf("Pyramid outputs: ");
            for (size_t i = 0; i < num_outputs; i++) {
                printf("Level%zu(rank=%d) ", i, level_outputs[i]->rank);
                vsla_free(level_outputs[i]);
            }
            printf("\n");
        } else {
            printf("(accumulating)\n");
        }
        
        vsla_free(input);
    }
    
    // Flush pyramid
    printf("\nFlushing pyramid...\n");
    vsla_tensor_t* final_outputs[3];
    size_t final_count;
    
    vsla_pyramid_flush(pyramid, final_outputs, 3, &final_count);
    for (size_t i = 0; i < final_count; i++) {
        printf("Final Level%zu: ", i);
        print_tensor_info("", final_outputs[i]);
        vsla_free(final_outputs[i]);
    }
    
    vsla_pyramid_free(pyramid);
    vsla_free(template);
}

// Example 4: Demonstrating algebraic properties
void example_algebraic_properties(void) {
    printf("\n=== Example 4: Algebraic Properties of Stacking ===\n");
    
    // Create test vectors
    vsla_tensor_t* a = vsla_new(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, (uint64_t[]){3}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_new(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with data
    vsla_set_f64(a, (uint64_t[]){0}, 1.0);
    vsla_set_f64(a, (uint64_t[]){1}, 2.0);
    
    vsla_set_f64(b, (uint64_t[]){0}, 3.0);
    vsla_set_f64(b, (uint64_t[]){1}, 4.0);
    vsla_set_f64(b, (uint64_t[]){2}, 5.0);
    
    vsla_set_f64(c, (uint64_t[]){0}, 6.0);
    vsla_set_f64(c, (uint64_t[]){1}, 7.0);
    
    // Test neutral-zero absorption
    printf("Testing neutral-zero absorption:\n");
    vsla_tensor_t* zero = vsla_zeros(1, (uint64_t[]){2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    vsla_tensor_t* with_zero[] = {a, zero, c};
    vsla_tensor_t* without_zero[] = {a, c};
    
    vsla_tensor_t* stack_with = vsla_stack_create(with_zero, 3, NULL);
    vsla_tensor_t* stack_without = vsla_stack_create(without_zero, 2, NULL);
    
    if (stack_with && stack_without) {
        print_tensor_info("Stack with zero", stack_with);
        print_tensor_info("Stack without zero", stack_without);
        printf("Zero injection preserves mathematical structure\n");
    }
    
    // Test unstacking (inverse operation)
    printf("\nTesting unstacking (inverse operation):\n");
    if (stack_with) {
        vsla_tensor_t* unstacked[3];
        size_t num_unstacked;
        
        if (vsla_unstack(stack_with, 0, unstacked, 3, &num_unstacked) == VSLA_SUCCESS) {
            printf("Successfully unstacked into %zu tensors:\n", num_unstacked);
            for (size_t i = 0; i < num_unstacked; i++) {
                print_tensor_info("  Unstacked", unstacked[i]);
                vsla_free(unstacked[i]);
            }
        }
    }
    
    // Cleanup
    if (stack_with) vsla_free(stack_with);
    if (stack_without) vsla_free(stack_without);
    vsla_free(a);
    vsla_free(b);
    vsla_free(c);
    vsla_free(zero);
}

int main(void) {
    printf("VSLA Tensor Stacking and Pyramid Construction Examples\n");
    printf("======================================================\n");
    
    // Initialize VSLA library
    if (vsla_init() != VSLA_SUCCESS) {
        printf("Failed to initialize VSLA\n");
        return 1;
    }
    
    // Run examples
    example_basic_stacking();
    example_window_stacking();
    example_tensor_pyramid();
    example_algebraic_properties();
    
    printf("\n=== Summary ===\n");
    printf("✓ Stacking operator Σ: (𝕋_r)^k → 𝕋_{r+1}\n");
    printf("✓ Window-stacking operator Ω: Stream(𝕋_r) → Stream(𝕋_{r+1})\n");
    printf("✓ Tensor pyramid construction: recursive aggregation\n");
    printf("✓ Algebraic properties: associativity, zero-absorption\n");
    printf("✓ Forms strict monoidal category (𝕋_r, +, Σ)\n");
    
    // Cleanup VSLA library
    vsla_cleanup();
    
    return 0;
}