/**
 * @file basic_usage.c
 * @brief Basic usage example for libvsla using the new context-based API.
 *
 * This example demonstrates the core concepts of the new VSLA API:
 * - Initializing a context
 * - Creating tensors with different shapes
 * - Performing basic arithmetic operations
 * - Cleaning up resources
 *
 * Compile with: gcc -I../include basic_usage.c ../build/libvsla.a -lm -o basic_usage
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "vsla/vsla.h"

// Helper function to print a tensor's data
void print_tensor_data(vsla_context_t* ctx, const vsla_tensor_t* tensor, const char* name) {
    if (!tensor) {
        printf("%s: NULL\n", name);
        return;
    }

    printf("%s (rank %u): [", name, vsla_get_rank(tensor));

    if (vsla_get_rank(tensor) == 1) {
        uint64_t shape[1];
        vsla_get_shape(tensor, shape);
        for (uint64_t i = 0; i < shape[0]; i++) {
            double value;
            uint64_t idx[] = {i};
            vsla_get_f64(ctx, tensor, idx, &value);
            printf("%.1f", value);
            if (i < shape[0] - 1) printf(", ");
        }
    } else {
        printf("...");
    }

    printf("]\n");
}

int main() {
    printf("=== VSLA Basic Usage Example (New API) ===\n\n");

    // 1. Initialize VSLA context
    vsla_config_t config = {
        .backend = VSLA_BACKEND_AUTO,
    };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }
    printf("VSLA context initialized.\n");

    // 2. Create tensors with different shapes
    uint64_t shape_a[] = {3};
    uint64_t shape_b[] = {5};

    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);

    if (!a || !b) {
        printf("Failed to create tensors\n");
        vsla_cleanup(ctx);
        return 1;
    }

    printf("Created two tensors with shapes [3] and [5].\n");

    // 3. Fill tensors with data
    vsla_fill(ctx, a, 1.0); // Fill tensor 'a' with 1.0
    vsla_fill(ctx, b, 2.0); // Fill tensor 'b' with 2.0

    printf("Filled tensors with data.\n");

    // 4. Perform variable-shape addition
    uint64_t result_shape[] = {5}; // max(3, 5) = 5
    vsla_tensor_t* result = vsla_tensor_zeros(ctx, 1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!result) {
        printf("Failed to create result tensor\n");
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_cleanup(ctx);
        return 1;
    }

    // VSLA will automatically pad tensor 'a' to the shape of 'result'
    vsla_error_t err = vsla_add(ctx, result, a, b);
    if (err != VSLA_SUCCESS) {
        printf("Addition failed: %s\n", vsla_error_string(err));
    } else {
        printf("Performed variable-shape addition.\n");
    }

    // 5. Clean up resources
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    vsla_cleanup(ctx);

    printf("Cleaned up resources.\n");
    printf("\n=== Example Complete ===\n");

    return 0;
}