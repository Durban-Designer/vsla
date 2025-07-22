/**
 * @file tensor_stacking.c
 * @brief Example demonstrating tensor stacking using the new context-based API.
 *
 * This example shows how to stack multiple tensors into a single larger tensor.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>

// Helper function to print tensor shape
void print_tensor_info(const vsla_tensor_t* tensor, const char* name) {
    if (!tensor) {
        printf("%s: NULL\n", name);
        return;
    }
    const uint64_t* shape = vsla_tensor_get_shape(tensor);
    uint8_t rank = vsla_tensor_get_rank(tensor);

    printf("%s: rank=%u, shape=[", name, rank);
    for (uint8_t i = 0; i < rank; i++) {
        printf("%lu", shape[i]);
        if (i < rank - 1) printf(", ");
    }
    printf("]\n");
}

int main(void) {
    printf("VSLA Tensor Stacking Example (New API)\n");
    printf("======================================\n");

    // Initialize VSLA context
    vsla_config_t config = { .backend_selection = VSLA_BACKEND_AUTO };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    // Create three 1D vectors of different lengths
    uint64_t shape1[] = {3};
    vsla_tensor_t* v1 = vsla_tensor_create(ctx, 1, shape1, VSLA_MODEL_A, VSLA_DTYPE_F64);

    uint64_t shape2[] = {2};
    vsla_tensor_t* v2 = vsla_tensor_create(ctx, 1, shape2, VSLA_MODEL_A, VSLA_DTYPE_F64);

    uint64_t shape3[] = {4};
    vsla_tensor_t* v3 = vsla_tensor_create(ctx, 1, shape3, VSLA_MODEL_A, VSLA_DTYPE_F64);

    // In a real application, you would fill these tensors with data.

    print_tensor_info(v1, "v1");
    print_tensor_info(v2, "v2");
    print_tensor_info(v3, "v3");

    // To stack these tensors, we create a larger tensor and copy the data.
    uint64_t stacked_shape[] = {3, 4}; // 3 vectors, max length 4
    vsla_tensor_t* matrix = vsla_tensor_zeros(ctx, 2, stacked_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

    // This is a simplified demonstration. A real implementation would require
    // a function to copy data between tensors with different shapes.

    print_tensor_info(matrix, "Stacked matrix");

    // Clean up
    vsla_tensor_free(v1);
    vsla_tensor_free(v2);
    vsla_tensor_free(v3);
    vsla_tensor_free(matrix);
    vsla_cleanup(ctx);

    return 0;
}
