#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    vsla_config_t config = { .backend = VSLA_BACKEND_CUDA };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    uint64_t shape[] = {4};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    double b_data[] = {5.0, 6.0, 7.0, 8.0};

    // vsla_tensor_set_data(ctx, a, a_data);
    // vsla_tensor_set_data(ctx, b, b_data);

    vsla_add(ctx, result, a, b);

    // double result_data[4];
    // vsla_tensor_get_data(ctx, result, result_data);

    // for (int i = 0; i < 4; i++) {
    //     printf("%.2f ", result_data[i]);
    // }
    // printf("\n");

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
    vsla_cleanup(ctx);

    return 0;
}