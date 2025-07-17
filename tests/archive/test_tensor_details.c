#include <stdio.h>
#include <stdint.h>
#include "vsla/vsla.h"

int main() {
    printf("Testing tensor memory details...\n");
    uint64_t shape[] = {10, 20};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (tensor) {
        printf("Tensor created successfully\n");
        printf("Rank: %d\n", tensor->rank);
        printf("Dtype: %d\n", tensor->dtype);
        printf("Shape: [%lu, %lu]\n", tensor->shape[0], tensor->shape[1]);
        printf("Cap: [%lu, %lu]\n", tensor->cap[0], tensor->cap[1]);
        
        // Calculate expected size
        size_t data_size = 1;
        for (uint8_t i = 0; i < tensor->rank; i++) {
            data_size *= tensor->cap[i];
        }
        data_size *= sizeof(float);
        printf("Expected data size: %zu bytes\n", data_size);
        
        vsla_free(tensor);
        printf("Tensor freed successfully\n");
    } else {
        printf("Failed to create tensor\n");
    }
    return 0;
}
