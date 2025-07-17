#include <stdio.h>
#include "vsla/vsla.h"

int main() {
    printf("Testing basic tensor creation...\n");
    uint64_t shape[] = {10, 20};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (tensor) {
        printf("Tensor created successfully\n");
        vsla_free(tensor);
        printf("Tensor freed successfully\n");
    } else {
        printf("Failed to create tensor\n");
    }
    return 0;
}
