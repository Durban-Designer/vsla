#include <stdio.h>
#include "vsla/vsla.h"

int main() {
    printf("Testing GPU initialization...\n");
    
    if (!vsla_has_gpu()) {
        printf("GPU not available\n");
        return 1;
    }
    
    printf("GPU available, initializing...\n");
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("Failed to initialize GPU context\n");
        return 1;
    }
    
    printf("GPU context initialized successfully\n");
    
    vsla_gpu_destroy(ctx);
    printf("GPU context destroyed successfully\n");
    
    return 0;
}
