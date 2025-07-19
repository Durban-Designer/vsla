#include "vsla/vsla_backend.h"

/* Include CPU backend implementation files */
#include "cpu/vsla_cpu_memory.c"
#include "cpu/vsla_cpu_arithmetic.c"
#include "cpu/vsla_cpu_tensor.c"
#include "cpu/vsla_cpu_linalg.c"
#include "cpu/vsla_cpu_reduction.c"
#include "cpu/vsla_cpu_advanced.c"

vsla_backend_interface_t* vsla_backend_cpu_create(void) {
    static vsla_backend_interface_t cpu_backend = {
        .caps = {
            .supports_gpu = false,
            .supports_multi_gpu = false,
            .supports_unified_memory = false,
            .supports_async = false,
            .max_tensor_size = SIZE_MAX,
            .name = "CPU",
            .version = "1.0.0"
        },
        
        /* Memory management */
        .allocate = cpu_allocate,
        .deallocate = cpu_deallocate,
        .copy_to_device = cpu_copy_to_device,
        .copy_to_host = cpu_copy_to_host,
        .synchronize = cpu_synchronize,
        
        /* Basic arithmetic operations */
        .add = cpu_add,
        .sub = cpu_sub,
        .scale = cpu_scale,
        .hadamard = cpu_hadamard,
        .fill = cpu_fill,
        
        /* Linear algebra operations */
        .matmul = cpu_matmul,
        .transpose = cpu_transpose,
        
        /* Tensor operations */
        .reshape = cpu_reshape,
        .broadcast = cpu_broadcast,
        
        /* Reduction operations */
        .sum = cpu_sum,
        .mean = cpu_mean,
        .norm = cpu_norm,
        .max = cpu_max,
        .min = cpu_min,
        
        /* Advanced operations */
        .conv = cpu_conv,
        .kron = cpu_kron,
        
        /* Backend lifecycle */
        .init = cpu_init,
        .cleanup = cpu_cleanup
    };
    
    return &cpu_backend;
}
