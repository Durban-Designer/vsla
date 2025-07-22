/**
 * @file vsla_backend_cpu_new.c
 * @brief VSLA CPU backend implementation following v3.1 specification
 * 
 * Unified CPU backend based on the mathematical specification.
 * Replaces the old implementation with proper VSLA semantics.
 */

#include "vsla/vsla_backend.h"
#include "vsla/vsla_unified.h"

/* Include CPU backend implementation files */
#include "cpu/vsla_cpu_helpers.c"
#include "cpu/vsla_cpu_memory.c"
#include "cpu/vsla_cpu_arithmetic.c"
#include "cpu/vsla_cpu_advanced.c"
#include "cpu/vsla_cpu_reduction.c"
#include "cpu/vsla_cpu_shrink.c"
#include "cpu/vsla_cpu_stacking.c"

/* Wrapper functions to match new interface */
static vsla_error_t cpu_allocate_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    (void)ctx; // Unused for CPU
    return cpu_allocate(tensor);
}

static vsla_error_t cpu_deallocate_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    (void)ctx; // Unused for CPU
    return cpu_deallocate(tensor);
}

static vsla_error_t cpu_copy_to_device_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    (void)ctx; // Unused for CPU
    return cpu_copy_to_device(tensor);
}

static vsla_error_t cpu_copy_to_host_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    (void)ctx; // Unused for CPU
    return cpu_copy_to_host(tensor);
}

static vsla_error_t cpu_synchronize_wrapper(vsla_context_t* ctx) {
    (void)ctx; // Unused for CPU
    return cpu_synchronize();
}

static vsla_error_t cpu_add_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)ctx; // Unused for CPU
    return cpu_add(out, a, b);
}

static vsla_error_t cpu_sub_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)ctx; // Unused for CPU
    return cpu_sub(out, a, b);
}

static vsla_error_t cpu_scale_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* tensor, double scalar) {
    (void)ctx; // Unused for CPU
    return cpu_scale(out, tensor, scalar);
}

static vsla_error_t cpu_hadamard_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)ctx; // Unused for CPU
    return cpu_hadamard(out, a, b);
}

static vsla_error_t cpu_fill_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor, double value) {
    (void)ctx; // Unused for CPU
    return cpu_fill(tensor, value);
}

static vsla_error_t cpu_sum_wrapper(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    (void)ctx; // Unused for CPU
    return cpu_sum(tensor, result);
}

static vsla_error_t cpu_norm_wrapper(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* result) {
    (void)ctx; // Unused for CPU
    return cpu_norm(tensor, result);
}

static vsla_error_t cpu_conv_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* signal, const vsla_tensor_t* kernel) {
    (void)ctx; // Unused for CPU
    return cpu_conv(out, signal, kernel);
}

static vsla_error_t cpu_kron_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)ctx; // Unused for CPU
    return cpu_kron(out, a, b);
}

static vsla_error_t cpu_shrink_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor) {
    (void)ctx; // Unused for CPU
    return cpu_shrink(tensor);
}

static vsla_error_t cpu_stack_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* const* tensors, size_t k) {
    (void)ctx; // Unused for CPU
    return cpu_stack(out, tensors, k);
}

// Placeholder implementations for operations not yet implemented
static vsla_error_t cpu_matmul_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b) {
    (void)ctx; (void)out; (void)a; (void)b;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_transpose_wrapper(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* in) {
    (void)ctx; (void)out; (void)in;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

static vsla_error_t cpu_reshape_wrapper(vsla_context_t* ctx, vsla_tensor_t* tensor, uint8_t new_rank, const uint64_t new_shape[]) {
    (void)ctx; (void)tensor; (void)new_rank; (void)new_shape;
    return VSLA_ERROR_NOT_IMPLEMENTED;
}

/* CPU Backend Interface */
static const vsla_backend_interface_t cpu_backend_interface = {
    /* Backend metadata */
    .caps = {
        .name = "CPU",
        .version = "3.1.0"
    },
    
    /* Memory management */
    .allocate = cpu_allocate_wrapper,
    .deallocate = cpu_deallocate_wrapper,
    .copy_to_device = cpu_copy_to_device_wrapper,
    .copy_to_host = cpu_copy_to_host_wrapper,
    .synchronize = cpu_synchronize_wrapper,
    
    /* Basic arithmetic operations */
    .add = cpu_add_wrapper,
    .sub = cpu_sub_wrapper,
    .scale = cpu_scale_wrapper,
    .hadamard = cpu_hadamard_wrapper,
    .fill = cpu_fill_wrapper,
    
    /* Linear algebra operations */
    .matmul = cpu_matmul_wrapper,
    .transpose = cpu_transpose_wrapper,
    
    /* Tensor operations */
    .reshape = cpu_reshape_wrapper,
    .broadcast = NULL,
    
    /* Reduction operations */
    .sum = cpu_sum_wrapper,
    .mean = NULL,
    .norm = cpu_norm_wrapper,
    .max = NULL,
    .min = NULL,
    
    /* Advanced operations */
    .conv = cpu_conv_wrapper,
    .kron = cpu_kron_wrapper,
    
    /* Structural operations */
    .stack = cpu_stack_wrapper,
    .shrink = cpu_shrink_wrapper,
    
    /* Backend lifecycle */
    .init = NULL,
    .cleanup = NULL
};

vsla_backend_interface_t* vsla_backend_cpu_create(void) {
    return (vsla_backend_interface_t*)&cpu_backend_interface;
}