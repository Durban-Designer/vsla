/**
 * @file new_benchmark.c
 * @brief A new benchmark suite for VSLA using the context-based API.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// High-precision timing
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Benchmark for the add operation
static void benchmark_add(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape[] = {size};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);
    vsla_fill(ctx, b, 2.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_add(ctx, result, a, b);
        total_time += get_time_ms() - start_time;
    }

    printf("Add operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

// Benchmark for the sub operation
static void benchmark_sub(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape[] = {size};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);
    vsla_fill(ctx, b, 2.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_sub(ctx, result, a, b);
        total_time += get_time_ms() - start_time;
    }

    printf("Sub operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

// Benchmark for the scale operation
static void benchmark_scale(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape[] = {size};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_scale(ctx, result, a, 2.0f);
        total_time += get_time_ms() - start_time;
    }

    printf("Scale operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(result);
}

// Benchmark for the hadamard operation
static void benchmark_hadamard(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape[] = {size};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);
    vsla_fill(ctx, b, 2.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_hadamard(ctx, result, a, b);
        total_time += get_time_ms() - start_time;
    }

    printf("Hadamard operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

// Benchmark for the conv operation
static void benchmark_conv(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape_a[] = {size};
    uint64_t shape_b[] = {size / 2};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F32);
    uint64_t shape_result[] = {size + size / 2 - 1};
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape_result, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);
    vsla_fill(ctx, b, 2.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_conv(ctx, result, a, b);
        total_time += get_time_ms() - start_time;
    }

    printf("Conv operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

// Benchmark for the kron operation
static void benchmark_kron(vsla_context_t* ctx, uint64_t size, int iterations) {
    uint64_t shape_a[] = {size};
    uint64_t shape_b[] = {size / 2};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F32);
    uint64_t shape_result[] = {size * (size / 2)};
    vsla_tensor_t* result = vsla_tensor_create(ctx, 1, shape_result, VSLA_MODEL_A, VSLA_DTYPE_F32);

    vsla_fill(ctx, a, 1.0f);
    vsla_fill(ctx, b, 2.0f);

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start_time = get_time_ms();
        vsla_kron(ctx, result, a, b);
        total_time += get_time_ms() - start_time;
    }

    printf("Kron operation with size %lu: %.6f ms per iteration\n", size, total_time / iterations);

    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(result);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <size> <iterations>\n", argv[0]);
        return 1;
    }

    uint64_t size = atol(argv[1]);
    int iterations = atoi(argv[2]);

    // Initialize VSLA context
    vsla_config_t config = { .backend = VSLA_BACKEND_AUTO };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    benchmark_add(ctx, size, iterations);
    benchmark_sub(ctx, size, iterations);
    benchmark_scale(ctx, size, iterations);
    benchmark_hadamard(ctx, size, iterations);
    benchmark_conv(ctx, size, iterations);
    benchmark_kron(ctx, size, iterations);

    vsla_cleanup(ctx);

    return 0;
}