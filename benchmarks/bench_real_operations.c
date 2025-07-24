/**
 * @file bench_real_operations.c
 * @brief Comprehensive benchmarks using real VSLA operations
 * 
 * This benchmark uses the actual implemented VSLA operations including
 * matrix multiplication, convolution, and optimized arithmetic to
 * demonstrate real-world performance with authentic workloads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// High-resolution timer
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Performance tracking structure
typedef struct {
    double total_time;
    double min_time;
    double max_time;
    uint64_t operations;
    size_t data_size;
} perf_stats_t;

static void init_perf_stats(perf_stats_t* stats) {
    stats->total_time = 0.0;
    stats->min_time = 1e9;
    stats->max_time = 0.0;
    stats->operations = 0;
    stats->data_size = 0;
}

static void update_perf_stats(perf_stats_t* stats, double time, uint64_t ops, size_t bytes) {
    stats->total_time += time;
    if (time < stats->min_time) stats->min_time = time;
    if (time > stats->max_time) stats->max_time = time;
    stats->operations += ops;
    stats->data_size += bytes;
}

static void print_perf_stats(const char* test_name, const perf_stats_t* stats, int iterations) {
    double avg_time = stats->total_time / iterations;
    double throughput = (stats->operations / 1e9) / stats->total_time; // GFLOPS
    double bandwidth = (stats->data_size / 1e9) / stats->total_time; // GB/s
    
    printf("\n=== %s Performance ===\n", test_name);
    printf("  Average time:     %.3f ms\n", avg_time * 1000);
    printf("  Min time:         %.3f ms\n", stats->min_time * 1000);
    printf("  Max time:         %.3f ms\n", stats->max_time * 1000);
    printf("  Throughput:       %.2f GFLOPS\n", throughput);
    printf("  Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Total operations: %lu\n", stats->operations);
    printf("  Total data:       %.2f MB\n", stats->data_size / 1e6);
}

/**
 * Test real matrix multiplication with various sizes
 */
static void benchmark_real_matmul(vsla_context_t* ctx) {
    printf("\n🔢 Testing Real Matrix Multiplication Operations\n");
    
    typedef struct {
        uint64_t m, k, n;
        const char* description;
    } matmul_test_t;
    
    const matmul_test_t tests[] = {
        {64, 64, 64, "Small matrices (deep learning layer)"},
        {128, 256, 128, "Medium matrices (transformer attention)"},
        {256, 512, 256, "Large matrices (dense layer)"},
        {512, 1024, 512, "Very large matrices (embedding layer)"},
        {1024, 2048, 512, "Asymmetric matrices (projection layer)"}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    const int iterations = 10;
    
    for (int t = 0; t < num_tests; t++) {
        const matmul_test_t* test = &tests[t];
        printf("\n--- %s: [%lu,%lu] @ [%lu,%lu] ---\n", 
               test->description, test->m, test->k, test->k, test->n);
        
        // Create matrices
        uint64_t a_shape[] = {test->m, test->k};
        uint64_t b_shape[] = {test->k, test->n};
        uint64_t c_shape[] = {test->m, test->n};
        
        vsla_tensor_t* A = vsla_tensor_create(ctx, 2, a_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* B = vsla_tensor_create(ctx, 2, b_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* C = vsla_tensor_create(ctx, 2, c_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!A || !B || !C) {
            printf("  ERROR: Failed to create matrices\n");
            continue;
        }
        
        // Initialize with test data
        vsla_fill(ctx, A, 1.0);
        vsla_fill(ctx, B, 0.5);
        vsla_fill(ctx, C, 0.0);
        
        // Benchmark real matrix multiplication
        perf_stats_t stats;
        init_perf_stats(&stats);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            vsla_matmul(ctx, C, A, B);
        }
        
        // Actual benchmark
        for (int i = 0; i < iterations; i++) {
            double start = get_time();
            vsla_error_t result = vsla_matmul(ctx, C, A, B);
            double end = get_time();
            
            if (result != VSLA_SUCCESS) {
                printf("  ERROR: Matrix multiplication failed with error %d\n", result);
                break;
            }
            
            uint64_t ops = 2 * test->m * test->k * test->n; // FLOPS for matmul
            size_t bytes = (test->m * test->k + test->k * test->n + test->m * test->n) * sizeof(double);
            update_perf_stats(&stats, end - start, ops, bytes);
        }
        
        print_perf_stats("Matrix Multiplication", &stats, iterations);
        
        // Verify correctness (spot check)
        size_t data_size;
        double* c_data = (double*)vsla_tensor_data(C, &data_size);
        if (c_data && test->m > 0 && test->n > 0) {
            double expected = test->k * 1.0 * 0.5; // sum of products
            printf("  Correctness check: C[0,0] = %.6f (expected: %.6f) %s\n", 
                   c_data[0], expected, 
                   fabs(c_data[0] - expected) < 1e-10 ? "✓" : "✗");
        }
        
        vsla_tensor_free(A);
        vsla_tensor_free(B);
        vsla_tensor_free(C);
    }
}

/**
 * Test real convolution operations
 */
static void benchmark_real_convolution(vsla_context_t* ctx) {
    printf("\n🔄 Testing Real Convolution Operations\n");
    
    typedef struct {
        uint64_t signal_len, kernel_len;
        const char* description;
    } conv_test_t;
    
    const conv_test_t tests[] = {
        {32, 3, "Small 1D convolution (edge detection)"},
        {128, 5, "Medium 1D convolution (feature extraction)"},
        {512, 7, "Large 1D convolution (audio processing)"},
        {1024, 11, "Very large 1D convolution (signal processing)"},
        {2048, 21, "Ultra large 1D convolution (time series)"}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    const int iterations = 20;
    
    for (int t = 0; t < num_tests; t++) {
        const conv_test_t* test = &tests[t];
        uint64_t output_len = test->signal_len + test->kernel_len - 1;
        
        printf("\n--- %s: signal[%lu] * kernel[%lu] -> output[%lu] ---\n", 
               test->description, test->signal_len, test->kernel_len, output_len);
        
        // Create tensors
        uint64_t signal_shape[] = {test->signal_len};
        uint64_t kernel_shape[] = {test->kernel_len};
        uint64_t output_shape[] = {output_len};
        
        vsla_tensor_t* signal = vsla_tensor_create(ctx, 1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* kernel = vsla_tensor_create(ctx, 1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* output = vsla_tensor_create(ctx, 1, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!signal || !kernel || !output) {
            printf("  ERROR: Failed to create convolution tensors\n");
            continue;
        }
        
        // Initialize with test data
        size_t data_size;
        double* signal_data = (double*)vsla_tensor_data_mut(signal, &data_size);
        double* kernel_data = (double*)vsla_tensor_data_mut(kernel, &data_size);
        
        // Create realistic test signal (sine wave + noise)
        for (uint64_t i = 0; i < test->signal_len; i++) {
            signal_data[i] = sin(2.0 * M_PI * i * 3.0 / test->signal_len) + 
                           0.1 * sin(2.0 * M_PI * i * 17.0 / test->signal_len);
        }
        
        // Create test kernel (Gaussian-like)
        double sum = 0.0;
        for (uint64_t i = 0; i < test->kernel_len; i++) {
            double x = (double)i - (double)(test->kernel_len - 1) / 2.0;
            kernel_data[i] = exp(-x * x / (2.0 * (test->kernel_len / 6.0) * (test->kernel_len / 6.0)));
            sum += kernel_data[i];
        }
        // Normalize kernel
        for (uint64_t i = 0; i < test->kernel_len; i++) {
            kernel_data[i] /= sum;
        }
        
        // Benchmark real convolution
        perf_stats_t stats;
        init_perf_stats(&stats);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            vsla_conv(ctx, output, signal, kernel);
        }
        
        // Actual benchmark
        for (int i = 0; i < iterations; i++) {
            double start = get_time();
            vsla_error_t result = vsla_conv(ctx, output, signal, kernel);
            double end = get_time();
            
            if (result != VSLA_SUCCESS) {
                printf("  ERROR: Convolution failed with error %d\n", result);
                break;
            }
            
            uint64_t ops = test->signal_len * test->kernel_len; // Approximate FLOPS
            size_t bytes = (test->signal_len + test->kernel_len + output_len) * sizeof(double);
            update_perf_stats(&stats, end - start, ops, bytes);
        }
        
        print_perf_stats("1D Convolution", &stats, iterations);
        
        // Verify correctness (spot check)
        double* output_data = (double*)vsla_tensor_data(output, &data_size);
        if (output_data) {
            printf("  Output sample: [%.6f, %.6f, %.6f, ..., %.6f]\n", 
                   output_data[0], 
                   output_len > 1 ? output_data[1] : 0.0,
                   output_len > 2 ? output_data[2] : 0.0,
                   output_data[output_len - 1]);
        }
        
        vsla_tensor_free(signal);
        vsla_tensor_free(kernel);
        vsla_tensor_free(output);
    }
}

/**
 * Test optimized broadcasting operations
 */
static void benchmark_optimized_broadcasting(vsla_context_t* ctx) {
    printf("\n📡 Testing Optimized Broadcasting Operations\n");
    
    typedef struct {
        uint64_t* shape_a;
        uint64_t* shape_b;
        uint64_t* shape_out;
        uint8_t rank;
        const char* description;
        const char* pattern;
    } broadcast_test_t;
    
    // Test configurations
    uint64_t equal_2d_a[] = {1000, 1000};
    uint64_t equal_2d_b[] = {1000, 1000};
    uint64_t equal_2d_out[] = {1000, 1000};
    
    uint64_t scalar_2d_a[] = {500, 500};
    uint64_t scalar_2d_b[] = {1, 1};
    uint64_t scalar_2d_out[] = {500, 500};
    
    uint64_t row_2d_a[] = {200, 1000};
    uint64_t row_2d_b[] = {1, 1000};
    uint64_t row_2d_out[] = {200, 1000};
    
    uint64_t col_2d_a[] = {1000, 200};
    uint64_t col_2d_b[] = {1000, 1};
    uint64_t col_2d_out[] = {1000, 200};
    
    uint64_t cnn_4d_a[] = {32, 64, 56, 56};
    uint64_t cnn_4d_b[] = {1, 64, 56, 56};
    uint64_t cnn_4d_out[] = {32, 64, 56, 56};
    
    const broadcast_test_t tests[] = {
        {equal_2d_a, equal_2d_b, equal_2d_out, 2, "Equal shapes (vectorized)", "EQUAL_SHAPES"},
        {scalar_2d_a, scalar_2d_b, scalar_2d_out, 2, "Scalar broadcasting", "SCALAR_BROADCAST"},
        {row_2d_a, row_2d_b, row_2d_out, 2, "2D row broadcasting", "ROW_BROADCAST"},
        {col_2d_a, col_2d_b, col_2d_out, 2, "2D column broadcasting", "COL_BROADCAST"},
        {cnn_4d_a, cnn_4d_b, cnn_4d_out, 4, "4D CNN batch broadcasting", "4D_BATCH"}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    const int iterations = 15;
    
    for (int t = 0; t < num_tests; t++) {
        const broadcast_test_t* test = &tests[t];
        printf("\n--- %s (%s) ---\n", test->description, test->pattern);
        
        // Create tensors
        vsla_tensor_t* A = vsla_tensor_create(ctx, test->rank, test->shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* B = vsla_tensor_create(ctx, test->rank, test->shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* C = vsla_tensor_create(ctx, test->rank, test->shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!A || !B || !C) {
            printf("  ERROR: Failed to create broadcasting tensors\n");
            continue;
        }
        
        // Initialize with test data
        vsla_fill(ctx, A, 2.0);
        vsla_fill(ctx, B, 3.0);
        vsla_fill(ctx, C, 0.0);
        
        // Calculate element counts
        uint64_t a_elems = 1, b_elems = 1, out_elems = 1;
        for (uint8_t i = 0; i < test->rank; i++) {
            a_elems *= test->shape_a[i];
            b_elems *= test->shape_b[i];
            out_elems *= test->shape_out[i];
        }
        
        printf("  Tensor shapes: A");
        for (uint8_t i = 0; i < test->rank; i++) printf("[%lu]", test->shape_a[i]);
        printf(" + B");
        for (uint8_t i = 0; i < test->rank; i++) printf("[%lu]", test->shape_b[i]);
        printf(" -> C");
        for (uint8_t i = 0; i < test->rank; i++) printf("[%lu]", test->shape_out[i]);
        printf("\n");
        
        // Benchmark optimized broadcasting
        perf_stats_t stats;
        init_perf_stats(&stats);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            vsla_add(ctx, C, A, B);
        }
        
        // Actual benchmark
        for (int i = 0; i < iterations; i++) {
            double start = get_time();
            vsla_error_t result = vsla_add(ctx, C, A, B);
            double end = get_time();
            
            if (result != VSLA_SUCCESS) {
                printf("  ERROR: Broadcasting addition failed with error %d\n", result);
                break;
            }
            
            uint64_t ops = out_elems; // One addition per output element
            size_t bytes = (a_elems + b_elems + out_elems) * sizeof(double);
            update_perf_stats(&stats, end - start, ops, bytes);
        }
        
        print_perf_stats("Optimized Broadcasting", &stats, iterations);
        
        // Verify correctness
        size_t data_size;
        double* c_data = (double*)vsla_tensor_data(C, &data_size);
        if (c_data) {
            double expected = 2.0 + 3.0; // A + B
            printf("  Correctness check: C[0] = %.6f (expected: %.6f) %s\n", 
                   c_data[0], expected, 
                   fabs(c_data[0] - expected) < 1e-10 ? "✓" : "✗");
        }
        
        vsla_tensor_free(A);
        vsla_tensor_free(B);
        vsla_tensor_free(C);
    }
}

/**
 * Test composite deep learning operations
 */
static void benchmark_composite_dl_operations(vsla_context_t* ctx) {
    printf("\n🧠 Testing Composite Deep Learning Operations\n");
    
    // Simulate a simple neural network layer: input -> matmul -> bias_add -> activation
    printf("\n--- Neural Network Layer Simulation ---\n");
    printf("Simulating: Dense layer with batch processing\n");
    
    const uint64_t batch_size = 64;
    const uint64_t input_dim = 512;
    const uint64_t output_dim = 256;
    const int iterations = 10;
    
    // Create tensors
    uint64_t input_shape[] = {batch_size, input_dim};
    uint64_t weight_shape[] = {input_dim, output_dim};
    uint64_t bias_shape[] = {1, output_dim}; // Same rank for broadcasting
    uint64_t output_shape[] = {batch_size, output_dim};
    uint64_t temp_shape[] = {batch_size, output_dim};
    
    vsla_tensor_t* input = vsla_tensor_create(ctx, 2, input_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* weights = vsla_tensor_create(ctx, 2, weight_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* bias = vsla_tensor_create(ctx, 2, bias_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* temp = vsla_tensor_create(ctx, 2, temp_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* output = vsla_tensor_create(ctx, 2, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!input || !weights || !bias || !temp || !output) {
        printf("  ERROR: Failed to create neural network tensors\n");
        return;
    }
    
    // Initialize with realistic values
    vsla_fill(ctx, input, 0.5);
    vsla_fill(ctx, weights, 0.1);
    vsla_fill(ctx, bias, 0.01);
    
    // Benchmark composite operation
    perf_stats_t stats;
    init_perf_stats(&stats);
    
    printf("  Layer configuration: [%lu,%lu] @ [%lu,%lu] + bias -> [%lu,%lu]\n",
           batch_size, input_dim, input_dim, output_dim, batch_size, output_dim);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        vsla_matmul(ctx, temp, input, weights);
        vsla_add(ctx, output, temp, bias);
    }
    
    // Actual benchmark
    for (int i = 0; i < iterations; i++) {
        double start = get_time();
        
        // 1. Matrix multiplication: input @ weights
        vsla_error_t result1 = vsla_matmul(ctx, temp, input, weights);
        
        // 2. Bias addition (broadcasting): temp + bias
        vsla_error_t result2 = vsla_add(ctx, output, temp, bias);
        
        double end = get_time();
        
        if (result1 != VSLA_SUCCESS || result2 != VSLA_SUCCESS) {
            printf("  ERROR: Composite operation failed\n");
            break;
        }
        
        // Calculate total operations
        uint64_t matmul_ops = 2 * batch_size * input_dim * output_dim;
        uint64_t add_ops = batch_size * output_dim;
        uint64_t total_ops = matmul_ops + add_ops;
        
        size_t total_bytes = (batch_size * input_dim + input_dim * output_dim + 
                             output_dim + batch_size * output_dim * 2) * sizeof(double);
        
        update_perf_stats(&stats, end - start, total_ops, total_bytes);
    }
    
    print_perf_stats("Composite Neural Network Layer", &stats, iterations);
    
    // Verify correctness
    size_t data_size;
    double* output_data = (double*)vsla_tensor_data(output, &data_size);
    if (output_data) {
        double expected_linear = input_dim * 0.5 * 0.1; // matmul result
        double expected_final = expected_linear + 0.01; // + bias
        printf("  Correctness check: output[0,0] = %.6f (expected: ~%.6f) %s\n", 
               output_data[0], expected_final,
               fabs(output_data[0] - expected_final) < 0.1 ? "✓" : "✗");
    }
    
    // Cleanup
    vsla_tensor_free(input);
    vsla_tensor_free(weights);
    vsla_tensor_free(bias);
    vsla_tensor_free(temp);
    vsla_tensor_free(output);
}

int main() {
    printf("🚀 VSLA Real Operations Benchmark Suite\n");
    printf("========================================\n");
    printf("Testing actual VSLA implementations with realistic workloads\n");
    
    // Initialize VSLA
    vsla_config_t config = {
        .backend = VSLA_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_THROUGHPUT,
        .enable_profiling = false,
        .verbose = false
    };
    
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("ERROR: Failed to initialize VSLA context\n");
        return 1;
    }
    
    printf("\nRunning comprehensive benchmarks with real VSLA operations...\n");
    
    // Run benchmark suite
    benchmark_real_matmul(ctx);
    benchmark_real_convolution(ctx);
    benchmark_optimized_broadcasting(ctx);
    benchmark_composite_dl_operations(ctx);
    
    printf("\n🎯 Summary\n");
    printf("===========\n");
    printf("✅ Matrix multiplication: Using real vsla_matmul() with VSLA semantics\n");
    printf("✅ Convolution: Using real vsla_conv() with Model A semiring\n");
    printf("✅ Broadcasting: Using optimized dispatch with SIMD vectorization\n");
    printf("✅ Composite operations: Realistic deep learning layer simulation\n");
    printf("✅ Memory efficiency: Variable-shape tensors with minimal representatives\n");
    printf("✅ Performance: Cache-optimized kernels with intelligent dispatch\n");
    
    // Cleanup
    vsla_cleanup(ctx);
    
    printf("\n🎉 VSLA Real Operations Benchmark Complete!\n");
    return 0;
}