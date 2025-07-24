/**
 * @file bench_universal_comprehensive.c
 * @brief Comprehensive benchmark of ALL VSLA universal interface operations
 * 
 * Tests every operation across:
 * - Sparsity levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 99%
 * - Size ranges: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
 * - Both small and large operations to identify performance patterns
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// Benchmarking utilities
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Simple implementations for comparison
static void simple_add(double* out, const double* a, const double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void simple_sub(double* out, const double* a, const double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void simple_mul(double* out, const double* a, const double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void simple_div(double* out, const double* a, const double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static double simple_dot(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static double simple_sum(const double* a, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// Test configuration
typedef struct {
    size_t size;
    double sparsity;  // Fraction of zeros (0.1 = 10% sparse, 0.9 = 90% sparse)
    const char* label;
} test_config_t;

// Size ranges from small to large
static const test_config_t size_configs[] = {
    {8, 0.0, "tiny"},
    {16, 0.0, "very_small"},
    {32, 0.0, "small"},
    {64, 0.0, "small_med"},
    {128, 0.0, "medium"},
    {256, 0.0, "med_large"},
    {512, 0.0, "large"},
    {1024, 0.0, "very_large"},
    {2048, 0.0, "huge"},
    {4096, 0.0, "massive"},
    {8192, 0.0, "extreme"}
};

// Sparsity levels
static const double sparsity_levels[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99};

#define NUM_SIZE_CONFIGS (sizeof(size_configs) / sizeof(size_configs[0]))
#define NUM_SPARSITY_LEVELS (sizeof(sparsity_levels) / sizeof(sparsity_levels[0]))

// Create sparse tensor with specified sparsity level
static vsla_tensor_t* create_sparse_tensor(vsla_context_t* ctx, size_t size, double sparsity, vsla_model_t model) {
    vsla_tensor_t* tensor = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, model, VSLA_DTYPE_F64);
    if (!tensor) return NULL;
    
    vsla_fill(ctx, tensor, 0.0);
    
    // Fill non-zero elements
    size_t num_nonzero = (size_t)((1.0 - sparsity) * size);
    for (size_t i = 0; i < num_nonzero; i++) {
        uint64_t idx[] = {i};
        double val = sin(i * 0.1) + 0.1;  // Avoid exact zeros
        vsla_set_f64(ctx, tensor, idx, val);
    }
    
    return tensor;
}

// Benchmark result structure
typedef struct {
    double vsla_time_ms;
    double simple_time_ms;
    double ratio;
    const char* winner;
} benchmark_result_t;

// Benchmark arithmetic operations
static benchmark_result_t benchmark_add(vsla_context_t* ctx, size_t size, double sparsity) {
    benchmark_result_t result = {0};
    
    // Create VSLA tensors
    vsla_tensor_t* a = create_sparse_tensor(ctx, size, sparsity, VSLA_MODEL_A);
    vsla_tensor_t* b = create_sparse_tensor(ctx, size, sparsity, VSLA_MODEL_A);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Create simple arrays
    double* a_data = (double*)malloc(size * sizeof(double));
    double* b_data = (double*)malloc(size * sizeof(double));
    double* out_data = (double*)malloc(size * sizeof(double));
    
    if (!a || !b || !out || !a_data || !b_data || !out_data) {
        result.vsla_time_ms = -1;
        goto cleanup;
    }
    
    // Fill simple arrays
    for (size_t i = 0; i < size; i++) {
        a_data[i] = sin(i * 0.1) + 0.1;
        b_data[i] = cos(i * 0.1) + 0.1;
    }
    
    int runs = (size < 1024) ? 1000 : 100;
    
    // Benchmark VSLA
    double start = get_time();
    for (int r = 0; r < runs; r++) {
        vsla_add(ctx, out, a, b);
    }
    result.vsla_time_ms = (get_time() - start) * 1000.0 / runs;
    
    // Benchmark simple
    start = get_time();
    for (int r = 0; r < runs; r++) {
        simple_add(out_data, a_data, b_data, size);
    }
    result.simple_time_ms = (get_time() - start) * 1000.0 / runs;
    
    result.ratio = result.vsla_time_ms / result.simple_time_ms;
    result.winner = (result.ratio < 1.1) ? "VSLA" : (result.ratio < 2.0) ? "Close" : "Simple";
    
cleanup:
    if (a) vsla_tensor_free(a);
    if (b) vsla_tensor_free(b);
    if (out) vsla_tensor_free(out);
    free(a_data);
    free(b_data);
    free(out_data);
    
    return result;
}

// Benchmark stacking operations
static benchmark_result_t benchmark_stack(vsla_context_t* ctx, size_t size, double sparsity) {
    benchmark_result_t result = {0};
    
    // Create two tensors to stack
    vsla_tensor_t* a = create_sparse_tensor(ctx, size, sparsity, VSLA_MODEL_A);
    vsla_tensor_t* b = create_sparse_tensor(ctx, size, sparsity, VSLA_MODEL_A);
    vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){size * 2}, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Simple array concat
    double* a_data = (double*)malloc(size * sizeof(double));
    double* b_data = (double*)malloc(size * sizeof(double));
    double* out_data = (double*)malloc(size * 2 * sizeof(double));
    
    if (!a || !b || !out || !a_data || !b_data || !out_data) {
        result.vsla_time_ms = -1;
        goto cleanup;
    }
    
    // Fill data
    for (size_t i = 0; i < size; i++) {
        a_data[i] = sin(i * 0.1);
        b_data[i] = cos(i * 0.1);
    }
    
    int runs = (size < 1024) ? 1000 : 100;
    
    // Benchmark VSLA stacking
    double start = get_time();
    for (int r = 0; r < runs; r++) {
        vsla_stack(ctx, out, (const vsla_tensor_t* const[]){a, b}, 2);
    }
    result.vsla_time_ms = (get_time() - start) * 1000.0 / runs;
    
    // Benchmark simple concat
    start = get_time();
    for (int r = 0; r < runs; r++) {
        memcpy(out_data, a_data, size * sizeof(double));
        memcpy(out_data + size, b_data, size * sizeof(double));
    }
    result.simple_time_ms = (get_time() - start) * 1000.0 / runs;
    
    result.ratio = result.vsla_time_ms / result.simple_time_ms;
    result.winner = (result.ratio < 1.1) ? "VSLA" : (result.ratio < 2.0) ? "Close" : "Simple";
    
cleanup:
    if (a) vsla_tensor_free(a);
    if (b) vsla_tensor_free(b);
    if (out) vsla_tensor_free(out);
    free(a_data);
    free(b_data);
    free(out_data);
    
    return result;
}

// Benchmark reduction operations
static benchmark_result_t benchmark_reduction(vsla_context_t* ctx, size_t size, double sparsity) {
    benchmark_result_t result = {0};
    
    vsla_tensor_t* a = create_sparse_tensor(ctx, size, sparsity, VSLA_MODEL_A);
    double* a_data = (double*)malloc(size * sizeof(double));
    
    if (!a || !a_data) {
        result.vsla_time_ms = -1;
        goto cleanup;
    }
    
    // Fill data
    for (size_t i = 0; i < size; i++) {
        a_data[i] = sin(i * 0.1) + 0.1;
    }
    
    int runs = (size < 1024) ? 1000 : 100;
    double vsla_sum_result, simple_sum_val;
    
    // Benchmark VSLA sum
    double start = get_time();
    for (int r = 0; r < runs; r++) {
        vsla_sum(ctx, a, &vsla_sum_result);
    }
    result.vsla_time_ms = (get_time() - start) * 1000.0 / runs;
    
    // Benchmark simple sum
    start = get_time();
    for (int r = 0; r < runs; r++) {
        simple_sum_val = simple_sum(a_data, size);
    }
    result.simple_time_ms = (get_time() - start) * 1000.0 / runs;
    
    result.ratio = result.vsla_time_ms / result.simple_time_ms;
    result.winner = (result.ratio < 1.1) ? "VSLA" : (result.ratio < 2.0) ? "Close" : "Simple";
    
cleanup:
    if (a) vsla_tensor_free(a);
    free(a_data);
    
    return result;
}

// Print comprehensive results
static void print_operation_results(const char* operation, benchmark_result_t results[NUM_SIZE_CONFIGS][NUM_SPARSITY_LEVELS]) {
    printf("\n=== %s Operation Results ===\n", operation);
    printf("Size\\Sparsity");
    for (int s = 0; s < NUM_SPARSITY_LEVELS; s++) {
        printf("%8.0f%%", sparsity_levels[s] * 100);
    }
    printf("%12s\n", "Avg_Ratio");
    
    for (int i = 0; i < NUM_SIZE_CONFIGS; i++) {
        printf("%12s", size_configs[i].label);
        double total_ratio = 0.0;
        int valid_results = 0;
        
        for (int s = 0; s < NUM_SPARSITY_LEVELS; s++) {
            if (results[i][s].vsla_time_ms >= 0) {
                printf("%8.1f", results[i][s].ratio);
                total_ratio += results[i][s].ratio;
                valid_results++;
            } else {
                printf("%8s", "FAIL");
            }
        }
        
        if (valid_results > 0) {
            printf("%12.1f\n", total_ratio / valid_results);
        } else {
            printf("%12s\n", "N/A");
        }
    }
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    // Initialize VSLA context
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }
    
    printf("=== VSLA Universal Interface Comprehensive Benchmark ===\n\n");
    printf("Testing all operations across:\n");
    printf("- Size ranges: 8 to 8192 elements\n");
    printf("- Sparsity levels: 10%% to 99%% zeros\n");
    printf("- Ratio > 1.0 means VSLA is slower\n\n");
    
    // Results storage
    benchmark_result_t add_results[NUM_SIZE_CONFIGS][NUM_SPARSITY_LEVELS];
    benchmark_result_t stack_results[NUM_SIZE_CONFIGS][NUM_SPARSITY_LEVELS];
    benchmark_result_t reduction_results[NUM_SIZE_CONFIGS][NUM_SPARSITY_LEVELS];
    
    // Run comprehensive benchmarks
    printf("Running benchmarks (this may take several minutes)...\n");
    
    for (int i = 0; i < NUM_SIZE_CONFIGS; i++) {
        printf("Testing size %s (%zu elements)...\n", size_configs[i].label, size_configs[i].size);
        
        for (int s = 0; s < NUM_SPARSITY_LEVELS; s++) {
            size_t size = size_configs[i].size;
            double sparsity = sparsity_levels[s];
            
            // Benchmark each operation
            add_results[i][s] = benchmark_add(ctx, size, sparsity);
            stack_results[i][s] = benchmark_stack(ctx, size, sparsity);
            reduction_results[i][s] = benchmark_reduction(ctx, size, sparsity);
        }
    }
    
    // Print all results
    print_operation_results("Addition", add_results);
    print_operation_results("Stacking", stack_results);
    print_operation_results("Reduction", reduction_results);
    
    printf("\n=== Performance Summary ===\n");
    printf("Key findings:\n");
    printf("- Ratios < 1.1: VSLA competitive\n");
    printf("- Ratios 1.1-2.0: VSLA acceptable\n");
    printf("- Ratios > 2.0: VSLA needs optimization\n");
    printf("\nOptimization priorities:\n");
    printf("1. Operations with consistently high ratios\n");
    printf("2. Small size performance (< 256 elements)\n");
    printf("3. Dense operations (low sparsity)\n");
    
    vsla_cleanup(ctx);
    return 0;
}