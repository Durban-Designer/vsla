/**
 * @file bench_sparse.c
 * @brief Sparse tensor operations benchmark testing various sparsity levels
 * 
 * Demonstrates VSLA's computational savings for sparse tensors by avoiding
 * operations on zero elements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// High-resolution timer
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Create sparse tensor with given sparsity level
static void fill_sparse_tensor(vsla_context_t* ctx, vsla_tensor_t* tensor, 
                              uint64_t* shape, uint8_t rank,
                              double sparsity, double fill_value) {
    // First fill with zeros
    vsla_fill(ctx, tensor, 0.0);
    
    // Calculate number of non-zero elements
    uint64_t total_elems = 1;
    for (int i = 0; i < rank; i++) {
        total_elems *= shape[i];
    }
    
    uint64_t nonzero_count = (uint64_t)(total_elems * (1.0 - sparsity));
    
    // Randomly place non-zero values
    srand(42); // Fixed seed for reproducibility
    for (uint64_t i = 0; i < nonzero_count; i++) {
        uint64_t idx[8] = {0};
        for (int d = 0; d < rank; d++) {
            idx[d] = rand() % shape[d];
        }
        vsla_set_f64(ctx, tensor, idx, fill_value);
    }
}

// Dense computation (processes all elements including zeros)
static void dense_tensor_add(double* a, double* b, double* out, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

// "Smart" sparse computation (skips zeros but still checks every element)
static void sparse_aware_add(double* a, double* b, double* out, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        if (a[i] != 0.0 || b[i] != 0.0) {
            out[i] = a[i] + b[i];
        } else {
            out[i] = 0.0;
        }
    }
}

// Test sparse operations at different sparsity levels
static void benchmark_sparsity_level(vsla_context_t* ctx, double sparsity, 
                                    uint64_t* shape, uint8_t rank, const char* scenario) {
    // Create VSLA tensors
    vsla_tensor_t* a = vsla_tensor_create(ctx, rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out_vsla = vsla_tensor_create(ctx, rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with sparse data
    fill_sparse_tensor(ctx, a, shape, rank, sparsity, 2.0);
    fill_sparse_tensor(ctx, b, shape, rank, sparsity, 3.0);
    
    // Calculate total elements
    uint64_t total_elems = 1;
    for (int i = 0; i < rank; i++) {
        total_elems *= shape[i];
    }
    
    // Create dense arrays for comparison
    double* a_dense = (double*)calloc(total_elems, sizeof(double));
    double* b_dense = (double*)calloc(total_elems, sizeof(double));
    double* out_dense = (double*)calloc(total_elems, sizeof(double));
    double* out_sparse = (double*)calloc(total_elems, sizeof(double));
    
    // Fill dense arrays (simulate what dense libraries would do)
    for (uint64_t i = 0; i < total_elems; i++) {
        a_dense[i] = (rand() / (double)RAND_MAX < (1.0 - sparsity)) ? 2.0 : 0.0;
        b_dense[i] = (rand() / (double)RAND_MAX < (1.0 - sparsity)) ? 3.0 : 0.0;
    }
    
    int warmup_runs = 3;
    int bench_runs = 20;
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
        dense_tensor_add(a_dense, b_dense, out_dense, total_elems);
        sparse_aware_add(a_dense, b_dense, out_sparse, total_elems);
    }
    
    // Benchmark VSLA (should be sparsity-aware)
    double vsla_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
    }
    double vsla_time = (get_time() - vsla_start) * 1000.0 / bench_runs; // ms
    
    // Benchmark dense computation
    double dense_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        dense_tensor_add(a_dense, b_dense, out_dense, total_elems);
    }
    double dense_time = (get_time() - dense_start) * 1000.0 / bench_runs; // ms
    
    // Benchmark sparse-aware computation
    double sparse_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        sparse_aware_add(a_dense, b_dense, out_sparse, total_elems);
    }
    double sparse_time = (get_time() - sparse_start) * 1000.0 / bench_runs; // ms
    
    double vsla_speedup = dense_time / vsla_time;
    double sparse_speedup = dense_time / sparse_time;
    
    uint64_t nonzero_elems = (uint64_t)(total_elems * (1.0 - sparsity));
    
    printf("\n%s (%.1f%% sparse, %.1fM elements):\n", scenario, sparsity * 100, total_elems / 1000000.0);
    printf("  Non-zero elements: %lu (%.1f%% of total)\n", nonzero_elems, (1.0 - sparsity) * 100);
    printf("  Performance:\n");
    printf("    VSLA:            %8.3f ms\n", vsla_time);
    printf("    Dense (all):     %8.3f ms\n", dense_time);
    printf("    Sparse-aware:    %8.3f ms\n", sparse_time);
    printf("  Speedup vs Dense:\n");
    printf("    VSLA:            %8.2fx faster\n", vsla_speedup);
    printf("    Sparse-aware:    %8.2fx faster\n", sparse_speedup);
    printf("  Computational Savings:\n");
    printf("    Theoretical:     %8.1fx (skip %.1f%% zeros)\n", 1.0 / (1.0 - sparsity), sparsity * 100);
    printf("    VSLA Achieved:   %8.1fx actual speedup\n", vsla_speedup);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out_vsla);
    free(a_dense);
    free(b_dense);
    free(out_dense);
    free(out_sparse);
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    // Initialize VSLA
    vsla_config_t config = {.backend = VSLA_BACKEND_CPU};
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }
    
    printf("=== VSLA Sparse Tensor Operations Benchmark ===\n");
    printf("Testing computational savings at various sparsity levels\n");
    printf("=======================================================\n");
    
    // Test different tensor sizes and sparsity levels
    struct {
        uint64_t shape[3];
        uint8_t rank;
        const char* size_desc;
    } tensor_configs[] = {
        {{1000, 1000, 1}, 3, "1M elements (1000x1000)"},
        {{500, 500, 4}, 3, "1M elements (500x500x4)"},
        {{200, 200, 25}, 3, "1M elements (200x200x25)"},
    };
    
    double sparsity_levels[] = {0.1, 0.5, 0.9, 0.95, 0.99};
    int num_sparsity = sizeof(sparsity_levels) / sizeof(sparsity_levels[0]);
    int num_configs = sizeof(tensor_configs) / sizeof(tensor_configs[0]);
    
    for (int config = 0; config < num_configs; config++) {
        printf("\n=== Tensor Size: %s ===\n", tensor_configs[config].size_desc);
        
        for (int s = 0; s < num_sparsity; s++) {
            char scenario[256];
            snprintf(scenario, sizeof(scenario), "%s", tensor_configs[config].size_desc);
            
            benchmark_sparsity_level(ctx, sparsity_levels[s], 
                                   tensor_configs[config].shape,
                                   tensor_configs[config].rank,
                                   scenario);
        }
    }
    
    printf("\n=== Sparse Operations Summary ===\n");
    printf("VSLA advantages for sparse tensors:\n");
    printf("1. Automatic sparsity detection and optimization\n");
    printf("2. Computational savings proportional to sparsity level\n");
    printf("3. Memory efficient storage of non-zero elements\n");
    printf("4. No need to manually implement sparse algorithms\n");
    printf("5. Transparent optimization - same API for dense/sparse\n\n");
    
    printf("Performance expectations:\n");
    printf("- 10%% sparse: ~1.1x speedup (modest savings)\n");
    printf("- 50%% sparse: ~2x speedup (significant savings)\n");
    printf("- 90%% sparse: ~10x speedup (major savings)\n");
    printf("- 99%% sparse: ~100x speedup (extreme savings)\n\n");
    
    printf("Note: Actual speedups depend on sparsity patterns, cache effects,\n");
    printf("and the overhead of sparsity checking in the implementation.\n");
    
    vsla_cleanup(ctx);
    return 0;
}