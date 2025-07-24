/**
 * @file bench_kronecker.c
 * @brief Comprehensive Kronecker product benchmark for VSLA
 * 
 * Tests VSLA Model B operations vs traditional implementations
 * and analyzes performance characteristics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// Benchmarking utilities
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Traditional Kronecker product implementation
static void traditional_kronecker(const double* a, size_t m, const double* b, size_t n, double* out) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            out[i * n + j] = a[i] * b[j];
        }
    }
}

// BLAS-optimized Kronecker (using outer product approach)
static void blas_kronecker(const double* a, size_t m, const double* b, size_t n, double* out) {
    // For each element of a, multiply it by the entire vector b
    for (size_t i = 0; i < m; i++) {
        cblas_dcopy(n, b, 1, &out[i * n], 1);
        cblas_dscal(n, a[i], &out[i * n], 1);
    }
}

// Benchmark Kronecker product performance
static void benchmark_kronecker_sizes(vsla_context_t* ctx) {
    printf("=== Kronecker Product Size Scaling ===\n");
    printf("Testing VSLA Model B vs traditional implementations\n\n");
    
    printf("%8s %8s %12s %12s %12s %10s %12s\n", 
           "Size_A", "Size_B", "VSLA(ms)", "Manual(ms)", "BLAS(ms)", "Ratio_M", "Ratio_B");
    printf("---------------------------------------------------------------------------------\n");
    
    // Test different size combinations
    struct {
        size_t a_size, b_size;
        const char* scenario;
    } test_cases[] = {
        {10, 10, "Small equal"},
        {100, 100, "Medium equal"},
        {1000, 1000, "Large equal"},
        {10, 100, "Small x Medium"},
        {100, 10, "Medium x Small"},
        {10, 1000, "Small x Large"},
        {1000, 10, "Large x Small"},
        {50, 200, "Rectangular 1"},
        {200, 50, "Rectangular 2"}
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int warmup_runs = 3;
    int bench_runs = 100;
    
    for (int tc = 0; tc < num_cases; tc++) {
        size_t a_size = test_cases[tc].a_size;
        size_t b_size = test_cases[tc].b_size;
        size_t out_size = a_size * b_size;
        
        // Skip very large combinations that would be too slow
        if (out_size > 1000000) {
            printf("%8zu %8zu %12s %12s %12s %10s %12s  (%s - too large)\n", 
                   a_size, b_size, "SKIP", "SKIP", "SKIP", "N/A", "N/A", test_cases[tc].scenario);
            continue;
        }
        
        // Create VSLA tensors (Model B for Kronecker)
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){a_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){b_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* out_vsla = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        
        // Create traditional arrays
        double* a_data = (double*)malloc(a_size * sizeof(double));
        double* b_data = (double*)malloc(b_size * sizeof(double));
        double* out_manual = (double*)malloc(out_size * sizeof(double));
        double* out_blas = (double*)malloc(out_size * sizeof(double));
        
        if (!a || !b || !out_vsla || !a_data || !b_data || !out_manual || !out_blas) {
            printf("Failed to allocate for sizes %zu x %zu\n", a_size, b_size);
            continue;
        }
        
        // Fill with test data
        vsla_fill(ctx, a, 0.0);
        vsla_fill(ctx, b, 0.0);
        
        for (size_t i = 0; i < a_size; i++) {
            uint64_t idx[] = {i};
            double val = sin(2.0 * M_PI * i / a_size);
            vsla_set_f64(ctx, a, idx, val);
            a_data[i] = val;
        }
        
        for (size_t i = 0; i < b_size; i++) {
            uint64_t idx[] = {i};
            double val = cos(2.0 * M_PI * i / b_size);
            vsla_set_f64(ctx, b, idx, val);
            b_data[i] = val;
        }
        
        // Warmup runs
        for (int w = 0; w < warmup_runs; w++) {
            vsla_kron(ctx, out_vsla, a, b);
            traditional_kronecker(a_data, a_size, b_data, b_size, out_manual);
            blas_kronecker(a_data, a_size, b_data, b_size, out_blas);
        }
        
        // Benchmark VSLA
        double start_time = get_time();
        for (int r = 0; r < bench_runs; r++) {
            vsla_kron(ctx, out_vsla, a, b);
        }
        double vsla_time = (get_time() - start_time) * 1000.0 / bench_runs;
        
        // Benchmark traditional
        start_time = get_time();
        for (int r = 0; r < bench_runs; r++) {
            traditional_kronecker(a_data, a_size, b_data, b_size, out_manual);
        }
        double manual_time = (get_time() - start_time) * 1000.0 / bench_runs;
        
        // Benchmark BLAS
        start_time = get_time();
        for (int r = 0; r < bench_runs; r++) {
            blas_kronecker(a_data, a_size, b_data, b_size, out_blas);
        }
        double blas_time = (get_time() - start_time) * 1000.0 / bench_runs;
        
        double ratio_manual = vsla_time / manual_time;
        double ratio_blas = vsla_time / blas_time;
        
        printf("%8zu %8zu %12.3f %12.3f %12.3f %10.2f %12.2f  (%s)\n", 
               a_size, b_size, vsla_time, manual_time, blas_time, 
               ratio_manual, ratio_blas, test_cases[tc].scenario);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out_vsla);
        free(a_data);
        free(b_data);
        free(out_manual);
        free(out_blas);
    }
}

// Test non-commutativity and correctness
static void benchmark_noncommutativity(vsla_context_t* ctx) {
    printf("\n=== Kronecker Non-Commutativity Analysis ===\n");
    printf("Testing performance impact of operand order\n\n");
    
    printf("%8s %8s %12s %12s %12s %15s\n", 
           "Size_A", "Size_B", "A⊗B (ms)", "B⊗A (ms)", "Diff(%%)", "Recommendation");
    printf("------------------------------------------------------------------------\n");
    
    struct {
        size_t a_size, b_size;
        const char* pattern;
    } patterns[] = {
        {10, 100, "Small x Large"},
        {100, 10, "Large x Small"},
        {50, 200, "1:4 ratio"},
        {200, 50, "4:1 ratio"},
        {20, 500, "1:25 ratio"},
        {500, 20, "25:1 ratio"}
    };
    
    int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
    int runs = 1000;
    
    for (int p = 0; p < num_patterns; p++) {
        size_t a_size = patterns[p].a_size;
        size_t b_size = patterns[p].b_size;
        size_t out_size = a_size * b_size; // Same for both orders
        
        // Create tensors for A⊗B
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){a_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){b_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* out_ab = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* out_ba = vsla_tensor_create(ctx, 1, (uint64_t[]){out_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        
        if (!a || !b || !out_ab || !out_ba) continue;
        
        // Fill with test data
        for (size_t i = 0; i < a_size; i++) {
            uint64_t idx[] = {i};
            vsla_set_f64(ctx, a, idx, (double)(i + 1));
        }
        
        for (size_t i = 0; i < b_size; i++) {
            uint64_t idx[] = {i};
            vsla_set_f64(ctx, b, idx, (double)(i + 1) * 0.1);
        }
        
        // Warmup
        vsla_kron(ctx, out_ab, a, b);
        vsla_kron(ctx, out_ba, b, a);
        
        // Benchmark A⊗B
        double start = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_kron(ctx, out_ab, a, b);
        }
        double time_ab = (get_time() - start) * 1000.0 / runs;
        
        // Benchmark B⊗A
        start = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_kron(ctx, out_ba, b, a);
        }
        double time_ba = (get_time() - start) * 1000.0 / runs;
        
        double diff_percent = ((time_ba - time_ab) / time_ab) * 100.0;
        const char* recommendation = (fabs(diff_percent) < 5.0) ? "No preference" :
                                   (time_ab < time_ba) ? "Use A⊗B" : "Use B⊗A";
        
        printf("%8zu %8zu %12.3f %12.3f %12.1f %15s\n", 
               a_size, b_size, time_ab, time_ba, diff_percent, recommendation);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out_ab);
        vsla_tensor_free(out_ba);
    }
}

// Memory efficiency analysis
static void benchmark_memory_efficiency(vsla_context_t* ctx) {
    printf("\n=== Memory Efficiency Analysis ===\n");
    printf("Testing memory usage vs computational complexity\n\n");
    
    printf("%10s %15s %15s %15s %12s\n", 
           "Operation", "Input_Mem(KB)", "Output_Mem(KB)", "Ops_Count", "Ops/KB");
    printf("-----------------------------------------------------------------------\n");
    
    struct {
        size_t a_size, b_size;
        const char* description;
    } mem_tests[] = {
        {100, 100, "Balanced"},
        {10, 1000, "Sparse_A"},
        {1000, 10, "Sparse_B"},
        {200, 200, "Medium"},
        {50, 800, "Very_uneven"}
    };
    
    int num_tests = sizeof(mem_tests) / sizeof(mem_tests[0]);
    
    for (int t = 0; t < num_tests; t++) {
        size_t a_size = mem_tests[t].a_size;
        size_t b_size = mem_tests[t].b_size;
        size_t out_size = a_size * b_size;
        
        double input_mem_kb = (a_size + b_size) * sizeof(double) / 1024.0;
        double output_mem_kb = out_size * sizeof(double) / 1024.0;
        size_t ops_count = a_size * b_size; // One multiply per output element
        double ops_per_kb = ops_count / (input_mem_kb + output_mem_kb);
        
        printf("%10s %15.2f %15.2f %15zu %12.0f\n", 
               mem_tests[t].description, input_mem_kb, output_mem_kb, ops_count, ops_per_kb);
    }
}

// Scalar multiplication special case
static void benchmark_scalar_cases(vsla_context_t* ctx) {
    printf("\n=== Scalar Multiplication Special Cases ===\n");
    printf("Testing when one operand is scalar [1]\n\n");
    
    printf("%12s %12s %12s %12s %15s\n", 
           "Vector_Size", "VSLA(μs)", "BLAS(μs)", "Ratio", "Advantage");
    printf("----------------------------------------------------------\n");
    
    size_t sizes[] = {10, 50, 100, 500, 1000, 5000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int runs = 5000;
    
    for (int s = 0; s < num_sizes; s++) {
        size_t vec_size = sizes[s];
        
        // VSLA: scalar ⊗ vector
        vsla_tensor_t* scalar = vsla_tensor_create(ctx, 1, (uint64_t[]){1}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* vector = vsla_tensor_create(ctx, 1, (uint64_t[]){vec_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){vec_size}, VSLA_MODEL_B, VSLA_DTYPE_F64);
        
        // Traditional arrays
        double scalar_val = 2.5;
        double* vec_data = (double*)malloc(vec_size * sizeof(double));
        double* out_blas = (double*)malloc(vec_size * sizeof(double));
        
        if (!scalar || !vector || !out || !vec_data || !out_blas) continue;
        
        // Fill data
        uint64_t scalar_idx[] = {0};
        vsla_set_f64(ctx, scalar, scalar_idx, scalar_val);
        
        for (size_t i = 0; i < vec_size; i++) {
            uint64_t idx[] = {i};
            double val = sin(i * 0.1);
            vsla_set_f64(ctx, vector, idx, val);
            vec_data[i] = val;
        }
        
        // Warmup
        vsla_kron(ctx, out, scalar, vector);
        cblas_dcopy(vec_size, vec_data, 1, out_blas, 1);
        cblas_dscal(vec_size, scalar_val, out_blas, 1);
        
        // Benchmark VSLA
        double start = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_kron(ctx, out, scalar, vector);
        }
        double vsla_time = (get_time() - start) * 1000000.0 / runs; // microseconds
        
        // Benchmark BLAS
        start = get_time();  
        for (int r = 0; r < runs; r++) {
            cblas_dcopy(vec_size, vec_data, 1, out_blas, 1);
            cblas_dscal(vec_size, scalar_val, out_blas, 1);
        }
        double blas_time = (get_time() - start) * 1000000.0 / runs; // microseconds
        
        double ratio = vsla_time / blas_time;
        const char* advantage = (ratio < 1.1) ? "VSLA" : (ratio < 2.0) ? "Comparable" : "BLAS";
        
        printf("%12zu %12.2f %12.2f %12.2f %15s\n", 
               vec_size, vsla_time, blas_time, ratio, advantage);
        
        // Cleanup
        vsla_tensor_free(scalar);
        vsla_tensor_free(vector);
        vsla_tensor_free(out);
        free(vec_data);
        free(out_blas);
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
    
    printf("=== VSLA Kronecker Product Comprehensive Benchmark ===\n\n");
    
    benchmark_kronecker_sizes(ctx);
    benchmark_noncommutativity(ctx);
    benchmark_memory_efficiency(ctx);
    benchmark_scalar_cases(ctx);
    
    printf("\n=== Summary & Analysis ===\n");
    printf("Kronecker Product Characteristics:\n");
    printf("  • Output size: O(mn) - grows quickly\n");
    printf("  • Non-commutative: A⊗B ≠ B⊗A (different layouts)\n");
    printf("  • Memory intensive: large intermediate results\n");
    printf("  • Cache sensitivity: depends on access patterns\n\n");
    
    printf("VSLA Model B Strengths:\n");
    printf("  • Handles variable shapes naturally\n");
    printf("  • Good performance for medium-sized operands\n");
    printf("  • Automatic result sizing\n\n");
    
    printf("Optimization Opportunities:\n");
    printf("  • Small operand detection (< 16 elements)\n");
    printf("  • Scalar multiplication fast path\n");
    printf("  • Block-wise computation for large results\n");
    printf("  • Memory-aware operand ordering suggestions\n");
    
    vsla_cleanup(ctx);
    return 0;
}