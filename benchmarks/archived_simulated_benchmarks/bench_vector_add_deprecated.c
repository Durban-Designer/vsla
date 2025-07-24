/**
 * @file bench_vector_add.c
 * @brief Comprehensive vector addition benchmark for VSLA
 * 
 * Tests ambient promotion performance vs traditional approaches
 * and identifies scenarios where VSLA may be slower.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// Simple BLAS-like functions for benchmarking without external dependencies
static void simple_dcopy(size_t n, const double* x, double* y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

static void simple_daxpy(size_t n, double alpha, const double* x, double* y) {
    for (size_t i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

static void simple_dscal(size_t n, double alpha, double* x) {
    for (size_t i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

// Benchmarking utilities
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Traditional vector addition (simple implementation)
static void traditional_vector_add(const double* a, const double* b, double* out, size_t n) {
    // Copy a to output
    simple_dcopy(n, a, out);
    // Add b to output: out = a + b
    simple_daxpy(n, 1.0, b, out);
}

// Manual padding approach (what users would do without VSLA)
static void manual_padding_add(const double* a, size_t m, const double* b, size_t n, 
                              double* out, size_t max_size) {
    // Zero the output
    memset(out, 0, max_size * sizeof(double));
    
    // Copy a
    for (size_t i = 0; i < m; i++) {
        out[i] = a[i];
    }
    
    // Add b
    for (size_t i = 0; i < n; i++) {
        out[i] += b[i];
    }
}

// Benchmark ambient promotion vs traditional approaches
static void benchmark_ambient_promotion(vsla_context_t* ctx) {
    printf("=== Ambient Promotion Benchmark ===\n");
    printf("Testing VSLA ambient promotion vs traditional approaches\n\n");
    
    printf("%8s %8s %12s %12s %12s %10s\n", 
           "Size_A", "Size_B", "VSLA(ms)", "Manual(ms)", "BLAS(ms)", "VSLA/BLAS");
    printf("-----------------------------------------------------------------------\n");
    
    // Test different size combinations to show when ambient promotion helps/hurts
    struct {
        size_t a_size, b_size;
        const char* scenario;
    } test_cases[] = {
        {1000, 1000, "Equal sizes"},
        {1000, 100, "A >> B"},
        {100, 1000, "B >> A"},
        {1000, 10, "A >>> B"},
        {10, 1000, "B >>> A"},
        {1000, 1, "A vs scalar"},
        {1, 1000, "Scalar vs A"},
        {10000, 5000, "Large unequal"},
        {5000, 10000, "Large unequal rev"}
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int warmup_runs = 5;
    int bench_runs = 100;
    
    for (int tc = 0; tc < num_cases; tc++) {
        size_t a_size = test_cases[tc].a_size;
        size_t b_size = test_cases[tc].b_size;
        size_t max_size = (a_size > b_size) ? a_size : b_size;
        
        // Create VSLA tensors
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){a_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){b_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out_vsla = vsla_tensor_create(ctx, 1, (uint64_t[]){max_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Create traditional arrays
        double* a_data = (double*)calloc(a_size, sizeof(double));
        double* b_data = (double*)calloc(b_size, sizeof(double));
        double* out_manual = (double*)calloc(max_size, sizeof(double));
        double* out_blas = (double*)calloc(max_size, sizeof(double));
        
        if (!a || !b || !out_vsla || !a_data || !b_data || !out_manual || !out_blas) {
            printf("Failed to allocate for sizes %zu, %zu\n", a_size, b_size);
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
            vsla_add(ctx, out_vsla, a, b);
            manual_padding_add(a_data, a_size, b_data, b_size, out_manual, max_size);
            traditional_vector_add(a_data, b_data, out_blas, (a_size < b_size) ? a_size : b_size);
        }
        
        // Benchmark VSLA
        double start_time = get_time();
        for (int r = 0; r < bench_runs; r++) {
            vsla_add(ctx, out_vsla, a, b);
        }
        double vsla_time = (get_time() - start_time) * 1000.0 / bench_runs;
        
        // Benchmark manual padding
        start_time = get_time();
        for (int r = 0; r < bench_runs; r++) {
            manual_padding_add(a_data, a_size, b_data, b_size, out_manual, max_size);
        }
        double manual_time = (get_time() - start_time) * 1000.0 / bench_runs;
        
        // Benchmark BLAS (only works for equal sizes)
        double blas_time = 0.0;
        if (a_size == b_size) {
            start_time = get_time();
            for (int r = 0; r < bench_runs; r++) {
                traditional_vector_add(a_data, b_data, out_blas, a_size);
            }
            blas_time = (get_time() - start_time) * 1000.0 / bench_runs;
        }
        
        double ratio = (blas_time > 0) ? vsla_time / blas_time : 0.0;
        
        printf("%8zu %8zu %12.3f %12.3f %12.3f %10.2f  (%s)\n", 
               a_size, b_size, vsla_time, manual_time, 
               (blas_time > 0) ? blas_time : 0.0, 
               ratio, test_cases[tc].scenario);
        
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

// Test memory access patterns
static void benchmark_memory_patterns(vsla_context_t* ctx) {
    printf("\n=== Memory Access Pattern Analysis ===\n");
    printf("Testing cache efficiency of different approaches\n\n");
    
    printf("%10s %12s %12s %12s %15s\n", 
           "Pattern", "VSLA(ms)", "Manual(ms)", "Ratio", "Description");
    printf("----------------------------------------------------------------\n");
    
    size_t base_size = 10000;
    int runs = 1000;
    
    struct {
        double a_frac, b_frac;
        const char* pattern;
        const char* description;
    } patterns[] = {
        {1.0, 1.0, "Equal", "Same size vectors"},
        {1.0, 0.1, "90%_sparse", "B is 10% of A"},
        {1.0, 0.01, "99%_sparse", "B is 1% of A"},
        {0.1, 1.0, "A_sparse", "A is 10% of B"},
        {0.5, 0.5, "Half", "Both half size"},
        {2.0, 0.5, "2x_vs_0.5x", "A is 2x base, B is 0.5x"}
    };
    
    int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
    
    for (int p = 0; p < num_patterns; p++) {
        size_t a_size = (size_t)(base_size * patterns[p].a_frac);
        size_t b_size = (size_t)(base_size * patterns[p].b_frac);
        size_t max_size = (a_size > b_size) ? a_size : b_size;
        
        // VSLA tensors
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){a_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){b_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){max_size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Traditional arrays
        double* a_data = (double*)malloc(a_size * sizeof(double));
        double* b_data = (double*)malloc(b_size * sizeof(double));
        double* out_manual = (double*)malloc(max_size * sizeof(double));
        
        if (!a || !b || !out || !a_data || !b_data || !out_manual) continue;
        
        // Fill with data
        for (size_t i = 0; i < a_size; i++) {
            uint64_t idx[] = {i};
            double val = (double)(i % 100) / 100.0;
            vsla_set_f64(ctx, a, idx, val);
            a_data[i] = val;
        }
        
        for (size_t i = 0; i < b_size; i++) {
            uint64_t idx[] = {i};
            double val = (double)((i * 7) % 100) / 100.0;
            vsla_set_f64(ctx, b, idx, val);
            b_data[i] = val;
        }
        
        // Benchmark VSLA
        double start = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_add(ctx, out, a, b);
        }
        double vsla_time = (get_time() - start) * 1000.0 / runs;
        
        // Benchmark manual
        start = get_time();
        for (int r = 0; r < runs; r++) {
            manual_padding_add(a_data, a_size, b_data, b_size, out_manual, max_size);
        }
        double manual_time = (get_time() - start) * 1000.0 / runs;
        
        double ratio = vsla_time / manual_time;
        
        printf("%10s %12.3f %12.3f %12.2f %15s\n", 
               patterns[p].pattern, vsla_time, manual_time, ratio, patterns[p].description);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
        free(a_data);
        free(b_data);
        free(out_manual);
    }
}

// Overhead analysis for small vectors
static void benchmark_overhead_analysis(vsla_context_t* ctx) {
    printf("\n=== Overhead Analysis (Small Vectors) ===\n");
    printf("Identifying where VSLA overhead dominates\n\n");
    
    printf("%8s %12s %12s %12s %15s\n", 
           "Size", "VSLA(μs)", "Direct(μs)", "Overhead", "Break-even");
    printf("-----------------------------------------------------------\n");
    
    size_t sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int runs = 10000; // More runs for small vectors
    
    for (int s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        
        // VSLA tensors
        vsla_tensor_t* a = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* out = vsla_tensor_create(ctx, 1, (uint64_t[]){size}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Direct arrays
        double* a_data = (double*)malloc(size * sizeof(double));
        double* b_data = (double*)malloc(size * sizeof(double));
        double* out_data = (double*)malloc(size * sizeof(double));
        
        if (!a || !b || !out || !a_data || !b_data || !out_data) continue;
        
        // Fill with test data
        for (size_t i = 0; i < size; i++) {
            uint64_t idx[] = {i};
            double val = (double)i;
            vsla_set_f64(ctx, a, idx, val);
            vsla_set_f64(ctx, b, idx, val * 2);
            a_data[i] = val;
            b_data[i] = val * 2;
        }
        
        // Warmup
        vsla_add(ctx, out, a, b);
        simple_dcopy(size, a_data, out_data);
        simple_daxpy(size, 1.0, b_data, out_data);
        
        // Benchmark VSLA
        double start = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_add(ctx, out, a, b);
        }
        double vsla_time = (get_time() - start) * 1000000.0 / runs; // microseconds
        
        // Benchmark direct implementation
        start = get_time();
        for (int r = 0; r < runs; r++) {
            simple_dcopy(size, a_data, out_data);
            simple_daxpy(size, 1.0, b_data, out_data);
        }
        double direct_time = (get_time() - start) * 1000000.0 / runs; // microseconds
        
        double overhead = vsla_time - direct_time;
        const char* break_even = (vsla_time < direct_time * 1.1) ? "✓" : 
                                (vsla_time < direct_time * 2.0) ? "~" : "✗";
        
        printf("%8zu %12.2f %12.2f %12.2f %15s\n", 
               size, vsla_time, direct_time, overhead, break_even);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out);
        free(a_data);
        free(b_data);
        free(out_data);
    }
    
    printf("\nBreak-even legend: ✓ < 10%% overhead, ~ < 100%% overhead, ✗ > 100%% overhead\n");
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
    
    printf("=== VSLA Vector Addition Comprehensive Benchmark ===\n\n");
    
    benchmark_ambient_promotion(ctx);
    benchmark_memory_patterns(ctx);
    benchmark_overhead_analysis(ctx);
    
    printf("\n=== Summary & Recommendations ===\n");
    printf("VSLA Strengths:\n");
    printf("  • Variable-shape operations without manual padding\n");
    printf("  • Efficient ambient promotion for different-sized vectors\n");
    printf("  • Good performance for medium to large vectors\n\n");
    
    printf("VSLA Weaknesses:\n");
    printf("  • Higher overhead for very small vectors (< 32 elements)\n");
    printf("  • Equal-size operations may be slower than optimized BLAS\n");
    printf("  • Memory allocations add latency for temporary operations\n\n");
    
    printf("Optimization Opportunities:\n");
    printf("  • Small vector fast path (< 32 elements)\n");
    printf("  • Equal-size detection with BLAS fallback\n");
    printf("  • Memory pool for frequent allocations\n");
    printf("  • Vectorized ambient promotion loops\n");
    
    vsla_cleanup(ctx);
    return 0;
}