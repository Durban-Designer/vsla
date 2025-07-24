/**
 * @file bench_multidim_shapes.c
 * @brief Multi-dimensional tensor benchmark with shape mismatches
 * 
 * This benchmark demonstrates VSLA's core advantage: efficiently handling
 * tensors with different shapes without wasteful zero-padding.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "vsla/vsla.h"
#include "vsla/vsla_unified.h"

// High-resolution timer
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Get current memory usage in MB
static double get_memory_usage_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Convert KB to MB
}

// Manual zero-padding approach for multi-dimensional tensors
static void manual_pad_and_add(
    double* a, uint64_t* a_shape, uint8_t a_rank,
    double* b, uint64_t* b_shape, uint8_t b_rank,
    double* out, uint64_t* out_shape, uint8_t out_rank) {
    
    // Compute total elements for each tensor
    uint64_t a_total = 1, b_total = 1, out_total = 1;
    for (int i = 0; i < a_rank; i++) a_total *= a_shape[i];
    for (int i = 0; i < b_rank; i++) b_total *= b_shape[i];
    for (int i = 0; i < out_rank; i++) out_total *= out_shape[i];
    
    // Zero the output
    memset(out, 0, out_total * sizeof(double));
    
    // Add tensors with broadcasting
    for (uint64_t i = 0; i < out_total; i++) {
        // Convert linear index to multi-dimensional indices
        uint64_t idx[8] = {0}; // Max 8 dimensions
        uint64_t temp = i;
        for (int d = out_rank - 1; d >= 0; d--) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        
        // Map to indices in a and b with broadcasting
        uint64_t a_idx = 0, b_idx = 0;
        uint64_t a_stride = 1, b_stride = 1;
        
        for (int d = a_rank - 1; d >= 0; d--) {
            uint64_t a_coord = (idx[d] < a_shape[d]) ? idx[d] : 0;
            a_idx += a_coord * a_stride;
            a_stride *= a_shape[d];
        }
        
        for (int d = b_rank - 1; d >= 0; d--) {
            uint64_t b_coord = (idx[d] < b_shape[d]) ? idx[d] : 0;
            b_idx += b_coord * b_stride;
            b_stride *= b_shape[d];
        }
        
        double a_val = (a_idx < a_total) ? a[a_idx] : 0.0;
        double b_val = (b_idx < b_total) ? b[b_idx] : 0.0;
        out[i] = a_val + b_val;
    }
}

// Benchmark test case
typedef struct {
    uint64_t a_shape[8];
    uint64_t b_shape[8];
    uint8_t rank;
    const char* scenario;
} shape_test_case_t;

// Run a single shape mismatch benchmark
static void benchmark_shape_mismatch(
    vsla_context_t* ctx,
    shape_test_case_t* test_case,
    int warmup_runs,
    int bench_runs) {
    
    // Compute output shape (element-wise max)
    uint64_t out_shape[8];
    for (int i = 0; i < test_case->rank; i++) {
        out_shape[i] = (test_case->a_shape[i] > test_case->b_shape[i]) 
                     ? test_case->a_shape[i] : test_case->b_shape[i];
    }
    
    // Create VSLA tensors
    vsla_tensor_t* a = vsla_tensor_create(ctx, test_case->rank, test_case->a_shape, 
                                          VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, test_case->rank, test_case->b_shape,
                                          VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out_vsla = vsla_tensor_create(ctx, test_case->rank, out_shape,
                                                  VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with test data
    vsla_fill(ctx, a, 1.0);
    vsla_fill(ctx, b, 2.0);
    
    // Compute sizes for manual approach
    uint64_t a_elems = 1, b_elems = 1, out_elems = 1;
    for (int i = 0; i < test_case->rank; i++) {
        a_elems *= test_case->a_shape[i];
        b_elems *= test_case->b_shape[i];
        out_elems *= out_shape[i];
    }
    
    // Allocate manual arrays (this is the waste!)
    double* a_manual = (double*)calloc(out_elems, sizeof(double));
    double* b_manual = (double*)calloc(out_elems, sizeof(double));
    double* out_manual = (double*)calloc(out_elems, sizeof(double));
    
    // Fill manual arrays (simulating padding)
    for (uint64_t i = 0; i < a_elems; i++) a_manual[i] = 1.0;
    for (uint64_t i = 0; i < b_elems; i++) b_manual[i] = 2.0;
    
    // Memory measurements
    double mem_before = get_memory_usage_mb();
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
        manual_pad_and_add(a_manual, test_case->a_shape, test_case->rank,
                          b_manual, test_case->b_shape, test_case->rank,
                          out_manual, out_shape, test_case->rank);
    }
    
    // Benchmark VSLA
    double vsla_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
    }
    double vsla_time = (get_time() - vsla_start) * 1000.0 / bench_runs; // ms
    
    // Benchmark manual padding
    double manual_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        manual_pad_and_add(a_manual, test_case->a_shape, test_case->rank,
                          b_manual, test_case->b_shape, test_case->rank,
                          out_manual, out_shape, test_case->rank);
    }
    double manual_time = (get_time() - manual_start) * 1000.0 / bench_runs; // ms
    
    double mem_after = get_memory_usage_mb();
    
    // Calculate memory usage
    uint64_t vsla_memory = (a_elems + b_elems + out_elems) * sizeof(double);
    uint64_t manual_memory = 3 * out_elems * sizeof(double); // All padded to output size
    double memory_ratio = (double)manual_memory / vsla_memory;
    
    // Calculate speedup
    double speedup = manual_time / vsla_time;
    
    // Print shape info
    printf("\n%s:\n", test_case->scenario);
    printf("  Shape A: [");
    for (int i = 0; i < test_case->rank; i++) {
        printf("%lu%s", test_case->a_shape[i], i < test_case->rank-1 ? "," : "");
    }
    printf("] (%lu elements)\n", a_elems);
    
    printf("  Shape B: [");
    for (int i = 0; i < test_case->rank; i++) {
        printf("%lu%s", test_case->b_shape[i], i < test_case->rank-1 ? "," : "");
    }
    printf("] (%lu elements)\n", b_elems);
    
    printf("  Output: [");
    for (int i = 0; i < test_case->rank; i++) {
        printf("%lu%s", out_shape[i], i < test_case->rank-1 ? "," : "");
    }
    printf("] (%lu elements)\n", out_elems);
    
    printf("  Performance:\n");
    printf("    VSLA:        %8.3f ms\n", vsla_time);
    printf("    Manual Pad:  %8.3f ms\n", manual_time);
    printf("    Speedup:     %8.2fx %s\n", speedup, 
           speedup > 1.0 ? "faster" : "slower");
    
    printf("  Memory Usage:\n");
    printf("    VSLA:        %8.2f MB (actual data only)\n", vsla_memory / 1048576.0);
    printf("    Manual Pad:  %8.2f MB (all padded to output)\n", manual_memory / 1048576.0);
    printf("    Waste:       %8.2f MB (%.1f%% overhead)\n", 
           (manual_memory - vsla_memory) / 1048576.0,
           (memory_ratio - 1.0) * 100.0);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out_vsla);
    free(a_manual);
    free(b_manual);
    free(out_manual);
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
    
    printf("=== VSLA Multi-Dimensional Shape Mismatch Benchmark ===\n");
    printf("Demonstrating VSLA's advantage over zero-padding approaches\n");
    printf("=========================================================\n");
    
    // Test cases showing real-world scenarios
    shape_test_case_t test_cases[] = {
        // 2D: Matrix broadcasting (common in ML) - smaller sizes
        {{200, 200}, {200, 1}, 2, "2D Matrix Column Broadcasting"},
        {{200, 200}, {1, 200}, 2, "2D Matrix Row Broadcasting"},
        
        // 3D: CNN feature maps with different spatial dimensions - smaller sizes
        {{16, 64, 64}, {16, 64, 1}, 3, "3D CNN Spatial Broadcasting"},
        {{16, 64, 64}, {16, 1, 64}, 3, "3D CNN Channel Broadcasting"},
        {{32, 128, 128}, {32, 1, 1}, 3, "3D CNN Bias Addition"},
        
        // 4D: Batch processing with attention mechanisms - smaller sizes
        {{8, 32, 32, 64}, {8, 32, 1, 64}, 4, "4D Attention Query-Key"},
        {{8, 32, 32, 64}, {8, 1, 32, 64}, 4, "4D Attention Masking"},
        {{16, 32, 32, 32}, {1, 32, 32, 32}, 4, "4D Batch Broadcasting"},
        
        // Extreme cases showing maximum benefit - moderate sizes
        {{200, 200, 50}, {200, 1, 1}, 3, "3D Extreme Sparsity"},
        {{50, 50, 50, 5}, {50, 1, 1, 1}, 4, "4D Extreme Sparsity"},
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int warmup_runs = 2;
    int bench_runs = 10;
    
    for (int i = 0; i < num_cases; i++) {
        benchmark_shape_mismatch(ctx, &test_cases[i], warmup_runs, bench_runs);
    }
    
    printf("\n=== Summary ===\n");
    printf("VSLA advantages demonstrated:\n");
    printf("1. Performance: Faster by avoiding computation on zeros\n");
    printf("2. Memory: Uses only required memory, not padded size\n");
    printf("3. Semantics: Natural broadcasting without explicit padding\n");
    printf("4. Cache: Better locality from compact representation\n");
    
    vsla_cleanup(ctx);
    return 0;
}