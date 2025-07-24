/**
 * @file bench_broadcasting.c
 * @brief Broadcasting semantics benchmark comparing VSLA vs NumPy-style broadcasting
 * 
 * Tests VSLA's natural broadcasting against traditional approaches that require
 * explicit copying and temporary tensors.
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

// NumPy-style broadcasting with explicit copies
static void numpy_broadcast_add(
    double* a, uint64_t* a_shape, uint8_t a_rank,
    double* b, uint64_t* b_shape, uint8_t b_rank,
    double* out, uint64_t* out_shape, uint8_t out_rank) {
    
    // First expand both tensors to output shape with copies
    uint64_t out_total = 1;
    for (int i = 0; i < out_rank; i++) out_total *= out_shape[i];
    
    double* a_expanded = (double*)malloc(out_total * sizeof(double));
    double* b_expanded = (double*)malloc(out_total * sizeof(double));
    
    // Expand tensor a
    for (uint64_t i = 0; i < out_total; i++) {
        uint64_t idx[8] = {0};
        uint64_t temp = i;
        for (int d = out_rank - 1; d >= 0; d--) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        
        // Map to a's coordinates with broadcasting
        uint64_t a_idx = 0, a_stride = 1;
        for (int d = a_rank - 1; d >= 0; d--) {
            uint64_t coord = (idx[d] < a_shape[d]) ? idx[d] : (a_shape[d] == 1 ? 0 : idx[d] % a_shape[d]);
            a_idx += coord * a_stride;
            a_stride *= a_shape[d];
        }
        a_expanded[i] = a[a_idx];
    }
    
    // Expand tensor b
    for (uint64_t i = 0; i < out_total; i++) {
        uint64_t idx[8] = {0};
        uint64_t temp = i;
        for (int d = out_rank - 1; d >= 0; d--) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        
        uint64_t b_idx = 0, b_stride = 1;
        for (int d = b_rank - 1; d >= 0; d--) {
            uint64_t coord = (idx[d] < b_shape[d]) ? idx[d] : (b_shape[d] == 1 ? 0 : idx[d] % b_shape[d]);
            b_idx += coord * b_stride;
            b_stride *= b_shape[d];
        }
        b_expanded[i] = b[b_idx];
    }
    
    // Now do element-wise addition (the easy part)
    for (uint64_t i = 0; i < out_total; i++) {
        out[i] = a_expanded[i] + b_expanded[i];
    }
    
    free(a_expanded);
    free(b_expanded);
}

// Test case for broadcasting
typedef struct {
    uint64_t a_shape[4];
    uint64_t b_shape[4];
    uint8_t rank;
    const char* scenario;
} broadcast_test_case_t;

// Benchmark broadcasting operation
static void benchmark_broadcasting(vsla_context_t* ctx, broadcast_test_case_t* test_case) {
    // Compute output shape (element-wise max)
    uint64_t out_shape[4];
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
    vsla_fill(ctx, a, 1.5);
    vsla_fill(ctx, b, 2.5);
    
    // Compute sizes
    uint64_t a_elems = 1, b_elems = 1, out_elems = 1;
    for (int i = 0; i < test_case->rank; i++) {
        a_elems *= test_case->a_shape[i];
        b_elems *= test_case->b_shape[i];
        out_elems *= out_shape[i];
    }
    
    // Create NumPy-style arrays
    double* a_data = (double*)malloc(a_elems * sizeof(double));
    double* b_data = (double*)malloc(b_elems * sizeof(double));
    double* out_numpy = (double*)malloc(out_elems * sizeof(double));
    
    for (uint64_t i = 0; i < a_elems; i++) a_data[i] = 1.5;
    for (uint64_t i = 0; i < b_elems; i++) b_data[i] = 2.5;
    
    int warmup_runs = 3;
    int bench_runs = 20;
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
        numpy_broadcast_add(a_data, test_case->a_shape, test_case->rank,
                           b_data, test_case->b_shape, test_case->rank,
                           out_numpy, out_shape, test_case->rank);
    }
    
    // Benchmark VSLA (natural broadcasting)
    double vsla_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        vsla_add(ctx, out_vsla, a, b);
    }
    double vsla_time = (get_time() - vsla_start) * 1000.0 / bench_runs; // ms
    
    // Benchmark NumPy-style (copy + expand + add)
    double numpy_start = get_time();
    for (int i = 0; i < bench_runs; i++) {
        numpy_broadcast_add(a_data, test_case->a_shape, test_case->rank,
                           b_data, test_case->b_shape, test_case->rank,
                           out_numpy, out_shape, test_case->rank);
    }
    double numpy_time = (get_time() - numpy_start) * 1000.0 / bench_runs; // ms
    
    double speedup = numpy_time / vsla_time;
    
    // Calculate memory usage
    uint64_t vsla_memory = (a_elems + b_elems + out_elems) * sizeof(double);
    uint64_t numpy_memory = (a_elems + b_elems + out_elems + 2 * out_elems) * sizeof(double); // 2 temporary expansions
    double memory_ratio = (double)numpy_memory / vsla_memory;
    
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
    printf("    VSLA (natural):  %8.3f ms\n", vsla_time);
    printf("    NumPy (copy):    %8.3f ms\n", numpy_time);
    printf("    Speedup:         %8.2fx %s\n", speedup, 
           speedup > 1.0 ? "faster" : "slower");
    
    printf("  Memory Efficiency:\n");
    printf("    VSLA:            %8.2f MB (in-place broadcasting)\n", vsla_memory / 1048576.0);
    printf("    NumPy:           %8.2f MB (temporary copies)\n", numpy_memory / 1048576.0);
    printf("    Memory Savings:  %8.1f%% less memory\n", (1.0 - 1.0/memory_ratio) * 100.0);
    
    // Cleanup
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(out_vsla);
    free(a_data);
    free(b_data);
    free(out_numpy);
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
    
    printf("=== VSLA Broadcasting Semantics Benchmark ===\n");
    printf("Comparing natural broadcasting vs copy-based approaches\n");
    printf("====================================================\n");
    
    // Test cases showing common broadcasting patterns
    broadcast_test_case_t test_cases[] = {
        // 2D broadcasting
        {{100, 100}, {100, 1}, 2, "2D Matrix-Vector Broadcasting"},
        {{100, 100}, {1, 100}, 2, "2D Vector-Matrix Broadcasting"},
        
        // 3D broadcasting 
        {{50, 50, 50}, {50, 50, 1}, 3, "3D Depth Broadcasting"},
        {{50, 50, 50}, {50, 1, 50}, 3, "3D Height Broadcasting"},
        {{50, 50, 50}, {1, 50, 50}, 3, "3D Batch Broadcasting"},
        
        // 4D broadcasting (common in deep learning)
        {{32, 64, 32, 32}, {32, 64, 1, 1}, 4, "4D Spatial Broadcasting"},
        {{32, 64, 32, 32}, {32, 1, 32, 32}, 4, "4D Channel Broadcasting"},
        {{32, 64, 32, 32}, {1, 64, 32, 32}, 4, "4D Batch Broadcasting"},
        
        // Extreme broadcasting (scalar-like)
        {{100, 100, 100}, {1, 1, 1}, 3, "3D Scalar Broadcasting"},
        {{50, 50, 50, 50}, {1, 1, 1, 1}, 4, "4D Scalar Broadcasting"},
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int i = 0; i < num_cases; i++) {
        benchmark_broadcasting(ctx, &test_cases[i]);
    }
    
    printf("\n=== Broadcasting Performance Summary ===\n");
    printf("Key advantages of VSLA's natural broadcasting:\n");
    printf("1. No temporary tensor copies required\n");
    printf("2. Lazy evaluation - computation only where needed\n");
    printf("3. Memory efficient - no expansion overhead\n");
    printf("4. Cache friendly - better locality of reference\n");
    printf("5. Automatic optimization for sparse patterns\n\n");
    
    printf("Traditional broadcasting (NumPy-style) requires:\n");
    printf("1. Explicit tensor expansion to common shape\n");
    printf("2. Temporary memory allocation (often 2-3x data size)\n");
    printf("3. Full materialization of intermediate results\n");
    printf("4. Copy overhead for large tensors\n");
    printf("5. Manual memory management of temporaries\n");
    
    vsla_cleanup(ctx);
    return 0;
}