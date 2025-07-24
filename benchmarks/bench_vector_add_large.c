/**
 * @file bench_vector_add_large.c
 * @brief Large-scale VSLA Vector Addition Benchmark with microsecond precision
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

// Manual vector addition with padding
static void manual_padding_add(const double* a, size_t a_size, 
                             const double* b, size_t b_size,
                             double* out, size_t out_size) {
    for (size_t i = 0; i < out_size; i++) {
        double va = (i < a_size) ? a[i] : 0.0;
        double vb = (i < b_size) ? b[i] : 0.0;
        out[i] = va + vb;
    }
}

// Benchmark configuration
typedef struct {
    size_t a_size;
    size_t b_size;
    const char* scenario;
} test_case_t;

// Main benchmark function
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
    
    printf("=== VSLA Vector Addition Large-Scale Benchmark ===\n");
    printf("Testing with larger sizes and microsecond precision\n\n");
    
    // Test cases with larger sizes
    test_case_t test_cases[] = {
        // Small vectors
        {100, 100, "Small equal"},
        {1000, 1000, "Medium equal"},
        {10000, 10000, "Large equal"},
        {100000, 100000, "XLarge equal"},
        {1000000, 1000000, "Huge equal"},
        
        // Ambient promotion cases
        {100000, 10000, "100k vs 10k"},
        {10000, 100000, "10k vs 100k"},
        {100000, 1000, "100k vs 1k"},
        {1000, 100000, "1k vs 100k"},
        {100000, 100, "100k vs 100"},
        {100, 100000, "100 vs 100k"},
        
        // Extreme sparsity
        {1000000, 10, "1M vs 10"},
        {10, 1000000, "10 vs 1M"},
        {1000000, 1, "1M vs scalar"},
        {1, 1000000, "scalar vs 1M"}
    };
    
    printf("  Size_A    Size_B      VSLA(μs)   Manual(μs)    Ratio   Scenario\n");
    printf("------------------------------------------------------------------\n");
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
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
        
        if (!a || !b || !out_vsla || !a_data || !b_data || !out_manual) {
            printf("Failed to allocate for sizes %zu, %zu\n", a_size, b_size);
            continue;
        }
        
        // Fill with test data
        vsla_fill(ctx, a, 1.0);
        vsla_fill(ctx, b, 2.0);
        for (size_t i = 0; i < a_size; i++) a_data[i] = 1.0;
        for (size_t i = 0; i < b_size; i++) b_data[i] = 2.0;
        
        // Determine number of runs based on size (fewer runs for larger sizes)
        int runs = 1;
        if (max_size < 1000) runs = 10000;
        else if (max_size < 10000) runs = 1000;
        else if (max_size < 100000) runs = 100;
        else if (max_size < 1000000) runs = 10;
        
        // Warmup
        vsla_add(ctx, out_vsla, a, b);
        manual_padding_add(a_data, a_size, b_data, b_size, out_manual, max_size);
        
        // Benchmark VSLA
        double start_time = get_time();
        for (int r = 0; r < runs; r++) {
            vsla_add(ctx, out_vsla, a, b);
        }
        double vsla_time = (get_time() - start_time) * 1000000.0 / runs; // microseconds
        
        // Benchmark manual
        start_time = get_time();
        for (int r = 0; r < runs; r++) {
            manual_padding_add(a_data, a_size, b_data, b_size, out_manual, max_size);
        }
        double manual_time = (get_time() - start_time) * 1000000.0 / runs; // microseconds
        
        double ratio = vsla_time / manual_time;
        
        printf("%8zu %8zu %12.1f %12.1f %8.2f   %s\n", 
               a_size, b_size, vsla_time, manual_time, ratio, test_cases[tc].scenario);
        
        // Cleanup
        vsla_tensor_free(a);
        vsla_tensor_free(b);
        vsla_tensor_free(out_vsla);
        free(a_data);
        free(b_data);
        free(out_manual);
    }
    
    printf("\n=== Performance Summary ===\n");
    printf("Times in microseconds (μs), smaller is better\n");
    printf("Ratio > 1.0 means VSLA is slower than manual\n\n");
    
    printf("Key observations:\n");
    printf("- Small vectors show VSLA overhead from tensor management\n");
    printf("- Large equal-size vectors benefit from VSLA optimizations\n");
    printf("- Ambient promotion cases vary based on size differences\n");
    printf("- Extreme sparsity shows the cost of zero-extension logic\n");
    
    vsla_cleanup(ctx);
    return 0;
}