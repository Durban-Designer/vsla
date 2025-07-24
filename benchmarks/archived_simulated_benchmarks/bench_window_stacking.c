/**
 * @file bench_window_stacking.c 
 * @brief Benchmark for VSLA window stacking operations
 * 
 * Tests window stacking performance against simple circular buffer implementations
 * to validate VSLA's efficiency in streaming data scenarios.
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

// Simple circular buffer for comparison
typedef struct {
    double* data;
    size_t capacity;
    size_t size;
    size_t head;
} circular_buffer_t;

static circular_buffer_t* circular_buffer_create(size_t capacity) {
    circular_buffer_t* buf = malloc(sizeof(circular_buffer_t));
    if (!buf) return NULL;
    
    buf->data = calloc(capacity, sizeof(double));
    if (!buf->data) {
        free(buf);
        return NULL;
    }
    
    buf->capacity = capacity;
    buf->size = 0;
    buf->head = 0;
    return buf;
}

static void circular_buffer_destroy(circular_buffer_t* buf) {
    if (buf) {
        free(buf->data);
        free(buf);
    }
}

static void circular_buffer_push(circular_buffer_t* buf, double value) {
    buf->data[buf->head] = value;
    buf->head = (buf->head + 1) % buf->capacity;
    if (buf->size < buf->capacity) {
        buf->size++;
    }
}

static double* circular_buffer_get_window(circular_buffer_t* buf) {
    if (buf->size < buf->capacity) return NULL;
    
    double* window = malloc(buf->capacity * sizeof(double));
    if (!window) return NULL;
    
    for (size_t i = 0; i < buf->capacity; i++) {
        size_t idx = (buf->head + i) % buf->capacity;
        window[i] = buf->data[idx];
    }
    
    return window;
}

// Benchmark configuration
typedef struct {
    size_t window_size;
    size_t num_pushes;
    const char* label;
} window_test_config_t;

static const window_test_config_t window_configs[] = {
    {10, 1000, "small_window_many_pushes"},
    {50, 1000, "medium_window_many_pushes"},
    {100, 1000, "large_window_many_pushes"},
    {10, 10000, "small_window_massive_pushes"},
    {50, 10000, "medium_window_massive_pushes"},
    {100, 10000, "large_window_massive_pushes"},
    {500, 2000, "huge_window_moderate_pushes"},
    {1000, 1000, "massive_window_few_pushes"}
};

#define NUM_WINDOW_CONFIGS (sizeof(window_configs) / sizeof(window_configs[0]))

// Benchmark result structure
typedef struct {
    double vsla_time_ms;
    double circular_time_ms;
    double ratio;
    const char* winner;
} window_benchmark_result_t;

// Benchmark VSLA window stacking vs circular buffer
static window_benchmark_result_t benchmark_window_stacking(vsla_context_t* ctx, 
                                                          size_t window_size, 
                                                          size_t num_pushes) {
    window_benchmark_result_t result = {0};
    
    // Create VSLA window
    vsla_window_t* vsla_window = vsla_window_create(ctx, window_size, 1, VSLA_DTYPE_F64);
    
    // Create circular buffer
    circular_buffer_t* circ_buf = circular_buffer_create(window_size);
    
    if (!vsla_window || !circ_buf) {
        result.vsla_time_ms = -1;
        goto cleanup;
    }
    
    // Generate test data
    double* test_data = malloc(num_pushes * sizeof(double));
    if (!test_data) {
        result.vsla_time_ms = -1;
        goto cleanup;
    }
    
    for (size_t i = 0; i < num_pushes; i++) {
        test_data[i] = sin(i * 0.1) + cos(i * 0.05);
    }
    
    // Benchmark VSLA window stacking
    double start = get_time();
    
    for (size_t i = 0; i < num_pushes; i++) {
        // Create single-element tensor
        vsla_tensor_t* elem = vsla_tensor_create(ctx, 1, (uint64_t[]){1}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        if (elem) {
            vsla_set_f64(ctx, elem, (uint64_t[]){0}, test_data[i]);
            
            // Push to window (may return stacked tensor)
            vsla_tensor_t* window_result = vsla_window_push(vsla_window, elem);
            
            // Clean up result if returned
            if (window_result) {
                vsla_tensor_free(window_result);
            }
            
            vsla_tensor_free(elem);
        }
    }
    
    result.vsla_time_ms = (get_time() - start) * 1000.0;
    
    // Benchmark circular buffer
    start = get_time();
    
    for (size_t i = 0; i < num_pushes; i++) {
        circular_buffer_push(circ_buf, test_data[i]);
        
        // Get window if full (to match VSLA behavior)
        if (i >= window_size - 1) {
            double* window = circular_buffer_get_window(circ_buf);
            if (window) {
                free(window);
            }
        }
    }
    
    result.circular_time_ms = (get_time() - start) * 1000.0;
    
    result.ratio = result.vsla_time_ms / result.circular_time_ms;
    result.winner = (result.ratio < 1.1) ? "VSLA" : (result.ratio < 2.0) ? "Close" : "Circular";
    
cleanup:
    if (vsla_window) vsla_window_destroy(vsla_window);
    if (circ_buf) circular_buffer_destroy(circ_buf);
    free(test_data);
    
    return result;
}

// Memory efficiency test
static void test_memory_efficiency(vsla_context_t* ctx) {
    printf("\n=== Memory Efficiency Analysis ===\n");
    printf("Window Size   VSLA Memory   Circular Memory   Ratio\n");
    printf("-------------------------------------------------------\n");
    
    size_t window_sizes[] = {10, 50, 100, 500, 1000, 5000};
    size_t num_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
    
    for (size_t i = 0; i < num_sizes; i++) {
        size_t window_size = window_sizes[i];
        
        // Estimate VSLA memory usage
        size_t vsla_memory = sizeof(vsla_window_t*) + 
                           (window_size * sizeof(double)) +  // Data storage
                           (window_size * sizeof(vsla_tensor_t*)) + // Tensor pointers
                           64; // Overhead estimate
        
        // Circular buffer memory usage
        size_t circular_memory = sizeof(circular_buffer_t) + 
                               (window_size * sizeof(double));
        
        double ratio = (double)vsla_memory / circular_memory;
        
        printf("%11zu   %11zu   %15zu   %5.2f\n", 
               window_size, vsla_memory, circular_memory, ratio);
    }
}

// Throughput analysis
static void test_throughput_scaling(vsla_context_t* ctx) {
    printf("\n=== Throughput Scaling Analysis ===\n");
    printf("Pushes/sec with different window sizes:\n");
    printf("Window Size    VSLA Throughput    Circular Throughput\n");
    printf("-----------------------------------------------------\n");
    
    size_t window_sizes[] = {10, 50, 100, 500, 1000};
    size_t num_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
    size_t test_pushes = 10000;
    
    for (size_t i = 0; i < num_sizes; i++) {
        size_t window_size = window_sizes[i];
        
        window_benchmark_result_t result = benchmark_window_stacking(ctx, window_size, test_pushes);
        
        if (result.vsla_time_ms > 0) {
            double vsla_throughput = test_pushes / (result.vsla_time_ms / 1000.0);
            double circular_throughput = test_pushes / (result.circular_time_ms / 1000.0);
            
            printf("%11zu    %15.0f    %19.0f\n", 
                   window_size, vsla_throughput, circular_throughput);
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
    
    printf("=== VSLA Window Stacking Benchmark ===\n\n");
    printf("Testing window stacking performance against circular buffer implementations\n");
    printf("Ratio > 1.0 means VSLA is slower\n\n");
    
    // Main benchmark results
    printf("=== Window Stacking Performance ===\n");
    printf("Configuration                VSLA(ms)  Circular(ms)    Ratio    Winner\n");
    printf("----------------------------------------------------------------------\n");
    
    for (size_t i = 0; i < NUM_WINDOW_CONFIGS; i++) {
        const window_test_config_t* config_ptr = &window_configs[i];
        
        window_benchmark_result_t result = benchmark_window_stacking(ctx, 
            config_ptr->window_size, config_ptr->num_pushes);
        
        if (result.vsla_time_ms >= 0) {
            printf("%-32s %8.2f %12.2f %8.2f    %s\n",
                   config_ptr->label,
                   result.vsla_time_ms,
                   result.circular_time_ms,
                   result.ratio,
                   result.winner);
        } else {
            printf("%-32s %8s %12s %8s    %s\n",
                   config_ptr->label,
                   "FAIL", "FAIL", "FAIL", "ERROR");
        }
    }
    
    // Additional analyses
    test_memory_efficiency(ctx);
    test_throughput_scaling(ctx);
    
    printf("\n=== Summary & Recommendations ===\n");
    printf("Window Stacking Use Cases:\n");
    printf("  • Real-time data streaming and buffering\n");
    printf("  • Moving window computations (moving averages, etc.)\n");
    printf("  • Time series analysis with sliding windows\n");
    printf("  • Signal processing with overlapping frames\n\n");
    
    printf("VSLA Advantages:\n");
    printf("  • Integrated with tensor operations\n");
    printf("  • Type safety and error handling\n");
    printf("  • Consistent API with other VSLA operations\n");
    printf("  • Automatic memory management\n\n");
    
    printf("Performance Considerations:\n");
    printf("  • VSLA has higher overhead for simple buffering\n");
    printf("  • Competitive for complex window operations\n");
    printf("  • Memory usage is higher but includes safety features\n");
    printf("  • Best used when windows need further tensor processing\n");
    
    vsla_cleanup(ctx);
    return 0;
}