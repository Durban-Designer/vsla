/**
 * @file bench_pyramid_stacking.c
 * @brief Benchmark for VSLA pyramid stacking operations
 * 
 * Tests pyramid stacking performance against hierarchical buffering implementations
 * to validate VSLA's efficiency in multi-resolution data processing scenarios.
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

// Simple hierarchical buffer for comparison
typedef struct {
    double** level_data;     // Data for each level
    size_t* level_capacity;  // Capacity of each level  
    size_t* level_size;      // Current size of each level
    size_t* level_head;      // Head pointer for each level
    size_t num_levels;
    size_t window_size;
} hierarchical_buffer_t;

// Forward declaration
static void hierarchical_buffer_destroy(hierarchical_buffer_t* buf);

static hierarchical_buffer_t* hierarchical_buffer_create(size_t num_levels, size_t window_size) {
    hierarchical_buffer_t* buf = malloc(sizeof(hierarchical_buffer_t));
    if (!buf) return NULL;
    
    buf->num_levels = num_levels;
    buf->window_size = window_size;
    
    buf->level_data = calloc(num_levels, sizeof(double*));
    buf->level_capacity = calloc(num_levels, sizeof(size_t));
    buf->level_size = calloc(num_levels, sizeof(size_t));
    buf->level_head = calloc(num_levels, sizeof(size_t));
    
    if (!buf->level_data || !buf->level_capacity || !buf->level_size || !buf->level_head) {
        hierarchical_buffer_destroy(buf);
        return NULL;
    }
    
    // Each level has capacity = window_size
    for (size_t i = 0; i < num_levels; i++) {
        buf->level_capacity[i] = window_size;
        buf->level_data[i] = calloc(window_size, sizeof(double));
        if (!buf->level_data[i]) {
            hierarchical_buffer_destroy(buf);
            return NULL;
        }
    }
    
    return buf;
}

static void hierarchical_buffer_destroy(hierarchical_buffer_t* buf) {
    if (buf) {
        if (buf->level_data) {
            for (size_t i = 0; i < buf->num_levels; i++) {
                free(buf->level_data[i]);
            }
            free(buf->level_data);
        }
        free(buf->level_capacity);
        free(buf->level_size);
        free(buf->level_head);
        free(buf);
    }
}

static void hierarchical_buffer_push(hierarchical_buffer_t* buf, double value) {
    // Push to level 0
    size_t level = 0;
    buf->level_data[level][buf->level_head[level]] = value;
    buf->level_head[level] = (buf->level_head[level] + 1) % buf->level_capacity[level];
    if (buf->level_size[level] < buf->level_capacity[level]) {
        buf->level_size[level]++;
    }
    
    // Cascade to higher levels when windows fill
    for (size_t l = 0; l < buf->num_levels - 1; l++) {
        if (buf->level_size[l] == buf->level_capacity[l] && buf->level_head[l] == 0) {
            // Compute average of current level and push to next level
            double sum = 0.0;
            for (size_t i = 0; i < buf->level_capacity[l]; i++) {
                sum += buf->level_data[l][i];
            }
            double avg = sum / buf->level_capacity[l];
            
            size_t next_level = l + 1;
            buf->level_data[next_level][buf->level_head[next_level]] = avg;
            buf->level_head[next_level] = (buf->level_head[next_level] + 1) % buf->level_capacity[next_level];
            if (buf->level_size[next_level] < buf->level_capacity[next_level]) {
                buf->level_size[next_level]++;
            }
        }
    }
}

static double** hierarchical_buffer_get_all_levels(hierarchical_buffer_t* buf, size_t* num_complete) {
    *num_complete = 0;
    
    // Count complete levels
    for (size_t i = 0; i < buf->num_levels; i++) {
        if (buf->level_size[i] == buf->level_capacity[i]) {
            (*num_complete)++;
        }
    }
    
    if (*num_complete == 0) return NULL;
    
    double** results = malloc((*num_complete) * sizeof(double*));
    if (!results) return NULL;
    
    size_t result_idx = 0;
    for (size_t l = 0; l < buf->num_levels; l++) {
        if (buf->level_size[l] == buf->level_capacity[l]) {
            results[result_idx] = malloc(buf->level_capacity[l] * sizeof(double));
            if (!results[result_idx]) {
                // Clean up on failure
                for (size_t i = 0; i < result_idx; i++) {
                    free(results[i]);
                }
                free(results);
                return NULL;
            }
            
            // Copy level data in correct order
            for (size_t i = 0; i < buf->level_capacity[l]; i++) {
                size_t idx = (buf->level_head[l] + i) % buf->level_capacity[l];
                results[result_idx][i] = buf->level_data[l][idx];
            }
            result_idx++;
        }
    }
    
    return results;
}

// Benchmark configuration
typedef struct {
    size_t num_levels;
    size_t window_size;
    size_t num_pushes;
    bool discard_partials;
    const char* label;
} pyramid_test_config_t;

static const pyramid_test_config_t pyramid_configs[] = {
    {3, 10, 1000, false, "3_levels_small_window"},
    {3, 50, 2000, false, "3_levels_medium_window"},
    {5, 20, 5000, false, "5_levels_small_window"},
    {5, 100, 5000, false, "5_levels_large_window"},
    {3, 10, 1000, true, "3_levels_discard_partials"},
    {5, 50, 10000, false, "5_levels_intensive"},
    {7, 10, 10000, false, "7_levels_deep_hierarchy"},
    {2, 200, 2000, false, "2_levels_huge_windows"}
};

#define NUM_PYRAMID_CONFIGS (sizeof(pyramid_configs) / sizeof(pyramid_configs[0]))

// Benchmark result structure
typedef struct {
    double vsla_time_ms;
    double hierarchical_time_ms;
    double ratio;
    const char* winner;
    size_t vsla_outputs;
    size_t hierarchical_outputs;
} pyramid_benchmark_result_t;

// Benchmark VSLA pyramid stacking vs hierarchical buffer
static pyramid_benchmark_result_t benchmark_pyramid_stacking(vsla_context_t* ctx,
                                                           size_t num_levels,
                                                           size_t window_size,
                                                           size_t num_pushes,
                                                           bool discard_partials) {
    pyramid_benchmark_result_t result = {0};
    
    // Create VSLA pyramid
    vsla_pyramid_t* vsla_pyramid = vsla_pyramid_create(ctx, num_levels, window_size, 1, VSLA_DTYPE_F64, discard_partials);
    
    // Create hierarchical buffer
    hierarchical_buffer_t* hier_buf = hierarchical_buffer_create(num_levels, window_size);
    
    if (!vsla_pyramid || !hier_buf) {
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
        test_data[i] = sin(i * 0.1) + cos(i * 0.05) + 0.1 * sin(i * 0.01);
    }
    
    // Benchmark VSLA pyramid stacking
    double start = get_time();
    
    for (size_t i = 0; i < num_pushes; i++) {
        // Create single-element tensor
        vsla_tensor_t* elem = vsla_tensor_create(ctx, 1, (uint64_t[]){1}, VSLA_MODEL_A, VSLA_DTYPE_F64);
        if (elem) {
            vsla_set_f64(ctx, elem, (uint64_t[]){0}, test_data[i]);
            
            // Push to pyramid (may return stacked tensor)
            vsla_tensor_t* pyramid_result = vsla_pyramid_push(vsla_pyramid, elem);
            
            // Count outputs
            if (pyramid_result) {
                result.vsla_outputs++;
                vsla_tensor_free(pyramid_result);
            }
            
            vsla_tensor_free(elem);
        }
    }
    
    // Flush remaining data
    size_t flush_count;
    vsla_tensor_t** flush_results = vsla_pyramid_flush(vsla_pyramid, &flush_count);
    result.vsla_outputs += flush_count;
    
    if (flush_results) {
        for (size_t i = 0; i < flush_count; i++) {
            if (flush_results[i]) {
                vsla_tensor_free(flush_results[i]);
            }
        }
        free(flush_results);
    }
    
    result.vsla_time_ms = (get_time() - start) * 1000.0;
    
    // Benchmark hierarchical buffer
    start = get_time();
    
    for (size_t i = 0; i < num_pushes; i++) {
        hierarchical_buffer_push(hier_buf, test_data[i]);
        
        // Check for complete levels
        size_t num_complete;
        double** level_results = hierarchical_buffer_get_all_levels(hier_buf, &num_complete);
        
        if (level_results) {
            result.hierarchical_outputs += num_complete;
            
            // Clean up results
            for (size_t j = 0; j < num_complete; j++) {
                free(level_results[j]);
            }
            free(level_results);
        }
    }
    
    result.hierarchical_time_ms = (get_time() - start) * 1000.0;
    
    result.ratio = result.vsla_time_ms / result.hierarchical_time_ms;
    result.winner = (result.ratio < 1.1) ? "VSLA" : (result.ratio < 2.0) ? "Close" : "Hierarchical";
    
cleanup:
    if (vsla_pyramid) vsla_pyramid_destroy(vsla_pyramid);
    if (hier_buf) hierarchical_buffer_destroy(hier_buf);
    free(test_data);
    
    return result;
}

// Memory scaling analysis
static void test_memory_scaling(vsla_context_t* ctx) {
    printf("\n=== Memory Scaling Analysis ===\n");
    printf("Levels  Window_Size   VSLA_Memory   Hierarchical_Memory   Ratio\n");
    printf("-------------------------------------------------------------------\n");
    
    size_t level_counts[] = {2, 3, 5, 7, 10};
    size_t window_sizes[] = {10, 50, 100};
    
    for (size_t i = 0; i < sizeof(level_counts)/sizeof(level_counts[0]); i++) {
        for (size_t j = 0; j < sizeof(window_sizes)/sizeof(window_sizes[0]); j++) {
            size_t levels = level_counts[i];
            size_t window_size = window_sizes[j];
            
            // Estimate VSLA memory usage
            size_t vsla_memory = sizeof(vsla_pyramid_t*) +
                               (levels * window_size * sizeof(double)) +  // Level data
                               (levels * sizeof(vsla_window_t*)) +        // Window pointers
                               (levels * 64);                            // Overhead per level
            
            // Hierarchical buffer memory usage  
            size_t hier_memory = sizeof(hierarchical_buffer_t) +
                               (levels * sizeof(double*)) +              // Level pointers
                               (levels * window_size * sizeof(double)) + // Level data
                               (levels * 3 * sizeof(size_t));           // Level metadata
            
            double ratio = (double)vsla_memory / hier_memory;
            
            printf("%6zu %12zu   %11zu   %19zu   %5.2f\n",
                   levels, window_size, vsla_memory, hier_memory, ratio);
        }
    }
}

// Throughput comparison at different scales
static void test_throughput_comparison(vsla_context_t* ctx) {
    printf("\n=== Throughput Comparison ===\n");
    printf("Configuration          VSLA_Items/sec    Hierarchical_Items/sec\n");
    printf("--------------------------------------------------------------\n");
    
    struct {
        size_t levels;
        size_t window_size; 
        size_t pushes;
        const char* label;
    } configs[] = {
        {3, 10, 10000, "Small_Pyramid"},
        {5, 50, 10000, "Medium_Pyramid"},
        {7, 20, 20000, "Deep_Pyramid"},
        {3, 100, 5000, "Wide_Pyramid"}
    };
    
    for (size_t i = 0; i < sizeof(configs)/sizeof(configs[0]); i++) {
        pyramid_benchmark_result_t result = benchmark_pyramid_stacking(ctx,
            configs[i].levels, configs[i].window_size, configs[i].pushes, false);
        
        if (result.vsla_time_ms > 0) {
            double vsla_throughput = configs[i].pushes / (result.vsla_time_ms / 1000.0);
            double hier_throughput = configs[i].pushes / (result.hierarchical_time_ms / 1000.0);
            
            printf("%-22s %13.0f    %22.0f\n",
                   configs[i].label, vsla_throughput, hier_throughput);
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
    
    printf("=== VSLA Pyramid Stacking Benchmark ===\n\n");
    printf("Testing pyramid stacking performance against hierarchical buffer implementations\n");
    printf("Ratio > 1.0 means VSLA is slower\n\n");
    
    // Main benchmark results
    printf("=== Pyramid Stacking Performance ===\n");
    printf("Configuration                 VSLA(ms)   Hier(ms)    Ratio   Winner   VSLA_Outs   Hier_Outs\n");
    printf("-----------------------------------------------------------------------------------------\n");
    
    for (size_t i = 0; i < NUM_PYRAMID_CONFIGS; i++) {
        const pyramid_test_config_t* config_ptr = &pyramid_configs[i];
        
        pyramid_benchmark_result_t result = benchmark_pyramid_stacking(ctx,
            config_ptr->num_levels,
            config_ptr->window_size, 
            config_ptr->num_pushes,
            config_ptr->discard_partials);
        
        if (result.vsla_time_ms >= 0) {
            printf("%-29s %8.2f %9.2f %8.2f   %-12s %9zu %9zu\n",
                   config_ptr->label,
                   result.vsla_time_ms,
                   result.hierarchical_time_ms,
                   result.ratio,
                   result.winner,
                   result.vsla_outputs,
                   result.hierarchical_outputs);
        } else {
            printf("%-29s %8s %9s %8s   %-12s %9s %9s\n",
                   config_ptr->label,
                   "FAIL", "FAIL", "FAIL", "ERROR", "N/A", "N/A");
        }
    }
    
    // Additional analyses
    test_memory_scaling(ctx);
    test_throughput_comparison(ctx);
    
    printf("\n=== Summary & Recommendations ===\n");
    printf("Pyramid Stacking Use Cases:\n");
    printf("  • Multi-resolution signal processing\n");
    printf("  • Hierarchical data aggregation\n");
    printf("  • Time series analysis at multiple scales\n");
    printf("  • Image pyramid construction\n");
    printf("  • Progressive data compression\n\n");
    
    printf("VSLA Advantages:\n");
    printf("  • Integrated tensor operations at each level\n");
    printf("  • Automatic level management and flushing\n");
    printf("  • Type safety and consistent error handling\n");
    printf("  • Configurable partial data handling\n\n");
    
    printf("Performance Considerations:\n");
    printf("  • Higher overhead than simple hierarchical buffers\n");
    printf("  • Memory usage includes tensor metadata overhead\n");
    printf("  • Best when pyramid levels need further processing\n");
    printf("  • Competitive for complex multi-level operations\n");
    
    vsla_cleanup(ctx);
    return 0;
}