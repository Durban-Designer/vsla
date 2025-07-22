/**
 * @file vsla_benchmark_suite.c
 * @brief Comprehensive VSLA benchmark suite with configurable backends and parameters
 * 
 * Usage:
 *   ./vsla_benchmark_suite [options]
 *   
 * Options:
 *   -b, --backend <cpu|cuda|auto>     Backend to test (default: auto)
 *   -s, --size <small|medium|large>   Tensor size preset (default: medium)
 *   -i, --iterations <N>              Number of iterations (default: 1000)
 *   -o, --operation <all|add|conv>    Operation to benchmark (default: all)
 *   -v, --verbose                     Verbose output
 *   -h, --help                        Show help
 *   
 * Example:
 *   ./vsla_benchmark_suite --backend cuda --size large --iterations 5000
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include <math.h>

// Benchmark configuration
typedef struct {
    vsla_backend_t backend;
    int size_preset;           // 0=small, 1=medium, 2=large
    int iterations;
    char operation[32];
    int verbose;
} benchmark_config_t;

// Test size configurations
typedef struct {
    const char* name;
    uint64_t dim1, dim2, dim3;
    double memory_mb;
} size_config_t;

static const size_config_t sizes[] = {
    {"small",  100,   50,   1,   0.04},    // ~40KB
    {"medium", 1000,  500,  1,   4.0},     // ~4MB  
    {"large",  10000, 1000, 1,   80.0},    // ~80MB
};

// Timing utilities
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Performance results
typedef struct {
    const char* operation;
    double time_ms;
    double ops_per_sec;
    double memory_bandwidth_gbps;
    vsla_error_t status;
} benchmark_result_t;

// Print help
static void print_help(const char* program_name) {
    printf("VSLA Benchmark Suite v1.0\n\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  -b, --backend <cpu|cuda|auto>     Backend to test (default: auto)\n");  
    printf("  -s, --size <small|medium|large>   Tensor size preset (default: medium)\n");
    printf("  -i, --iterations <N>              Number of iterations (default: 1000)\n");
    printf("  -o, --operation <all|add|sub|scale|hadamard|conv|kron|sum|norm|fill>\n");
    printf("                                     Operation to benchmark (default: all)\n");
    printf("  -v, --verbose                     Verbose output\n");
    printf("  -h, --help                        Show this help\n\n");
    printf("Size Presets:\n");
    for (int i = 0; i < 3; i++) {
        printf("  %-8s: %lux%lu (%.1f MB)\n", 
               sizes[i].name, sizes[i].dim1, sizes[i].dim2, sizes[i].memory_mb);
    }
    printf("\nExample:\n");
    printf("  %s --backend cuda --size large --iterations 5000\n", program_name);
}

// Parse command line arguments
static int parse_args(int argc, char* argv[], benchmark_config_t* config) {
    // Set defaults
    config->backend = VSLA_BACKEND_AUTO;
    config->size_preset = 1; // medium
    config->iterations = 1000;
    strcpy(config->operation, "all");
    config->verbose = 0;
    
    static struct option long_options[] = {
        {"backend",     required_argument, 0, 'b'},
        {"size",        required_argument, 0, 's'},
        {"iterations",  required_argument, 0, 'i'},
        {"operation",   required_argument, 0, 'o'},
        {"verbose",     no_argument,       0, 'v'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    while ((c = getopt_long(argc, argv, "b:s:i:o:vh", long_options, NULL)) != -1) {
        switch (c) {
            case 'b':
                if (strcmp(optarg, "cpu") == 0) config->backend = VSLA_BACKEND_CPU;
                else if (strcmp(optarg, "cuda") == 0) config->backend = VSLA_BACKEND_CUDA;  
                else if (strcmp(optarg, "auto") == 0) config->backend = VSLA_BACKEND_AUTO;
                else {
                    fprintf(stderr, "Invalid backend: %s\n", optarg);
                    return -1;
                }
                break;
            case 's':
                if (strcmp(optarg, "small") == 0) config->size_preset = 0;
                else if (strcmp(optarg, "medium") == 0) config->size_preset = 1;
                else if (strcmp(optarg, "large") == 0) config->size_preset = 2;
                else {
                    fprintf(stderr, "Invalid size: %s\n", optarg);
                    return -1;
                }
                break;
            case 'i':
                config->iterations = atoi(optarg);
                if (config->iterations <= 0) {
                    fprintf(stderr, "Invalid iterations: %s\n", optarg);
                    return -1;
                }
                break;
            case 'o':
                strncpy(config->operation, optarg, sizeof(config->operation) - 1);
                break;
            case 'v':
                config->verbose = 1;
                break;
            case 'h':
                print_help(argv[0]);
                exit(0);
            case '?':
                return -1;
            default:
                return -1;
        }
    }
    
    return 0;
}

// Create test tensors
static vsla_tensor_t* create_test_tensor(vsla_context_t* ctx, int size_preset, double fill_value) {
    const size_config_t* size = &sizes[size_preset];
    uint64_t shape[] = {size->dim1, size->dim2};
    uint8_t rank = (size->dim2 > 1) ? 2 : 1;
    
    vsla_tensor_t* tensor = vsla_tensor_create(ctx, rank, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (tensor) {
        vsla_fill(ctx, tensor, fill_value);
    }
    return tensor;
}

// Benchmark individual operation
static benchmark_result_t benchmark_operation(vsla_context_t* ctx, 
                                             const char* op_name,
                                             int size_preset,
                                             int iterations,
                                             int verbose) {
    benchmark_result_t result = {0};
    result.operation = op_name;
    result.status = VSLA_ERROR_NOT_IMPLEMENTED;
    
    // Create test tensors with appropriate sizes for different operations
    vsla_tensor_t* a = NULL;
    vsla_tensor_t* b = NULL;
    vsla_tensor_t* out = NULL;
    
    if (strcmp(op_name, "conv") == 0) {
        // For convolution: out_size = a_size + b_size - 1
        // Use smaller tensors to get reasonable output sizes
        uint64_t a_size = (size_preset == 0) ? 5 : (size_preset == 1) ? 10 : 20;
        uint64_t b_size = (size_preset == 0) ? 3 : (size_preset == 1) ? 5 : 10;  
        uint64_t out_size = a_size + b_size - 1;
        
        uint64_t shape_a[] = {a_size};
        uint64_t shape_b[] = {b_size};
        uint64_t shape_out[] = {out_size};
        
        if (verbose) {
            printf("      Conv tensor sizes: a=%lu, b=%lu, out=%lu\n", a_size, b_size, out_size);
        }
        
        a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F64);
        b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F64);
        out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (a) vsla_fill(ctx, a, 1.0);
        if (b) vsla_fill(ctx, b, 2.0);
        if (out) vsla_fill(ctx, out, 0.0);
    } else if (strcmp(op_name, "kron") == 0) {
        // For Kronecker: out_size = a_size * b_size
        // Use much smaller tensors to avoid huge outputs
        const size_config_t* size = &sizes[size_preset];
        uint64_t a_size = (size_preset == 0) ? 8 : (size_preset == 1) ? 16 : 32;
        uint64_t b_size = (size_preset == 0) ? 4 : (size_preset == 1) ? 8 : 16;
        uint64_t out_size = a_size * b_size;
        
        uint64_t shape_a[] = {a_size};
        uint64_t shape_b[] = {b_size};
        uint64_t shape_out[] = {out_size};
        
        a = vsla_tensor_create(ctx, 1, shape_a, VSLA_MODEL_B, VSLA_DTYPE_F64);
        b = vsla_tensor_create(ctx, 1, shape_b, VSLA_MODEL_B, VSLA_DTYPE_F64);
        out = vsla_tensor_create(ctx, 1, shape_out, VSLA_MODEL_B, VSLA_DTYPE_F64);
        
        if (a) vsla_fill(ctx, a, 2.0);
        if (b) vsla_fill(ctx, b, 3.0);
        if (out) vsla_fill(ctx, out, 0.0);
    } else {
        // For other operations, use standard tensor sizes
        a = create_test_tensor(ctx, size_preset, 1.0);
        b = create_test_tensor(ctx, size_preset, 2.0);
        out = create_test_tensor(ctx, size_preset, 0.0);
    }
    
    if (!a || !b || !out) {
        result.status = VSLA_ERROR_MEMORY;
        goto cleanup;
    }
    
    if (verbose) {
        printf("    Running %s (%d iterations)... ", op_name, iterations);
        fflush(stdout);
    }
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        if (strcmp(op_name, "add") == 0) {
            vsla_add(ctx, out, a, b);
        } else if (strcmp(op_name, "fill") == 0) {
            vsla_fill(ctx, out, 42.0);
        }
        // Add more operations as needed
    }
    
    // Synchronize before timing
    vsla_synchronize(ctx);
    
    // Benchmark
    double start_time = get_time_ms();
    
    for (int i = 0; i < iterations; i++) {
        if (strcmp(op_name, "add") == 0) {
            result.status = vsla_add(ctx, out, a, b);
        } else if (strcmp(op_name, "sub") == 0) {
            result.status = vsla_sub(ctx, out, a, b);
        } else if (strcmp(op_name, "scale") == 0) {
            result.status = vsla_scale(ctx, out, a, 2.5);
        } else if (strcmp(op_name, "hadamard") == 0) {
            result.status = vsla_hadamard(ctx, out, a, b);
        } else if (strcmp(op_name, "fill") == 0) {
            result.status = vsla_fill(ctx, out, 42.0);
        } else if (strcmp(op_name, "sum") == 0) {
            double sum_val;
            result.status = vsla_sum(ctx, a, &sum_val);
        } else if (strcmp(op_name, "norm") == 0) {
            double norm_val;
            result.status = vsla_norm(ctx, a, &norm_val);
        } else if (strcmp(op_name, "conv") == 0) {
            result.status = vsla_conv(ctx, out, a, b);
        } else if (strcmp(op_name, "kron") == 0) {
            result.status = vsla_kron(ctx, out, a, b);
        } else {
            result.status = VSLA_ERROR_NOT_IMPLEMENTED;
            break;
        }
        
        if (result.status != VSLA_SUCCESS) break;
    }
    
    // Synchronize after timing
    vsla_synchronize(ctx);
    double end_time = get_time_ms();
    
    // Calculate performance metrics
    result.time_ms = (end_time - start_time) / iterations;
    result.ops_per_sec = 1000.0 / result.time_ms;
    
    // Estimate memory bandwidth (simplified)
    const size_config_t* size = &sizes[size_preset];
    double bytes_per_op = size->memory_mb * 1024.0 * 1024.0 * 3.0; // Rough estimate: read A, B, write C
    result.memory_bandwidth_gbps = (bytes_per_op * result.ops_per_sec) / (1024.0 * 1024.0 * 1024.0);
    
    if (verbose) {
        if (result.status == VSLA_SUCCESS) {
            printf("%.3f ms/op (%.1f ops/sec)\n", result.time_ms, result.ops_per_sec);
        } else {
            printf("FAILED (%s)\n", vsla_error_string(result.status));
        }
    }
    
cleanup:
    if (a) vsla_tensor_free(a);
    if (b) vsla_tensor_free(b); 
    if (out) vsla_tensor_free(out);
    
    return result;
}

// Print results table
static void print_results_table(benchmark_result_t* results, int count, 
                               vsla_backend_t backend, int size_preset,
                               int iterations) {
    printf("\n🎯 VSLA BENCHMARK RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    // Print configuration
    const char* backend_name = (backend == VSLA_BACKEND_CPU) ? "CPU" : 
                              (backend == VSLA_BACKEND_CUDA) ? "CUDA" : "AUTO";
    printf("Backend: %s | Size: %s (%.1f MB) | Iterations: %d\n", 
           backend_name, sizes[size_preset].name, sizes[size_preset].memory_mb, iterations);
    printf("───────────────────────────────────────────────────────────────\n");
    
    // Table header
    printf("%-12s %12s %12s %12s %s\n", 
           "Operation", "Time (ms)", "Ops/sec", "BW (GB/s)", "Status");
    printf("───────────────────────────────────────────────────────────────\n");
    
    // Table rows
    double total_time = 0.0;
    int success_count = 0;
    
    for (int i = 0; i < count; i++) {
        const char* status_str = (results[i].status == VSLA_SUCCESS) ? "✅ OK" : "❌ FAIL";
        
        printf("%-12s %12.3f %12.1f %12.3f %s\n",
               results[i].operation,
               results[i].time_ms,
               results[i].ops_per_sec, 
               results[i].memory_bandwidth_gbps,
               status_str);
        
        if (results[i].status == VSLA_SUCCESS) {
            total_time += results[i].time_ms;
            success_count++;
        }
    }
    
    printf("───────────────────────────────────────────────────────────────\n");
    printf("Summary: %d/%d operations successful, avg time: %.3f ms/op\n", 
           success_count, count, success_count > 0 ? total_time / success_count : 0.0);
    printf("═══════════════════════════════════════════════════════════════\n");
}

// Main benchmark function
int main(int argc, char* argv[]) {
    benchmark_config_t config;
    
    // Parse arguments
    if (parse_args(argc, argv, &config) != 0) {
        print_help(argv[0]);
        return 1;
    }
    
    printf("🚀 VSLA Benchmark Suite Starting...\n");
    
    // Initialize VSLA context with specified backend
    vsla_config_t vsla_config = {
        .backend = config.backend,
        .device_id = -1,
        .memory_limit = 0,
        .optimization_hint = VSLA_HINT_NONE,
        .enable_profiling = false,
        .verbose = config.verbose
    };
    
    vsla_context_t* ctx = vsla_init(&vsla_config);
    if (!ctx) {
        fprintf(stderr, "❌ Failed to initialize VSLA context\n");
        return 1;
    }
    
    // Get actual backend info
    vsla_backend_t actual_backend;
    char device_name[64] = "Unknown";
    double memory_gb = 0.0;
    vsla_get_runtime_info(ctx, &actual_backend, device_name, &memory_gb);
    
    if (config.verbose) {
        printf("✅ VSLA initialized: %s (%.1f GB memory)\n", device_name, memory_gb);
        printf("   Size: %s (%lux%lu, %.1f MB)\n", 
               sizes[config.size_preset].name,
               sizes[config.size_preset].dim1,
               sizes[config.size_preset].dim2, 
               sizes[config.size_preset].memory_mb);
        printf("   Iterations: %d\n\n", config.iterations);
    }
    
    // Define operations to benchmark
    const char* operations[] = {"add", "sub", "scale", "hadamard", "fill", "sum", "norm", "conv", "kron"};
    int num_ops = sizeof(operations) / sizeof(operations[0]);
    
    // Filter operations if specific one requested
    const char** ops_to_run = (const char**)operations;
    int ops_count = num_ops;
    
    if (strcmp(config.operation, "all") != 0) {
        ops_to_run = (const char**)&config.operation;
        ops_count = 1;
    }
    
    // Run benchmarks
    benchmark_result_t* results = malloc(ops_count * sizeof(benchmark_result_t));
    if (!results) {
        fprintf(stderr, "❌ Memory allocation failed\n");
        vsla_cleanup(ctx);
        return 1;
    }
    
    for (int i = 0; i < ops_count; i++) {
        const char* op = (ops_count == 1) ? config.operation : operations[i];
        results[i] = benchmark_operation(ctx, op, config.size_preset, 
                                       config.iterations, config.verbose);
    }
    
    // Print results
    print_results_table(results, ops_count, actual_backend, 
                       config.size_preset, config.iterations);
    
    // Cleanup
    free(results);
    vsla_cleanup(ctx);
    
    return 0;
}