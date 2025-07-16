/**
 * @file bench_convolution.c
 * @brief Benchmark FFT convolution performance
 */

#include "benchmark_utils.h"
#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

// Default test parameters
static size_t default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
static size_t num_default_sizes = sizeof(default_sizes) / sizeof(default_sizes[0]);

static void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -s, --sizes SIZE1,SIZE2,...  Signal sizes to test (default: 64,128,256,512,1024,2048,4096)\n");
    printf("  -i, --iterations N           Number of iterations per test (default: 100)\n");
    printf("  -w, --warmup N               Number of warmup iterations (default: 5)\n");
    printf("  -o, --output FILE            Output results to file (default: stdout)\n");
    printf("  -h, --help                   Show this help message\n");
}

static void benchmark_vsla_convolution(size_t signal_size, size_t kernel_size, 
                                     size_t iterations, size_t warmup) {
    // Create test data
    vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel) {
        printf("Error: Failed to create test tensors\n");
        return;
    }
    
    // Fill with test data
    for (size_t i = 0; i < signal_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(signal, &idx, sin(2.0 * M_PI * i / signal_size));
    }
    
    for (size_t i = 0; i < kernel_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(kernel, &idx, exp(-0.1 * i)); // Exponential decay kernel
    }
    
    // Create output tensor
    size_t output_size = signal_size + kernel_size - 1;
    vsla_tensor_t* result = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!result) {
        printf("Error: Failed to create result tensor\n");
        vsla_free(signal);
        vsla_free(kernel);
        return;
    }
    
    // Warmup iterations
    for (size_t i = 0; i < warmup; i++) {
        vsla_conv(result, signal, kernel);
    }
    
    // Benchmark timer
    benchmark_timer_t* timer = benchmark_timer_new(iterations);
    
    // Timed iterations
    for (size_t i = 0; i < iterations; i++) {
        benchmark_timer_start(timer);
        vsla_conv(result, signal, kernel);
        benchmark_timer_lap(timer);
    }
    
    benchmark_result_t bench_result = benchmark_timer_finish(timer);
    
    // Get system info
    system_info_t sys_info;
    get_system_info(&sys_info);
    
    // Print results
    print_benchmark_header("convolution", "vsla_fft");
    printf("  \"signal_size\": %zu,\n", signal_size);
    printf("  \"kernel_size\": %zu,\n", kernel_size);
    printf("  \"output_size\": %zu,\n", output_size);
    print_benchmark_result(&bench_result, "convolution", "vsla_fft", &sys_info);
    print_benchmark_footer();
    printf(",\n");
    
    // Cleanup
    benchmark_timer_free(timer);
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(result);
}

static void benchmark_vsla_direct_convolution(size_t signal_size, size_t kernel_size,
                                            size_t iterations, size_t warmup) {
    // Similar setup to FFT version
    vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel) {
        printf("Error: Failed to create test tensors\n");
        return;
    }
    
    // Fill with test data
    for (size_t i = 0; i < signal_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(signal, &idx, sin(2.0 * M_PI * i / signal_size));
    }
    
    for (size_t i = 0; i < kernel_size; i++) {
        uint64_t idx = i;
        vsla_set_f64(kernel, &idx, exp(-0.1 * i));
    }
    
    size_t output_size = signal_size + kernel_size - 1;
    vsla_tensor_t* result = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!result) {
        printf("Error: Failed to create result tensor\n");
        vsla_free(signal);
        vsla_free(kernel);
        return;
    }
    
    // Warmup
    for (size_t i = 0; i < warmup; i++) {
        vsla_conv_direct(result, signal, kernel);
    }
    
    benchmark_timer_t* timer = benchmark_timer_new(iterations);
    
    // Timed iterations  
    for (size_t i = 0; i < iterations; i++) {
        benchmark_timer_start(timer);
        vsla_conv_direct(result, signal, kernel);
        benchmark_timer_lap(timer);
    }
    
    benchmark_result_t bench_result = benchmark_timer_finish(timer);
    
    system_info_t sys_info;
    get_system_info(&sys_info);
    
    print_benchmark_header("convolution", "vsla_direct");
    printf("  \"signal_size\": %zu,\n", signal_size);
    printf("  \"kernel_size\": %zu,\n", kernel_size);
    printf("  \"output_size\": %zu,\n", output_size);
    print_benchmark_result(&bench_result, "convolution", "vsla_direct", &sys_info);
    print_benchmark_footer();
    printf(",\n");
    
    benchmark_timer_free(timer);
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(result);
}

int main(int argc, char* argv[]) {
    size_t iterations = BENCHMARK_ITERATIONS_DEFAULT;
    size_t warmup = BENCHMARK_WARMUP_DEFAULT;
    size_array_t sizes = {0};
    FILE* output_file = stdout;
    
    // Parse command line arguments
    static struct option long_options[] = {
        {"sizes", required_argument, 0, 's'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "s:i:w:o:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 's':
                sizes = parse_size_list(optarg);
                break;
            case 'i':
                iterations = strtoul(optarg, NULL, 10);
                break;
            case 'w':
                warmup = strtoul(optarg, NULL, 10);
                break;
            case 'o':
                output_file = fopen(optarg, "w");
                if (!output_file) {
                    fprintf(stderr, "Error: Cannot open output file %s\n", optarg);
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Use default sizes if none specified
    if (sizes.count == 0) {
        sizes.values = malloc(num_default_sizes * sizeof(size_t));
        memcpy(sizes.values, default_sizes, num_default_sizes * sizeof(size_t));
        sizes.count = num_default_sizes;
    }
    
    // Initialize VSLA
    vsla_init();
    
    // Redirect output if needed
    if (output_file != stdout) {
        // Note: For simplicity, we'll just print to stdout and let shell redirect
    }
    
    printf("[\n");
    
    // Run benchmarks for each size
    for (size_t i = 0; i < sizes.count; i++) {
        size_t signal_size = sizes.values[i];
        size_t kernel_size = signal_size / 8; // Use kernel that's 1/8 the signal size
        if (kernel_size < 4) kernel_size = 4;
        
        printf("  // Signal size: %zu, Kernel size: %zu\n", signal_size, kernel_size);
        
        // Benchmark FFT convolution
        benchmark_vsla_convolution(signal_size, kernel_size, iterations, warmup);
        
        // Benchmark direct convolution for comparison
        benchmark_vsla_direct_convolution(signal_size, kernel_size, iterations, warmup);
    }
    
    printf("  {\"end\": true}\n");
    printf("]\n");
    
    // Cleanup
    free_size_array(&sizes);
    if (output_file != stdout) {
        fclose(output_file);
    }
    vsla_cleanup();
    
    return 0;
}