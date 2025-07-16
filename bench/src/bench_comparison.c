#include "benchmark_utils.h"
#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <time.h>

// Simulate manual padding approach (what users do with NumPy/PyTorch)
typedef struct {
    double* data;
    size_t len;
    size_t capacity;
} manual_vector_t;

manual_vector_t* manual_vector_new(size_t len) {
    manual_vector_t* vec = malloc(sizeof(manual_vector_t));
    vec->len = len;
    vec->capacity = len;
    vec->data = calloc(len, sizeof(double));
    return vec;
}

void manual_vector_free(manual_vector_t* vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

// Manual zero-padding to common size (what TensorFlow/PyTorch users do)
manual_vector_t* manual_pad_to_size(manual_vector_t* vec, size_t target_size) {
    manual_vector_t* padded = manual_vector_new(target_size);
    
    // Copy original data
    for (size_t i = 0; i < vec->len && i < target_size; i++) {
        padded->data[i] = vec->data[i];
    }
    
    // Zero padding is automatic (calloc)
    return padded;
}

// Manual convolution with pre-padded arrays (TensorFlow/PyTorch equivalent)
manual_vector_t* manual_conv_padded(manual_vector_t* signal, manual_vector_t* kernel) {
    size_t out_len = signal->len + kernel->len - 1;
    manual_vector_t* result = manual_vector_new(out_len);
    
    // Direct convolution on padded arrays
    for (size_t i = 0; i < signal->len; i++) {
        for (size_t j = 0; j < kernel->len; j++) {
            size_t k = i + j;
            if (k < out_len) {
                result->data[k] += signal->data[i] * kernel->data[j];
            }
        }
    }
    
    return result;
}

// Benchmark VSLA approach (automatic shape handling)
static void benchmark_vsla_automatic(size_t signal_size, size_t kernel_size, 
                                     size_t iterations, size_t warmup) {
    // Create VSLA tensors with different shapes
    vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    size_t output_size = signal_size + kernel_size - 1;
    vsla_tensor_t* result = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!signal || !kernel || !result) {
        printf("Error: Failed to create VSLA tensors\n");
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
    
    // Warmup
    for (size_t i = 0; i < warmup; i++) {
        vsla_conv(result, signal, kernel);
    }
    
    // Benchmark timing
    benchmark_timer_t* timer = benchmark_timer_new(iterations);
    benchmark_timer_start(timer);
    
    for (size_t i = 0; i < iterations; i++) {
        struct timespec iter_start;
        clock_gettime(CLOCK_MONOTONIC, &iter_start);
        
        vsla_conv(result, signal, kernel);  // Automatic shape promotion
        
        struct timespec iter_end;
        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        
        double elapsed = (iter_end.tv_sec - iter_start.tv_sec) + 
                        (iter_end.tv_nsec - iter_start.tv_nsec) * 1e-9;
        timer->iteration_times[timer->num_iterations] = elapsed * 1e6;
        timer->num_iterations++;
    }
    
    benchmark_result_t bench_result = benchmark_timer_finish(timer);
    
    // Get system info
    system_info_t sys_info;
    get_system_info(&sys_info);
    
    // Print results
    print_benchmark_header("variable_shape_conv", "vsla_automatic");
    printf("  \"signal_size\": %zu,\\n", signal_size);
    printf("  \"kernel_size\": %zu,\\n", kernel_size);
    printf("  \"output_size\": %zu,\\n", output_size);
    print_benchmark_result(&bench_result, "variable_shape_conv", "vsla_automatic", &sys_info);
    print_benchmark_footer();
    printf(",\n");
    
    // Cleanup
    benchmark_timer_free(timer);
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(result);
}

// Benchmark manual padding approach (TensorFlow/PyTorch equivalent)
static void benchmark_manual_padding(size_t signal_size, size_t kernel_size,
                                    size_t iterations, size_t warmup) {
    // Create manual vectors with original sizes
    manual_vector_t* signal_orig = manual_vector_new(signal_size);
    manual_vector_t* kernel_orig = manual_vector_new(kernel_size);
    
    // Fill with same test data as VSLA
    for (size_t i = 0; i < signal_size; i++) {
        signal_orig->data[i] = sin(2.0 * M_PI * i / signal_size);
    }
    
    for (size_t i = 0; i < kernel_size; i++) {
        kernel_orig->data[i] = exp(-0.1 * i);
    }
    
    // Warmup
    for (size_t i = 0; i < warmup; i++) {
        // User must manually determine target size and pad
        size_t max_size = (signal_size > kernel_size) ? signal_size : kernel_size;
        manual_vector_t* signal_padded = manual_pad_to_size(signal_orig, max_size);
        manual_vector_t* kernel_padded = manual_pad_to_size(kernel_orig, max_size);
        manual_vector_t* result = manual_conv_padded(signal_padded, kernel_padded);
        
        manual_vector_free(signal_padded);
        manual_vector_free(kernel_padded);
        manual_vector_free(result);
    }
    
    // Benchmark timing (including manual padding overhead)
    benchmark_timer_t* timer = benchmark_timer_new(iterations);
    benchmark_timer_start(timer);
    
    for (size_t i = 0; i < iterations; i++) {
        struct timespec iter_start;
        clock_gettime(CLOCK_MONOTONIC, &iter_start);
        
        // Step 1: User must manually determine common size
        size_t max_size = (signal_size > kernel_size) ? signal_size : kernel_size;
        
        // Step 2: Manual padding (what TensorFlow/PyTorch requires)
        manual_vector_t* signal_padded = manual_pad_to_size(signal_orig, max_size);
        manual_vector_t* kernel_padded = manual_pad_to_size(kernel_orig, max_size);
        
        // Step 3: Convolution on padded arrays
        manual_vector_t* result = manual_conv_padded(signal_padded, kernel_padded);
        
        struct timespec iter_end;
        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        
        double elapsed = (iter_end.tv_sec - iter_start.tv_sec) + 
                        (iter_end.tv_nsec - iter_start.tv_nsec) * 1e-9;
        timer->iteration_times[timer->num_iterations] = elapsed * 1e6;
        timer->num_iterations++;
        
        // Cleanup
        manual_vector_free(signal_padded);
        manual_vector_free(kernel_padded);
        manual_vector_free(result);
    }
    
    benchmark_result_t bench_result = benchmark_timer_finish(timer);
    
    // Get system info
    system_info_t sys_info;
    get_system_info(&sys_info);
    
    // Print results
    print_benchmark_header("variable_shape_conv", "manual_padding");
    printf("  \"signal_size\": %zu,\\n", signal_size);
    printf("  \"kernel_size\": %zu,\\n", kernel_size);
    printf("  \"total_ops\": 3,\\n");  // pad signal + pad kernel + conv
    print_benchmark_result(&bench_result, "variable_shape_conv", "manual_padding", &sys_info);
    print_benchmark_footer();
    printf(",\n");
    
    // Cleanup
    manual_vector_free(signal_orig);
    manual_vector_free(kernel_orig);
}

static void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -s, --sizes SIZE1,SIZE2,...  Signal sizes to test (default: 128,256,512)\n");
    printf("  -k, --kernels SIZE1,SIZE2... Kernel sizes to test (default: 16,32,64)\n");
    printf("  -i, --iterations N           Number of iterations per test (default: 50)\n");
    printf("  -w, --warmup N               Number of warmup iterations (default: 5)\n");
    printf("  -o, --output FILE            Output results to file (default: stdout)\n");
    printf("  -h, --help                   Show this help message\n");
}

int main(int argc, char* argv[]) {
    size_t iterations = 50;
    size_t warmup = 5;
    size_t signal_sizes[] = {128, 256, 512};
    size_t kernel_sizes[] = {16, 32, 64};
    size_t num_signal_sizes = 3;
    size_t num_kernel_sizes = 3;
    
    // Parse command line arguments (simplified for this example)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Initialize VSLA
    vsla_init();
    
    printf("[\n");
    
    // Run comparison benchmarks for each size combination
    for (size_t s = 0; s < num_signal_sizes; s++) {
        for (size_t k = 0; k < num_kernel_sizes; k++) {
            size_t signal_size = signal_sizes[s];
            size_t kernel_size = kernel_sizes[k];
            
            printf("  // Signal size: %zu, Kernel size: %zu\n", signal_size, kernel_size);
            
            // Benchmark VSLA automatic approach
            benchmark_vsla_automatic(signal_size, kernel_size, iterations, warmup);
            
            // Benchmark manual padding approach  
            benchmark_manual_padding(signal_size, kernel_size, iterations, warmup);
        }
    }
    
    printf("  {\"end\": true}\n");
    printf("]\n");
    
    vsla_cleanup();
    
    return 0;
}