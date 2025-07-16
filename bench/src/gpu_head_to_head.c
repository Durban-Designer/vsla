#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "vsla/vsla.h"

double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

void print_json_result(const char* method, const char* operation, 
                      int size1, int size2, int size3, int iterations,
                      double mean_time_us, double std_time_us, 
                      double min_time_us, double max_time_us) {
    printf("{\n");
    printf("  \"method\": \"%s\",\n", method);
    printf("  \"operation\": \"%s\",\n", operation);
    printf("  \"size1\": %d,\n", size1);
    printf("  \"size2\": %d,\n", size2);
    printf("  \"size3\": %d,\n", size3);
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"mean_time_us\": %.3f,\n", mean_time_us);
    printf("  \"std_time_us\": %.3f,\n", std_time_us);
    printf("  \"min_time_us\": %.3f,\n", min_time_us);
    printf("  \"max_time_us\": %.3f\n", max_time_us);
    printf("}\n");
}

void benchmark_matrix_multiplication_vsla_gpu(int m, int n, int k, int iterations) {
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("{\"error\": \"VSLA GPU not available\"}\n");
        return;
    }
    
    // Create test matrices
    uint64_t shape_a[] = {m, k};
    uint64_t shape_b[] = {k, n};
    uint64_t shape_result[] = {m, n};
    
    vsla_tensor_t* a = vsla_new(2, shape_a, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_new(2, shape_b, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(2, shape_result, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (!a || !b || !result) {
        printf("{\"error\": \"VSLA tensor creation failed\"}\n");
        return;
    }
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    srand(42); // Consistent random seed
    for (int i = 0; i < m * k; i++) {
        a_data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b_data[i] = (float)rand() / RAND_MAX;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("{\"error\": \"VSLA GPU context creation failed\"}\n");
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Create GPU tensors
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(result, ctx);
    
    if (!gpu_a || !gpu_b || !gpu_result) {
        printf("{\"error\": \"VSLA GPU tensor creation failed\"}\n");
        vsla_gpu_destroy(ctx);
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Allocate GPU memory
    if (vsla_gpu_tensor_alloc(gpu_a, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_b, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_result, ctx) != VSLA_SUCCESS) {
        printf("{\"error\": \"VSLA GPU memory allocation failed\"}\n");
        vsla_gpu_tensor_free(gpu_a);
        vsla_gpu_tensor_free(gpu_b);
        vsla_gpu_tensor_free(gpu_result);
        vsla_gpu_destroy(ctx);
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Copy data to GPU
    vsla_gpu_tensor_copy_to_gpu(gpu_a, a->data, false);
    vsla_gpu_tensor_copy_to_gpu(gpu_b, b->data, false);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_gpu_matmul(gpu_result, gpu_a, gpu_b, ctx);
        vsla_gpu_tensor_sync(gpu_result);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_gpu_matmul(gpu_result, gpu_a, gpu_b, ctx);
        vsla_gpu_tensor_sync(gpu_result);
        double end = get_time_us();
        times[i] = end - start;
    }
    
    // Calculate statistics
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (int i = 0; i < iterations; i++) {
        sum += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    double mean_time = sum / iterations;
    
    double variance = 0.0;
    for (int i = 0; i < iterations; i++) {
        double diff = times[i] - mean_time;
        variance += diff * diff;
    }
    double std_time = sqrt(variance / iterations);
    
    print_json_result("vsla_gpu", "matrix_multiplication", m, n, k, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    vsla_gpu_destroy(ctx);
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

void benchmark_vector_addition_vsla_gpu(int size1, int size2, int iterations) {
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("{\"error\": \"VSLA GPU not available\"}\n");
        return;
    }
    
    // Create test vectors
    uint64_t shape1[] = {size1};
    uint64_t shape2[] = {size2};
    uint64_t max_size = (size1 > size2) ? size1 : size2;
    uint64_t result_shape[] = {max_size};
    
    vsla_tensor_t* a = vsla_new(1, shape1, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_new(1, shape2, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (!a || !b || !result) {
        printf("{\"error\": \"VSLA tensor creation failed\"}\n");
        return;
    }
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int i = 0; i < size1; i++) {
        a_data[i] = (float)i;
    }
    for (int i = 0; i < size2; i++) {
        b_data[i] = (float)i;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("{\"error\": \"VSLA GPU context creation failed\"}\n");
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Create GPU tensors
    vsla_gpu_tensor_t* gpu_a = vsla_gpu_tensor_from_cpu(a, ctx);
    vsla_gpu_tensor_t* gpu_b = vsla_gpu_tensor_from_cpu(b, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(result, ctx);
    
    if (!gpu_a || !gpu_b || !gpu_result) {
        printf("{\"error\": \"VSLA GPU tensor creation failed\"}\n");
        vsla_gpu_destroy(ctx);
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Allocate GPU memory
    if (vsla_gpu_tensor_alloc(gpu_a, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_b, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_result, ctx) != VSLA_SUCCESS) {
        printf("{\"error\": \"VSLA GPU memory allocation failed\"}\n");
        vsla_gpu_tensor_free(gpu_a);
        vsla_gpu_tensor_free(gpu_b);
        vsla_gpu_tensor_free(gpu_result);
        vsla_gpu_destroy(ctx);
        vsla_free(a); vsla_free(b); vsla_free(result);
        return;
    }
    
    // Copy data to GPU
    vsla_gpu_tensor_copy_to_gpu(gpu_a, a->data, false);
    vsla_gpu_tensor_copy_to_gpu(gpu_b, b->data, false);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
        vsla_gpu_tensor_sync(gpu_result);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_gpu_add(gpu_result, gpu_a, gpu_b, ctx);
        vsla_gpu_tensor_sync(gpu_result);
        double end = get_time_us();
        times[i] = end - start;
    }
    
    // Calculate statistics
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (int i = 0; i < iterations; i++) {
        sum += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    double mean_time = sum / iterations;
    
    double variance = 0.0;
    for (int i = 0; i < iterations; i++) {
        double diff = times[i] - mean_time;
        variance += diff * diff;
    }
    double std_time = sqrt(variance / iterations);
    
    print_json_result("vsla_gpu", "vector_addition", size1, size2, max_size, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_gpu_tensor_free(gpu_a);
    vsla_gpu_tensor_free(gpu_b);
    vsla_gpu_tensor_free(gpu_result);
    vsla_gpu_destroy(ctx);
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

void benchmark_convolution_vsla_gpu(int signal_size, int kernel_size, int iterations) {
    if (!vsla_has_gpu() || !vsla_gpu_is_available()) {
        printf("{\"error\": \"VSLA GPU not available\"}\n");
        return;
    }
    
    // Create test signal and kernel tensors
    uint64_t signal_shape[] = {signal_size};
    uint64_t kernel_shape[] = {kernel_size};
    uint64_t result_shape[] = {signal_size + kernel_size - 1}; // Convolution output size
    
    vsla_tensor_t* signal = vsla_new(1, signal_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* kernel = vsla_new(1, kernel_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    
    if (!signal || !kernel || !result) {
        printf("{\"error\": \"VSLA tensor creation failed\"}\n");
        return;
    }
    
    // Initialize data
    float* signal_data = (float*)signal->data;
    float* kernel_data = (float*)kernel->data;
    
    srand(42); // Consistent random seed
    for (int i = 0; i < signal_size; i++) {
        signal_data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel_data[i] = (float)rand() / RAND_MAX;
    }
    
    vsla_gpu_context_t* ctx = vsla_gpu_init(-1);
    if (!ctx) {
        printf("{\"error\": \"VSLA GPU context creation failed\"}\n");
        vsla_free(signal); vsla_free(kernel); vsla_free(result);
        return;
    }
    
    // Create GPU tensors
    vsla_gpu_tensor_t* gpu_signal = vsla_gpu_tensor_from_cpu(signal, ctx);
    vsla_gpu_tensor_t* gpu_kernel = vsla_gpu_tensor_from_cpu(kernel, ctx);
    vsla_gpu_tensor_t* gpu_result = vsla_gpu_tensor_from_cpu(result, ctx);
    
    if (!gpu_signal || !gpu_kernel || !gpu_result) {
        printf("{\"error\": \"VSLA GPU tensor creation failed\"}\n");
        vsla_gpu_destroy(ctx);
        vsla_free(signal); vsla_free(kernel); vsla_free(result);
        return;
    }
    
    // Allocate GPU memory
    if (vsla_gpu_tensor_alloc(gpu_signal, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_kernel, ctx) != VSLA_SUCCESS ||
        vsla_gpu_tensor_alloc(gpu_result, ctx) != VSLA_SUCCESS) {
        printf("{\"error\": \"VSLA GPU memory allocation failed\"}\n");
        vsla_gpu_tensor_free(gpu_signal);
        vsla_gpu_tensor_free(gpu_kernel);
        vsla_gpu_tensor_free(gpu_result);
        vsla_gpu_destroy(ctx);
        vsla_free(signal); vsla_free(kernel); vsla_free(result);
        return;
    }
    
    // Copy data to GPU
    vsla_gpu_tensor_copy_to_gpu(gpu_signal, signal->data, false);
    vsla_gpu_tensor_copy_to_gpu(gpu_kernel, kernel->data, false);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_gpu_conv_fft(gpu_result, gpu_signal, gpu_kernel, ctx);
        vsla_gpu_tensor_sync(gpu_result);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_gpu_conv_fft(gpu_result, gpu_signal, gpu_kernel, ctx);
        vsla_gpu_tensor_sync(gpu_result);
        double end = get_time_us();
        times[i] = end - start;
    }
    
    // Calculate statistics
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (int i = 0; i < iterations; i++) {
        sum += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    double mean_time = sum / iterations;
    
    double variance = 0.0;
    for (int i = 0; i < iterations; i++) {
        double diff = times[i] - mean_time;
        variance += diff * diff;
    }
    double std_time = sqrt(variance / iterations);
    
    print_json_result("vsla_gpu", "convolution", signal_size, kernel_size, 0, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_gpu_tensor_free(gpu_signal);
    vsla_gpu_tensor_free(gpu_kernel);
    vsla_gpu_tensor_free(gpu_result);
    vsla_gpu_destroy(ctx);
    vsla_free(signal);
    vsla_free(kernel);
    vsla_free(result);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s matrix_multiply <size> | vector_add <size1> <size2> | convolution <signal_size> <kernel_size>\n", argv[0]);
        return 1;
    }
    
    vsla_init();
    
    if (strcmp(argv[1], "matrix_multiply") == 0 && argc >= 3) {
        int size = atoi(argv[2]);
        int iterations = (argc >= 4) ? atoi(argv[3]) : 10;
        benchmark_matrix_multiplication_vsla_gpu(size, size, size, iterations);
    } else if (strcmp(argv[1], "vector_add") == 0 && argc >= 4) {
        int size1 = atoi(argv[2]);
        int size2 = atoi(argv[3]);
        int iterations = (argc >= 5) ? atoi(argv[4]) : 10;
        benchmark_vector_addition_vsla_gpu(size1, size2, iterations);
    } else if (strcmp(argv[1], "convolution") == 0 && argc >= 4) {
        int signal_size = atoi(argv[2]);
        int kernel_size = atoi(argv[3]);
        int iterations = (argc >= 5) ? atoi(argv[4]) : 10;
        benchmark_convolution_vsla_gpu(signal_size, kernel_size, iterations);
    } else {
        printf("Invalid arguments\n");
        return 1;
    }
    
    vsla_cleanup();
    return 0;
}