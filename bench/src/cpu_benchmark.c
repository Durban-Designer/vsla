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

void benchmark_vector_addition_cpu(int size1, int size2, int iterations) {
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
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_add(result, a, b);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_add(result, a, b);
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
    
    print_json_result("vsla_cpu", "vector_addition", size1, size2, max_size, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

void benchmark_convolution_cpu(int size1, int size2, int iterations) {
    // Create test tensors for convolution (Model A operation)
    uint64_t shape1[] = {size1};
    uint64_t shape2[] = {size2};
    uint64_t result_size = size1 + size2 - 1;
    uint64_t result_shape[] = {result_size};
    
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
    
    srand(42); // Consistent random seed
    for (int i = 0; i < size1; i++) {
        a_data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < size2; i++) {
        b_data[i] = (float)rand() / RAND_MAX;
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_conv(result, a, b);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_conv(result, a, b);
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
    
    print_json_result("vsla_cpu", "convolution", size1, size2, result_size, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

void benchmark_kronecker_cpu(int size1, int size2, int iterations) {
    // Create test tensors for Kronecker product (Model B operation)
    uint64_t shape1[] = {size1};
    uint64_t shape2[] = {size2};
    uint64_t result_size = size1 * size2;
    uint64_t result_shape[] = {result_size};
    
    vsla_tensor_t* a = vsla_new(1, shape1, VSLA_MODEL_B, VSLA_DTYPE_F32);
    vsla_tensor_t* b = vsla_new(1, shape2, VSLA_MODEL_B, VSLA_DTYPE_F32);
    vsla_tensor_t* result = vsla_new(1, result_shape, VSLA_MODEL_B, VSLA_DTYPE_F32);
    
    if (!a || !b || !result) {
        printf("{\"error\": \"VSLA tensor creation failed\"}\n");
        return;
    }
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    srand(42); // Consistent random seed
    for (int i = 0; i < size1; i++) {
        a_data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < size2; i++) {
        b_data[i] = (float)rand() / RAND_MAX;
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        vsla_kron(result, a, b);
    }
    
    // Benchmark
    double* times = malloc(iterations * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        vsla_kron(result, a, b);
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
    
    print_json_result("vsla_cpu", "kronecker", size1, size2, result_size, iterations,
                     mean_time, std_time, min_time, max_time);
    
    // Cleanup
    free(times);
    vsla_free(a);
    vsla_free(b);
    vsla_free(result);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s vector_add <size1> <size2> [iterations] | convolution <size1> <size2> [iterations] | kronecker <size1> <size2> [iterations]\n", argv[0]);
        return 1;
    }
    
    vsla_init();
    
    if (strcmp(argv[1], "vector_add") == 0 && argc >= 4) {
        int size1 = atoi(argv[2]);
        int size2 = atoi(argv[3]);
        int iterations = (argc >= 5) ? atoi(argv[4]) : 10;
        benchmark_vector_addition_cpu(size1, size2, iterations);
    } else if (strcmp(argv[1], "convolution") == 0 && argc >= 4) {
        int size1 = atoi(argv[2]);
        int size2 = atoi(argv[3]);
        int iterations = (argc >= 5) ? atoi(argv[4]) : 10;
        benchmark_convolution_cpu(size1, size2, iterations);
    } else if (strcmp(argv[1], "kronecker") == 0 && argc >= 4) {
        int size1 = atoi(argv[2]);
        int size2 = atoi(argv[3]);
        int iterations = (argc >= 5) ? atoi(argv[4]) : 10;
        benchmark_kronecker_cpu(size1, size2, iterations);
    } else {
        printf("{\"error\": \"Invalid arguments\"}\n");
        return 1;
    }
    
    vsla_cleanup();
    return 0;
}