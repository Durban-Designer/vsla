/**
 * @file bench_fair_comparison.c
 * @brief Fair benchmark comparison between VSLA and established C libraries
 */

#include "benchmark_utils.h"
#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

// Test vector addition: VSLA vs OpenBLAS
static void benchmark_vector_addition(size_t size1, size_t size2, size_t iterations) {
    printf("// Vector Addition Benchmark: VSLA vs OpenBLAS\n");
    printf("// Size1: %zu, Size2: %zu\n", size1, size2);
    
    // VSLA approach
    {
        vsla_tensor_t* a = vsla_new(1, &size1, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* b = vsla_new(1, &size2, VSLA_MODEL_A, VSLA_DTYPE_F64);
        size_t result_size = (size1 > size2) ? size1 : size2;
        vsla_tensor_t* result = vsla_new(1, &result_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Fill with test data
        for (size_t i = 0; i < size1; i++) {
            uint64_t idx = i;
            vsla_set_f64(a, &idx, (double)i);
        }
        for (size_t i = 0; i < size2; i++) {
            uint64_t idx = i;
            vsla_set_f64(b, &idx, (double)i);
        }
        
        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (size_t i = 0; i < iterations; i++) {
            vsla_add(result, a, b);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double vsla_time = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / iterations / 1000.0;
        
        printf("{\n");
        printf("  \"method\": \"vsla_automatic\",\n");
        printf("  \"mean_time_us\": %.3f,\n", vsla_time);
        printf("  \"memory_efficient\": true\n");
        printf("},\n");
        
        vsla_free(a);
        vsla_free(b);
        vsla_free(result);
    }
    
    // OpenBLAS approach (manual padding)
    {
        size_t result_size = (size1 > size2) ? size1 : size2;
        double* a = calloc(result_size, sizeof(double));
        double* b = calloc(result_size, sizeof(double));
        double* result = calloc(result_size, sizeof(double));
        
        // Fill with test data (manual padding)
        for (size_t i = 0; i < size1; i++) {
            a[i] = (double)i;
        }
        for (size_t i = 0; i < size2; i++) {
            b[i] = (double)i;
        }
        
        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (size_t i = 0; i < iterations; i++) {
            memcpy(result, a, result_size * sizeof(double));
            cblas_daxpy(result_size, 1.0, b, 1, result, 1);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double blas_time = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / iterations / 1000.0;
        
        printf("{\n");
        printf("  \"method\": \"openblas_manual_padding\",\n");
        printf("  \"mean_time_us\": %.3f,\n", blas_time);
        printf("  \"memory_efficient\": false\n");
        printf("}\n");
        
        free(a);
        free(b);
        free(result);
    }
}

// Test convolution: VSLA vs FFTW
static void benchmark_convolution(size_t signal_size, size_t kernel_size, size_t iterations) {
    printf("// Convolution Benchmark: VSLA vs FFTW\n");
    printf("// Signal: %zu, Kernel: %zu\n", signal_size, kernel_size);
    
    // VSLA approach
    {
        vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        size_t result_size = signal_size + kernel_size - 1;
        vsla_tensor_t* result = vsla_new(1, &result_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        // Fill with test data
        for (size_t i = 0; i < signal_size; i++) {
            uint64_t idx = i;
            vsla_set_f64(signal, &idx, sin(2.0 * M_PI * i / signal_size));
        }
        for (size_t i = 0; i < kernel_size; i++) {
            uint64_t idx = i;
            vsla_set_f64(kernel, &idx, exp(-0.1 * i));
        }
        
        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (size_t i = 0; i < iterations; i++) {
            vsla_conv(result, signal, kernel);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double vsla_time = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / iterations / 1000.0;
        
        printf("{\n");
        printf("  \"method\": \"vsla_fft\",\n");
        printf("  \"mean_time_us\": %.3f,\n", vsla_time);
        printf("  \"automatic_shapes\": true\n");
        printf("},\n");
        
        vsla_free(signal);
        vsla_free(kernel);
        vsla_free(result);
    }
    
#ifdef HAVE_FFTW3
    // FFTW approach
    {
        size_t result_size = signal_size + kernel_size - 1;
        size_t fft_size = 1;
        while (fft_size < result_size) fft_size *= 2;
        
        double* signal = calloc(fft_size, sizeof(double));
        double* kernel = calloc(fft_size, sizeof(double));
        fftw_complex* signal_fft = fftw_alloc_complex(fft_size);
        fftw_complex* kernel_fft = fftw_alloc_complex(fft_size);
        fftw_complex* result_fft = fftw_alloc_complex(fft_size);
        double* result = calloc(fft_size, sizeof(double));
        
        // Fill with test data
        for (size_t i = 0; i < signal_size; i++) {
            signal[i] = sin(2.0 * M_PI * i / signal_size);
        }
        for (size_t i = 0; i < kernel_size; i++) {
            kernel[i] = exp(-0.1 * i);
        }
        
        // Create FFTW plans
        fftw_plan signal_plan = fftw_plan_dft_r2c_1d(fft_size, signal, signal_fft, FFTW_ESTIMATE);
        fftw_plan kernel_plan = fftw_plan_dft_r2c_1d(fft_size, kernel, kernel_fft, FFTW_ESTIMATE);
        fftw_plan result_plan = fftw_plan_dft_c2r_1d(fft_size, result_fft, result, FFTW_ESTIMATE);
        
        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (size_t i = 0; i < iterations; i++) {
            fftw_execute(signal_plan);
            fftw_execute(kernel_plan);
            
            // Multiply in frequency domain
            for (size_t j = 0; j < fft_size/2 + 1; j++) {
                double real = signal_fft[j][0] * kernel_fft[j][0] - signal_fft[j][1] * kernel_fft[j][1];
                double imag = signal_fft[j][0] * kernel_fft[j][1] + signal_fft[j][1] * kernel_fft[j][0];
                result_fft[j][0] = real;
                result_fft[j][1] = imag;
            }
            
            fftw_execute(result_plan);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double fftw_time = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / iterations / 1000.0;
        
        printf("{\n");
        printf("  \"method\": \"fftw3_manual\",\n");
        printf("  \"mean_time_us\": %.3f,\n", fftw_time);
        printf("  \"automatic_shapes\": false\n");
        printf("}\n");
        
        fftw_destroy_plan(signal_plan);
        fftw_destroy_plan(kernel_plan);
        fftw_destroy_plan(result_plan);
        fftw_free(signal_fft);
        fftw_free(kernel_fft);
        fftw_free(result_fft);
        free(signal);
        free(kernel);
        free(result);
    }
#else
    printf("{\n");
    printf("  \"method\": \"fftw3_manual\",\n");
    printf("  \"mean_time_us\": \"not_available\",\n");
    printf("  \"automatic_shapes\": false\n");
    printf("}\n");
#endif
}

int main(int argc, char* argv[]) {
    printf("[\n");
    printf("// Fair Benchmark: VSLA vs Established C Libraries\n");
    printf("// System: %s\n", getenv("USER"));
    printf("// Date: %s\n", __DATE__);
    
    // Initialize VSLA
    vsla_init();
    
    // Vector addition tests
    printf("// Vector Addition Tests\n");
    benchmark_vector_addition(100, 150, 1000);
    printf(",\n");
    benchmark_vector_addition(1000, 1500, 1000);
    printf(",\n");
    
    // Convolution tests
    printf("// Convolution Tests\n");
    benchmark_convolution(128, 32, 100);
    printf(",\n");
    benchmark_convolution(512, 64, 100);
    
    vsla_cleanup();
    
    printf("\n]\n");
    return 0;
}