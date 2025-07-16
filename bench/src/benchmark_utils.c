/**
 * @file benchmark_utils.c
 * @brief Implementation of benchmark utilities
 */

#include "benchmark_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/utsname.h>

benchmark_timer_t* benchmark_timer_new(size_t max_iterations) {
    benchmark_timer_t* timer = malloc(sizeof(benchmark_timer_t));
    if (!timer) return NULL;
    
    timer->iteration_times = malloc(max_iterations * sizeof(double));
    if (!timer->iteration_times) {
        free(timer);
        return NULL;
    }
    
    timer->num_iterations = 0;
    timer->capacity = max_iterations;
    timer->start_memory = get_current_memory();
    
    return timer;
}

void benchmark_timer_free(benchmark_timer_t* timer) {
    if (!timer) return;
    free(timer->iteration_times);
    free(timer);
}

void benchmark_timer_start(benchmark_timer_t* timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start_wall);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &timer->start_cpu);
    timer->start_memory = get_current_memory();
    timer->num_iterations = 0;
}

void benchmark_timer_lap(benchmark_timer_t* timer) {
    if (timer->num_iterations >= timer->capacity) return;
    
    struct timespec end_wall;
    clock_gettime(CLOCK_MONOTONIC, &end_wall);
    
    double elapsed = (end_wall.tv_sec - timer->start_wall.tv_sec) + 
                    (end_wall.tv_nsec - timer->start_wall.tv_nsec) * 1e-9;
    
    timer->iteration_times[timer->num_iterations] = elapsed * 1e6; // Convert to microseconds
    timer->num_iterations++;
    
    // Reset start time for next iteration
    timer->start_wall = end_wall;
}

benchmark_result_t benchmark_timer_finish(benchmark_timer_t* timer) {
    benchmark_result_t result = {0};
    
    if (timer->num_iterations == 0) return result;
    
    // Calculate statistics
    result.iterations = timer->num_iterations;
    result.mean_time_us = calculate_mean(timer->iteration_times, timer->num_iterations);
    result.std_time_us = calculate_std(timer->iteration_times, timer->num_iterations, result.mean_time_us);
    
    // Find min/max
    result.min_time_us = timer->iteration_times[0];
    result.max_time_us = timer->iteration_times[0];
    for (size_t i = 1; i < timer->num_iterations; i++) {
        if (timer->iteration_times[i] < result.min_time_us) {
            result.min_time_us = timer->iteration_times[i];
        }
        if (timer->iteration_times[i] > result.max_time_us) {
            result.max_time_us = timer->iteration_times[i];
        }
    }
    
    result.peak_memory_bytes = get_peak_memory();
    
    return result;
}

size_t get_peak_memory(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024; // Convert KB to bytes on Linux
    }
    return 0;
}

size_t get_current_memory(void) {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[256];
    size_t memory = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu kB", &memory);
            memory *= 1024; // Convert to bytes
            break;
        }
    }
    
    fclose(file);
    return memory;
}

void get_system_info(system_info_t* info) {
    if (!info) return;
    
    struct utsname uname_data;
    uname(&uname_data);
    
    // Get CPU info
    FILE* file = fopen("/proc/cpuinfo", "r");
    if (file) {
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    strncpy(info->cpu_model, colon + 2, sizeof(info->cpu_model) - 1);
                    info->cpu_model[sizeof(info->cpu_model) - 1] = '\0';
                    // Remove newline
                    char* newline = strchr(info->cpu_model, '\n');
                    if (newline) *newline = '\0';
                }
                break;
            }
        }
        fclose(file);
    }
    
    // OS version
    snprintf(info->os_version, sizeof(info->os_version), "%s %s", 
             uname_data.sysname, uname_data.release);
    
    // Compiler info
    #ifdef __GNUC__
    snprintf(info->compiler, sizeof(info->compiler), "GCC %d.%d.%d", 
             __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #elif defined(__clang__)
    snprintf(info->compiler, sizeof(info->compiler), "Clang %s", __clang_version__);
    #else
    strncpy(info->compiler, "Unknown", sizeof(info->compiler));
    #endif
    
    // Memory info
    file = fopen("/proc/meminfo", "r");
    if (file) {
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                size_t mem_kb;
                sscanf(line, "MemTotal: %zu kB", &mem_kb);
                info->total_memory_gb = mem_kb / (1024 * 1024);
                break;
            }
        }
        fclose(file);
    }
    
    // Number of cores
    info->num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // BLAS library (simplified detection)
    #ifdef OPENBLAS_VERSION
    strncpy(info->blas_library, "OpenBLAS", sizeof(info->blas_library));
    #elif defined(MKL_VERSION)
    strncpy(info->blas_library, "Intel MKL", sizeof(info->blas_library));
    #else
    strncpy(info->blas_library, "Generic BLAS", sizeof(info->blas_library));
    #endif
}

double calculate_mean(const double* values, size_t count) {
    if (count == 0) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}

double calculate_std(const double* values, size_t count, double mean) {
    if (count <= 1) return 0.0;
    
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / (count - 1));
}

double calculate_median(double* values, size_t count) {
    if (count == 0) return 0.0;
    
    // Simple bubble sort for small arrays
    for (size_t i = 0; i < count - 1; i++) {
        for (size_t j = 0; j < count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                double temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    if (count % 2 == 0) {
        return (values[count/2 - 1] + values[count/2]) / 2.0;
    } else {
        return values[count/2];
    }
}

void remove_outliers(double* values, size_t* count, double percentile) {
    if (*count <= 4) return; // Need minimum samples
    
    // Sort values
    for (size_t i = 0; i < *count - 1; i++) {
        for (size_t j = 0; j < *count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                double temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    size_t remove_count = (size_t)(*count * percentile);
    size_t new_count = *count - 2 * remove_count;
    
    // Shift remaining values to start of array
    for (size_t i = 0; i < new_count; i++) {
        values[i] = values[i + remove_count];
    }
    
    *count = new_count;
}

void print_benchmark_header(const char* benchmark_name, const char* method) {
    printf("{\n");
    printf("  \"benchmark\": \"%s\",\n", benchmark_name);
    printf("  \"method\": \"%s\",\n", method);
    printf("  \"timestamp\": \"%ld\",\n", time(NULL));
}

void print_benchmark_result(const benchmark_result_t* result, 
                           const char* benchmark_name,
                           const char* method,
                           const system_info_t* sys_info) {
    printf("  \"results\": {\n");
    printf("    \"iterations\": %lu,\n", result->iterations);
    printf("    \"mean_time_us\": %.3f,\n", result->mean_time_us);
    printf("    \"std_time_us\": %.3f,\n", result->std_time_us);
    printf("    \"min_time_us\": %.3f,\n", result->min_time_us);
    printf("    \"max_time_us\": %.3f,\n", result->max_time_us);
    printf("    \"peak_memory_mb\": %.3f\n", result->peak_memory_bytes / (1024.0 * 1024.0));
    printf("  },\n");
    
    printf("  \"system_info\": {\n");
    printf("    \"cpu\": \"%s\",\n", sys_info->cpu_model);
    printf("    \"os\": \"%s\",\n", sys_info->os_version);
    printf("    \"compiler\": \"%s\",\n", sys_info->compiler);
    printf("    \"memory_gb\": %zu,\n", sys_info->total_memory_gb);
    printf("    \"cores\": %d,\n", sys_info->num_cores);
    printf("    \"blas\": \"%s\"\n", sys_info->blas_library);
    printf("  }\n");
}

void print_benchmark_footer(void) {
    printf("}\n");
}

size_array_t parse_size_list(const char* str) {
    size_array_t result = {0};
    
    if (!str) return result;
    
    // Count commas to estimate array size
    size_t count = 1;
    for (const char* p = str; *p; p++) {
        if (*p == ',') count++;
    }
    
    result.values = malloc(count * sizeof(size_t));
    if (!result.values) return result;
    
    char* str_copy = strdup(str);
    char* token = strtok(str_copy, ",");
    
    result.count = 0;
    while (token && result.count < count) {
        result.values[result.count] = (size_t)strtoul(token, NULL, 10);
        result.count++;
        token = strtok(NULL, ",");
    }
    
    free(str_copy);
    return result;
}

void free_size_array(size_array_t* array) {
    if (array && array->values) {
        free(array->values);
        array->values = NULL;
        array->count = 0;
    }
}

void generate_random_data(double* data, size_t count, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < count; i++) {
        data[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Range [-1, 1]
    }
}

void generate_test_matrix(double* matrix, size_t rows, size_t cols, unsigned int seed) {
    generate_random_data(matrix, rows * cols, seed);
}

int compare_results(const double* a, const double* b, size_t count, double tolerance) {
    for (size_t i = 0; i < count; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            return 0; // Results differ
        }
    }
    return 1; // Results match
}

double compute_relative_error(const double* computed, const double* reference, size_t count) {
    double max_error = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (reference[i] != 0.0) {
            double rel_error = fabs((computed[i] - reference[i]) / reference[i]);
            if (rel_error > max_error) {
                max_error = rel_error;
            }
        }
    }
    return max_error;
}