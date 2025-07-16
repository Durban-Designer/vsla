/**
 * @file benchmark_utils.h
 * @brief Common utilities for VSLA benchmarks
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <sys/resource.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Benchmark result structure
 */
typedef struct {
    double wall_time_us;      ///< Wall clock time in microseconds
    double cpu_time_us;       ///< CPU time in microseconds  
    size_t peak_memory_bytes; ///< Peak memory usage in bytes
    uint64_t iterations;      ///< Number of iterations performed
    double mean_time_us;      ///< Mean time per iteration
    double std_time_us;       ///< Standard deviation of times
    double min_time_us;       ///< Minimum time observed
    double max_time_us;       ///< Maximum time observed
} benchmark_result_t;

/**
 * @brief System information for reproducibility
 */
typedef struct {
    char cpu_model[256];      ///< CPU model string
    char os_version[128];     ///< Operating system version
    char compiler[128];       ///< Compiler version
    size_t total_memory_gb;   ///< Total system memory in GB
    int num_cores;            ///< Number of CPU cores
    char blas_library[64];    ///< BLAS implementation
} system_info_t;

/**
 * @brief Benchmark timer for high-resolution timing
 */
typedef struct {
    struct timespec start_wall;
    struct timespec start_cpu;
    size_t start_memory;
    double *iteration_times;
    size_t num_iterations;
    size_t capacity;
} benchmark_timer_t;

// Timing functions
benchmark_timer_t* benchmark_timer_new(size_t max_iterations);
void benchmark_timer_free(benchmark_timer_t* timer);
void benchmark_timer_start(benchmark_timer_t* timer);
void benchmark_timer_lap(benchmark_timer_t* timer);
benchmark_result_t benchmark_timer_finish(benchmark_timer_t* timer);

// High-resolution timing
static inline double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline double get_cpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Memory measurement
size_t get_peak_memory(void);
size_t get_current_memory(void);

// System information
void get_system_info(system_info_t* info);

// Statistical functions
double calculate_mean(const double* values, size_t count);
double calculate_std(const double* values, size_t count, double mean);
double calculate_median(double* values, size_t count);  // Note: modifies array
void remove_outliers(double* values, size_t* count, double percentile);

// JSON output functions
void print_benchmark_header(const char* benchmark_name, const char* method);
void print_benchmark_result(const benchmark_result_t* result, 
                           const char* benchmark_name,
                           const char* method,
                           const system_info_t* sys_info);
void print_benchmark_footer(void);

// Command-line parsing helpers
typedef struct {
    size_t* values;
    size_t count;
} size_array_t;

size_array_t parse_size_list(const char* str);
void free_size_array(size_array_t* array);

// Test data generation
void generate_random_data(double* data, size_t count, unsigned int seed);
void generate_test_matrix(double* matrix, size_t rows, size_t cols, unsigned int seed);

// Verification helpers
int compare_results(const double* a, const double* b, size_t count, double tolerance);
double compute_relative_error(const double* computed, const double* reference, size_t count);

// Benchmark runner macros
#define BENCHMARK_ITERATIONS_DEFAULT 100
#define BENCHMARK_WARMUP_DEFAULT 5
#define BENCHMARK_OUTLIER_PERCENTILE 0.05

#define RUN_BENCHMARK(timer, code) do { \
    benchmark_timer_start(timer); \
    code; \
    benchmark_timer_lap(timer); \
} while(0)

#ifdef __cplusplus
}
#endif

#endif /* BENCHMARK_UTILS_H */