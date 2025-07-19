/**
 * @file test_unified_framework.h
 * @brief Unified test framework for VSLA operations across all backends
 * 
 * This framework provides a unified way to test all VSLA operations
 * using the context-based interface. Tests can be run against different
 * backends by simply changing the configuration.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_TEST_UNIFIED_FRAMEWORK_H
#define VSLA_TEST_UNIFIED_FRAMEWORK_H

#include <vsla/vsla.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Test result structure */
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
    char* current_test_name;
} test_results_t;

/* Test configuration */
typedef struct {
    vsla_backend_t backend;
    int device_id;
    bool verbose;
    double tolerance;
} test_config_t;

/* Global test state */
extern test_results_t g_test_results;
extern test_config_t g_test_config;
extern vsla_context_t* g_test_ctx;

/* Test macros */
#define UNIFIED_TEST_INIT(backend) unified_test_framework_init(backend)
#define UNIFIED_TEST_CLEANUP() unified_test_framework_cleanup()
#define UNIFIED_TEST_CASE(name) unified_test_case_begin(name)
#define UNIFIED_TEST_END() unified_test_case_end()
#define ASSERT_SUCCESS(expr) unified_test_assert_success(expr, #expr, __FILE__, __LINE__)
#define ASSERT_ERROR(expr, expected_error) unified_test_assert_error(expr, expected_error, #expr, __FILE__, __LINE__)
#define ASSERT_NULL(ptr) unified_test_assert_null(ptr, #ptr, __FILE__, __LINE__)
#define ASSERT_NOT_NULL(ptr) unified_test_assert_not_null(ptr, #ptr, __FILE__, __LINE__)
#define ASSERT_DOUBLE_EQ(a, b) unified_test_assert_double_eq(a, b, #a " == " #b, __FILE__, __LINE__)
#define ASSERT_TENSOR_EQ(a, b) unified_test_assert_tensor_eq(a, b, #a " == " #b, __FILE__, __LINE__)
#define ASSERT_SHAPE_EQ(tensor, expected_shape, rank) unified_test_assert_shape_eq(tensor, expected_shape, rank, __FILE__, __LINE__)

/* Framework functions */
bool unified_test_framework_init(vsla_backend_t backend);
void unified_test_framework_cleanup(void);
void unified_test_case_begin(const char* name);
void unified_test_case_end(void);
void unified_test_print_summary(void);

/* Assertion functions */
bool unified_test_assert_success(vsla_error_t result, const char* expr, const char* file, int line);
bool unified_test_assert_error(vsla_error_t result, vsla_error_t expected, const char* expr, const char* file, int line);
bool unified_test_assert_null(const void* ptr, const char* expr, const char* file, int line);
bool unified_test_assert_not_null(const void* ptr, const char* expr, const char* file, int line);
bool unified_test_assert_double_eq(double a, double b, const char* expr, const char* file, int line);
bool unified_test_assert_tensor_eq(const vsla_tensor_t* a, const vsla_tensor_t* b, const char* expr, const char* file, int line);
bool unified_test_assert_shape_eq(const vsla_tensor_t* tensor, const uint64_t* expected_shape, uint8_t rank, const char* file, int line);

/* Utility functions */
vsla_tensor_t* unified_test_create_tensor_1d(uint64_t size, vsla_model_t model, vsla_dtype_t dtype);
vsla_tensor_t* unified_test_create_tensor_2d(uint64_t rows, uint64_t cols, vsla_model_t model, vsla_dtype_t dtype);
void unified_test_fill_tensor_sequential(vsla_tensor_t* tensor, double start_value);
void unified_test_fill_tensor_random(vsla_tensor_t* tensor, double min_val, double max_val);
void unified_test_print_tensor(const vsla_tensor_t* tensor, const char* name);

/* Test data generators */
void unified_test_generate_test_data_add(vsla_tensor_t* a, vsla_tensor_t* b, vsla_tensor_t* expected);
void unified_test_generate_test_data_matmul(vsla_tensor_t* a, vsla_tensor_t* b, vsla_tensor_t* expected);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_TEST_UNIFIED_FRAMEWORK_H */