/**
 * @file test_framework.h
 * @brief Simple test framework for VSLA library
 * 
 * @copyright MIT License
 */

#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Test statistics */
extern int g_tests_run;
extern int g_tests_passed;
extern int g_tests_failed;

/* Test suite management */
typedef struct {
    const char* name;
    void (*setup)(void);
    void (*teardown)(void);
    void (*run_tests)(void);
} test_suite_t;

/* Macros for test definition */
#define TEST_SUITE(name) \
    void test_suite_##name(void)

#define RUN_TEST(test_func) \
    do { \
        printf("  Running %s... ", #test_func); \
        fflush(stdout); \
        g_tests_run++; \
        if (test_func()) { \
            printf("PASS\n"); \
            g_tests_passed++; \
        } else { \
            printf("FAIL\n"); \
            g_tests_failed++; \
        } \
    } while(0)

/* Assertion macros */
#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            printf("\n    ASSERTION FAILED: %s at %s:%d\n", #expr, __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_FALSE(expr) \
    ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            printf("\n    ASSERTION FAILED: %s != %s (%ld != %ld) at %s:%d\n", \
                   #a, #b, (long)(a), (long)(b), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            printf("\n    ASSERTION FAILED: %s == %s (%ld == %ld) at %s:%d\n", \
                   #a, #b, (long)(a), (long)(b), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_DOUBLE_EQ(a, b, eps) \
    do { \
        double _diff = fabs((double)(a) - (double)(b)); \
        if (_diff > (eps)) { \
            printf("\n    ASSERTION FAILED: %s != %s (%.6f != %.6f, diff=%.6f > %.6f) at %s:%d\n", \
                   #a, #b, (double)(a), (double)(b), _diff, (double)(eps), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_NULL(ptr) \
    do { \
        if ((ptr) != NULL) { \
            printf("\n    ASSERTION FAILED: %s is not NULL at %s:%d\n", #ptr, __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf("\n    ASSERTION FAILED: %s is NULL at %s:%d\n", #ptr, __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_STR_EQ(a, b) \
    do { \
        if (strcmp((a), (b)) != 0) { \
            printf("\n    ASSERTION FAILED: %s != %s (\"%s\" != \"%s\") at %s:%d\n", \
                   #a, #b, (a), (b), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

/* Test function declaration */
#define DECLARE_TEST(name) int name(void)

/* Memory leak detection helpers */
void* test_malloc(size_t size);
void test_free(void* ptr);
int test_check_leaks(void);
void test_reset_memory_tracking(void);

/* Test utilities */
void print_test_summary(void);
int run_test_suite(const char* suite_name);
void register_test_suite(const test_suite_t* suite);

/* Assertion macros for void functions */
#define ASSERT_TRUE_VOID(expr) \
    do { \
        if (!(expr)) { \
            printf("\n    ASSERTION FAILED: %s at %s:%d\n", #expr, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_EQ_VOID(a, b) \
    do { \
        if ((a) != (b)) { \
            printf("\n    ASSERTION FAILED: %s != %s (%ld != %ld) at %s:%d\n", \
                   #a, #b, (long)(a), (long)(b), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_NOT_NULL_VOID(ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf("\n    ASSERTION FAILED: %s is NULL at %s:%d\n", #ptr, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ_VOID(a, b, eps) \
    do { \
        double _diff = fabs((double)(a) - (double)(b)); \
        if (_diff > (eps)) { \
            printf("\n    ASSERTION FAILED: %s != %s (%.6f != %.6f, diff=%.6f > %.6f) at %s:%d\n", \
                   #a, #b, (double)(a), (double)(b), _diff, (double)(eps), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ(a, b, eps) \
    do { \
        double _diff = fabs((double)(a) - (double)(b)); \
        if (_diff > (eps)) { \
            printf("\n    ASSERTION FAILED: %s != %s (%.6f != %.6f, diff=%.6f > %.6f) at %s:%d\n", \
                   #a, #b, (double)(a), (double)(b), _diff, (double)(eps), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

/* Simplified test case macro for void functions */
#define TEST_CASE(name, func) \
    do { \
        printf("    Running %s...", name); \
        func(); \
        printf(" PASSED\n"); \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* TEST_FRAMEWORK_H */