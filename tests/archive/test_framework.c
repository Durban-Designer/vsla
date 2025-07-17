/**
 * @file test_framework.c
 * @brief Simple test framework implementation
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include <stdlib.h>
#include <string.h>

/* Global test statistics */
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

/* Memory tracking for leak detection */
static int g_malloc_count = 0;
static int g_free_count = 0;

void* test_malloc(size_t size) {
    g_malloc_count++;
    return malloc(size);
}

void test_free(void* ptr) {
    if (ptr) {
        g_free_count++;
        free(ptr);
    }
}

int test_check_leaks(void) {
    return g_malloc_count - g_free_count;
}

void test_reset_memory_tracking(void) {
    g_malloc_count = 0;
    g_free_count = 0;
}

void print_test_summary(void) {
    printf("\n");
    printf("=== Test Summary ===\n");
    printf("Tests run:    %d\n", g_tests_run);
    printf("Tests passed: %d\n", g_tests_passed);
    printf("Tests failed: %d\n", g_tests_failed);
    
    if (g_tests_failed == 0) {
        printf("Result: ALL TESTS PASSED\n");
    } else {
        printf("Result: %d TESTS FAILED\n", g_tests_failed);
    }
    
    int leaks = test_check_leaks();
    if (leaks > 0) {
        printf("Memory leaks: %d allocations not freed\n", leaks);
    } else {
        printf("Memory leaks: None detected\n");
    }
    printf("==================\n");
}

/* Test suite registry */
#define MAX_SUITES 32
static const test_suite_t* g_suites[MAX_SUITES];
static int g_num_suites = 0;

void register_test_suite(const test_suite_t* suite) {
    if (g_num_suites < MAX_SUITES) {
        g_suites[g_num_suites++] = suite;
    }
}

int run_test_suite(const char* suite_name) {
    for (int i = 0; i < g_num_suites; i++) {
        if (strcmp(g_suites[i]->name, suite_name) == 0) {
            printf("Running test suite: %s\n", suite_name);
            
            if (g_suites[i]->setup) {
                g_suites[i]->setup();
            }
            
            test_reset_memory_tracking();
            g_suites[i]->run_tests();
            
            if (g_suites[i]->teardown) {
                g_suites[i]->teardown();
            }
            
            return 1;
        }
    }
    return 0;  /* Suite not found */
}

/* Main test runner */
int main(int argc, char* argv[]) {
    /* Declare test suites */
    extern void register_all_test_suites(void);
    register_all_test_suites();
    
    printf("VSLA Library Test Suite\n");
    printf("=======================\n");
    
    /* Parse command line arguments */
    const char* target_suite = NULL;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--suite=", 8) == 0) {
            target_suite = argv[i] + 8;
            break;
        }
    }
    
    if (target_suite) {
        /* Run specific test suite */
        if (!run_test_suite(target_suite)) {
            printf("Test suite '%s' not found\n", target_suite);
            return 1;
        }
    } else {
        /* Run all test suites */
        for (int i = 0; i < g_num_suites; i++) {
            printf("Running test suite: %s\n", g_suites[i]->name);
            
            if (g_suites[i]->setup) {
                g_suites[i]->setup();
            }
            
            test_reset_memory_tracking();
            g_suites[i]->run_tests();
            
            if (g_suites[i]->teardown) {
                g_suites[i]->teardown();
            }
            
            printf("\n");
        }
    }
    
    print_test_summary();
    return g_tests_failed > 0 ? 1 : 0;
}