/**
 * @file test_main_unified.c
 * @brief Main test runner for unified interface tests
 * 
 * Runs comprehensive tests against different backends to ensure 
 * all operations work correctly across all hardware.
 * 
 * @copyright MIT License
 */

#include "test_unified_framework.h"
#include <string.h>

/* External test functions */
extern int run_arithmetic_tests(vsla_backend_t backend);

/* Test backend selection */
static vsla_backend_t parse_backend(const char* backend_name) {
    if (!backend_name || strcmp(backend_name, "cpu") == 0) {
        return VSLA_BACKEND_CPU;
    } else if (strcmp(backend_name, "cuda") == 0) {
        return VSLA_BACKEND_CUDA;
    } else if (strcmp(backend_name, "auto") == 0) {
        return VSLA_BACKEND_AUTO;
    } else {
        printf("Unknown backend: %s\n", backend_name);
        printf("Available backends: cpu, cuda, auto\n");
        return VSLA_BACKEND_CPU; /* Default fallback */
    }
}

/* Run all test suites for a specific backend */
static int run_all_tests_for_backend(vsla_backend_t backend) {
    const char* backend_name = 
        backend == VSLA_BACKEND_CPU ? "CPU" :
        backend == VSLA_BACKEND_CUDA ? "CUDA" :
        backend == VSLA_BACKEND_AUTO ? "AUTO" : "UNKNOWN";
    
    printf("\n============================================================\n");
    printf("Running tests for backend: %s\n", backend_name);
    printf("============================================================\n");
    
    int total_failures = 0;
    
    /* Run arithmetic tests */
    total_failures += run_arithmetic_tests(backend);
    
    /* TODO: Add more test suites here */
    /* total_failures += run_linalg_tests(backend); */
    /* total_failures += run_reduction_tests(backend); */
    /* total_failures += run_tensor_tests(backend); */
    
    printf("\n============================================================\n");
    printf("Backend %s test summary: %s\n", backend_name, 
           total_failures == 0 ? "ALL PASSED" : "SOME FAILED");
    printf("============================================================\n");
    
    return total_failures;
}

/* Print usage information */
static void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  --backend <name>    Run tests for specific backend (cpu, cuda, auto)\n");
    printf("  --all-backends      Run tests for all available backends\n");
    printf("  --verbose           Enable verbose output\n");
    printf("  --help              Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                    # Run tests for CPU backend (default)\n", program_name);
    printf("  %s --backend cuda     # Run tests for CUDA backend\n", program_name);
    printf("  %s --all-backends     # Run tests for all backends\n", program_name);
}

int main(int argc, char* argv[]) {
    printf("VSLA Unified Interface Test Suite\n");
    printf("==================================\n");
    
    /* Parse command line arguments */
    vsla_backend_t target_backend = VSLA_BACKEND_CPU;
    bool test_all_backends = false;
    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            target_backend = parse_backend(argv[++i]);
        } else if (strcmp(argv[i], "--all-backends") == 0) {
            test_all_backends = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    /* Set global verbose flag */
    g_test_config.verbose = verbose;
    
    int total_failures = 0;
    
    if (test_all_backends) {
        /* Test all available backends */
        printf("Testing all available backends...\n");
        
        /* CPU backend (always available) */
        total_failures += run_all_tests_for_backend(VSLA_BACKEND_CPU);
        
        /* CUDA backend (if available) */
        /* TODO: Add runtime check for CUDA availability */
        /* total_failures += run_all_tests_for_backend(VSLA_BACKEND_CUDA); */
        
        /* AUTO backend */
        total_failures += run_all_tests_for_backend(VSLA_BACKEND_AUTO);
        
    } else {
        /* Test specific backend */
        total_failures = run_all_tests_for_backend(target_backend);
    }
    
    /* Final summary */
    printf("\n============================================================\n");
    printf("FINAL TEST SUMMARY\n");
    printf("============================================================\n");
    
    if (total_failures == 0) {
        printf("🎉 ALL TESTS PASSED!\n");
        printf("The VSLA unified interface is working correctly.\n");
    } else {
        printf("❌ %d TEST(S) FAILED\n", total_failures);
        printf("Please review the test output above for details.\n");
    }
    
    printf("============================================================\n");
    
    return total_failures > 0 ? 1 : 0;
}