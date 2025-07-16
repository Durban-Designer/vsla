/**
 * @file test_utils.c
 * @brief Tests for utility functions and library initialization
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <string.h>

static void utils_test_setup(void) {
    // Ensure we start in a clean state
}

static void utils_test_teardown(void) {
    // Cleanup after tests
}

static int test_library_initialization(void) {
    // Test basic initialization
    ASSERT_EQ(VSLA_SUCCESS, vsla_init());
    
    // Test double initialization (should be safe)
    ASSERT_EQ(VSLA_SUCCESS, vsla_init());
    
    // Test cleanup
    ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    
    // Test double cleanup (should be safe)
    ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    
    // Test re-initialization after cleanup
    ASSERT_EQ(VSLA_SUCCESS, vsla_init());
    ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    return 1;
}

static int test_version_info(void) {
    const char* version = vsla_version();
    
    // Version string should not be NULL
    ASSERT_NOT_NULL(version);
    
    // Version string should not be empty
    ASSERT_TRUE(strlen(version) > 0);
    
    // Version should contain at least a number (basic sanity check)
    int has_digit = 0;
    for (const char* p = version; *p; p++) {
        if (*p >= '0' && *p <= '9') {
            has_digit = 1;
            break;
        }
    }
    ASSERT_TRUE(has_digit);
    return 1;
}

static int test_fftw_detection(void) {
    // Test FFTW detection (should return 0 or 1)
    int has_fftw = vsla_has_fftw();
    ASSERT_TRUE(has_fftw == 0 || has_fftw == 1);
    
    // The result should be consistent between calls
    int has_fftw2 = vsla_has_fftw();
    ASSERT_EQ(has_fftw, has_fftw2);
    return 1;
}

static int test_error_strings(void) {
    // Test all defined error codes have valid strings
    const char* str;
    
    str = vsla_error_string(VSLA_SUCCESS);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_NULL_POINTER);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_INVALID_SHAPE);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_MEMORY);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_NOT_IMPLEMENTED);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_IO);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_INCOMPATIBLE_MODELS);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_INDEX_OUT_OF_BOUNDS);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    str = vsla_error_string(VSLA_ERROR_INVALID_DTYPE);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    
    // Test invalid error code
    str = vsla_error_string((vsla_error_t)999);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
    return 1;
}

static int test_dtype_sizes(void) {
    // Test all defined data types have valid sizes
    size_t size;
    
    size = vsla_dtype_size(VSLA_DTYPE_F32);
    ASSERT_EQ(4, size);  // float is 4 bytes
    
    size = vsla_dtype_size(VSLA_DTYPE_F64);
    ASSERT_EQ(8, size);  // double is 8 bytes
    
    size = vsla_dtype_size(VSLA_DTYPE_I32);
    ASSERT_EQ(4, size);  // int32_t is 4 bytes
    
    size = vsla_dtype_size(VSLA_DTYPE_I64);
    ASSERT_EQ(8, size);  // int64_t is 8 bytes
    
    size = vsla_dtype_size(VSLA_DTYPE_U32);
    ASSERT_EQ(4, size);  // uint32_t is 4 bytes
    
    size = vsla_dtype_size(VSLA_DTYPE_U64);
    ASSERT_EQ(8, size);  // uint64_t is 8 bytes
    
    // Test invalid dtype
    size = vsla_dtype_size((vsla_dtype_t)999);
    ASSERT_EQ(0, size);  // Should return 0 for invalid types
    return 1;
}

static int test_power_of_two_utilities(void) {
    // Test is_pow2 function
    ASSERT_TRUE(vsla_is_pow2(1));    // 2^0
    ASSERT_TRUE(vsla_is_pow2(2));    // 2^1
    ASSERT_TRUE(vsla_is_pow2(4));    // 2^2
    ASSERT_TRUE(vsla_is_pow2(8));    // 2^3
    ASSERT_TRUE(vsla_is_pow2(16));   // 2^4
    ASSERT_TRUE(vsla_is_pow2(32));   // 2^5
    ASSERT_TRUE(vsla_is_pow2(64));   // 2^6
    ASSERT_TRUE(vsla_is_pow2(128));  // 2^7
    ASSERT_TRUE(vsla_is_pow2(256));  // 2^8
    ASSERT_TRUE(vsla_is_pow2(512));  // 2^9
    ASSERT_TRUE(vsla_is_pow2(1024)); // 2^10
    
    // Test non-powers of two
    ASSERT_FALSE(vsla_is_pow2(0));
    ASSERT_FALSE(vsla_is_pow2(3));
    ASSERT_FALSE(vsla_is_pow2(5));
    ASSERT_FALSE(vsla_is_pow2(6));
    ASSERT_FALSE(vsla_is_pow2(7));
    ASSERT_FALSE(vsla_is_pow2(9));
    ASSERT_FALSE(vsla_is_pow2(10));
    ASSERT_FALSE(vsla_is_pow2(15));
    ASSERT_FALSE(vsla_is_pow2(17));
    ASSERT_FALSE(vsla_is_pow2(100));
    ASSERT_FALSE(vsla_is_pow2(1000));
    
    // Test next_pow2 function
    ASSERT_EQ(1, vsla_next_pow2(0));
    ASSERT_EQ(1, vsla_next_pow2(1));
    ASSERT_EQ(2, vsla_next_pow2(2));
    ASSERT_EQ(4, vsla_next_pow2(3));
    ASSERT_EQ(4, vsla_next_pow2(4));
    ASSERT_EQ(8, vsla_next_pow2(5));
    ASSERT_EQ(8, vsla_next_pow2(6));
    ASSERT_EQ(8, vsla_next_pow2(7));
    ASSERT_EQ(8, vsla_next_pow2(8));
    ASSERT_EQ(16, vsla_next_pow2(9));
    ASSERT_EQ(16, vsla_next_pow2(10));
    ASSERT_EQ(16, vsla_next_pow2(15));
    ASSERT_EQ(16, vsla_next_pow2(16));
    ASSERT_EQ(32, vsla_next_pow2(17));
    ASSERT_EQ(128, vsla_next_pow2(100));
    ASSERT_EQ(1024, vsla_next_pow2(1000));
    
    // Test some larger values
    ASSERT_EQ(2048, vsla_next_pow2(1025));
    ASSERT_EQ(4096, vsla_next_pow2(4096));
    ASSERT_EQ(8192, vsla_next_pow2(4097));
    return 1;
}

static int test_library_state_consistency(void) {
    // Test that library functions work correctly across init/cleanup cycles
    
    // Initialize
    ASSERT_EQ(VSLA_SUCCESS, vsla_init());
    
    // Test basic operations work
    uint64_t shape[] = {3};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Fill tensor with test data
    uint64_t idx;
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        ASSERT_EQ(VSLA_SUCCESS, vsla_set_f64(tensor, &idx, (double)(i + 1)));
    }
    
    // Verify data
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double val = vsla_get_f64(tensor, &idx);
        ASSERT_FLOAT_EQ((double)(i + 1), val, 1e-12);
    }
    
    vsla_free(tensor);
    
    // Cleanup
    ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    
    // Re-initialize and test again
    ASSERT_EQ(VSLA_SUCCESS, vsla_init());
    
    tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);
    
    // Test operations still work
    ASSERT_EQ(VSLA_SUCCESS, vsla_fill(tensor, 42.0));
    
    for (size_t i = 0; i < 3; i++) {
        idx = i;
        double val = vsla_get_f64(tensor, &idx);
        ASSERT_FLOAT_EQ(42.0, val, 1e-12);
    }
    
    vsla_free(tensor);
    ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    return 1;
}

static int test_utility_functions_thread_safety(void) {
    // Test that utility functions are thread-safe (basic test)
    // Note: This is a simple test - real thread safety would require pthread testing
    
    // These functions should be safe to call multiple times concurrently
    for (int i = 0; i < 100; i++) {
        ASSERT_NOT_NULL(vsla_version());
        ASSERT_TRUE(vsla_has_fftw() >= 0);
        ASSERT_NOT_NULL(vsla_error_string(VSLA_SUCCESS));
        ASSERT_TRUE(vsla_dtype_size(VSLA_DTYPE_F64) == 8);
        ASSERT_TRUE(vsla_is_pow2(16) == 1);
        ASSERT_TRUE(vsla_next_pow2(100) == 128);
    }
    return 1;
}

static int test_edge_cases(void) {
    // Test edge cases and boundary conditions
    
    // Test very large power of 2 values
    ASSERT_TRUE(vsla_is_pow2(1ULL << 32));
    ASSERT_TRUE(vsla_is_pow2(1ULL << 40));
    
    // Test next_pow2 with large values (within reason to avoid overflow)
    uint64_t large_val = 1ULL << 30;
    uint64_t next = vsla_next_pow2(large_val + 1);
    ASSERT_EQ(1ULL << 31, next);
    
    // Test next_pow2 near overflow (implementation dependent)
    uint64_t very_large = (1ULL << 62) + 1;
    next = vsla_next_pow2(very_large);
    // Should either return 1ULL << 63 or handle overflow gracefully
    ASSERT_TRUE(next >= very_large);
    return 1;
}

static int test_multiple_init_cleanup_cycles(void) {
    // Test multiple init/cleanup cycles work correctly
    for (int cycle = 0; cycle < 5; cycle++) {
        ASSERT_EQ(VSLA_SUCCESS, vsla_init());
        
        // Test basic functionality in each cycle
        const char* version = vsla_version();
        ASSERT_NOT_NULL(version);
        ASSERT_TRUE(strlen(version) > 0);
        
        // Test that dtypes work
        ASSERT_EQ(8, vsla_dtype_size(VSLA_DTYPE_F64));
        
        // Test power utilities
        ASSERT_TRUE(vsla_is_pow2(64));
        ASSERT_EQ(16, vsla_next_pow2(15));
        
        ASSERT_EQ(VSLA_SUCCESS, vsla_cleanup());
    }
    return 1;
}

static void run_utils_tests(void) {
    TEST_CASE("Library Initialization", test_library_initialization);
    TEST_CASE("Version Information", test_version_info);
    TEST_CASE("FFTW Detection", test_fftw_detection);
    TEST_CASE("Error Strings", test_error_strings);
    TEST_CASE("Data Type Sizes", test_dtype_sizes);
    TEST_CASE("Power of Two Utilities", test_power_of_two_utilities);
    TEST_CASE("Library State Consistency", test_library_state_consistency);
    TEST_CASE("Utility Thread Safety", test_utility_functions_thread_safety);
    TEST_CASE("Edge Cases", test_edge_cases);
    TEST_CASE("Multiple Init/Cleanup Cycles", test_multiple_init_cleanup_cycles);
}

static const test_suite_t utils_suite = {
    .name = "utils",
    .setup = utils_test_setup,
    .teardown = utils_test_teardown,
    .run_tests = run_utils_tests
};

void register_utils_tests(void) {
    register_test_suite(&utils_suite);
}