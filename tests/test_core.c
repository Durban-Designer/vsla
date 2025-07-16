/**
 * @file test_core.c
 * @brief Tests for core utility functions
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla_core.h"

DECLARE_TEST(error_string_test) {
    ASSERT_STR_EQ(vsla_error_string(VSLA_SUCCESS), "Success");
    ASSERT_STR_EQ(vsla_error_string(VSLA_ERROR_NULL_POINTER), "Null pointer passed where not allowed");
    ASSERT_STR_EQ(vsla_error_string(VSLA_ERROR_MEMORY), "Memory allocation failed");
    ASSERT_STR_EQ(vsla_error_string((vsla_error_t)999), "Unknown error");
    return 1;
}

DECLARE_TEST(dtype_size_test) {
    ASSERT_EQ(vsla_dtype_size(VSLA_DTYPE_F64), sizeof(double));
    ASSERT_EQ(vsla_dtype_size(VSLA_DTYPE_F32), sizeof(float));
    ASSERT_EQ(vsla_dtype_size((vsla_dtype_t)999), 0);
    return 1;
}

DECLARE_TEST(next_pow2_test) {
    ASSERT_EQ(vsla_next_pow2(0), 1);
    ASSERT_EQ(vsla_next_pow2(1), 1);
    ASSERT_EQ(vsla_next_pow2(2), 2);
    ASSERT_EQ(vsla_next_pow2(3), 4);
    ASSERT_EQ(vsla_next_pow2(7), 8);
    ASSERT_EQ(vsla_next_pow2(8), 8);
    ASSERT_EQ(vsla_next_pow2(9), 16);
    ASSERT_EQ(vsla_next_pow2(1023), 1024);
    ASSERT_EQ(vsla_next_pow2(1024), 1024);
    
    /* Test overflow */
    ASSERT_EQ(vsla_next_pow2(UINT64_MAX), 0);
    ASSERT_EQ(vsla_next_pow2((UINT64_MAX >> 1) + 1), 0);
    
    return 1;
}

DECLARE_TEST(is_pow2_test) {
    ASSERT_FALSE(vsla_is_pow2(0));
    ASSERT_TRUE(vsla_is_pow2(1));
    ASSERT_TRUE(vsla_is_pow2(2));
    ASSERT_FALSE(vsla_is_pow2(3));
    ASSERT_TRUE(vsla_is_pow2(4));
    ASSERT_FALSE(vsla_is_pow2(5));
    ASSERT_FALSE(vsla_is_pow2(6));
    ASSERT_FALSE(vsla_is_pow2(7));
    ASSERT_TRUE(vsla_is_pow2(8));
    ASSERT_TRUE(vsla_is_pow2(1024));
    ASSERT_FALSE(vsla_is_pow2(1023));
    return 1;
}

static void core_test_setup(void) {
    /* Setup for core tests */
}

static void core_test_teardown(void) {
    /* Teardown for core tests */
}

static void run_core_tests(void) {
    RUN_TEST(error_string_test);
    RUN_TEST(dtype_size_test);
    RUN_TEST(next_pow2_test);
    RUN_TEST(is_pow2_test);
}

static const test_suite_t core_suite = {
    .name = "core",
    .setup = core_test_setup,
    .teardown = core_test_teardown,
    .run_tests = run_core_tests
};

void register_core_tests(void) {
    register_test_suite(&core_suite);
}