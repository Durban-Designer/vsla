/**
 * @file test_utils.c
 * @brief Tests for utility functions
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"

static int test_library_initialization(void) {
    vsla_context_t* ctx = vsla_init(NULL);
    ASSERT_NOT_NULL(ctx);
    vsla_cleanup(ctx);
    return 1;
}

static int test_version_info(void) {
    const char* version = vsla_version();
    ASSERT_NOT_NULL(version);
    ASSERT(strlen(version) > 0);
    return 1;
}

static void run_utils_tests(void) {
    TEST_CASE("Library Initialization and Cleanup", test_library_initialization);
    TEST_CASE("Version Information", test_version_info);
}

static const test_suite_t utils_suite = {
    .name = "utils",
    .setup = NULL,
    .teardown = NULL,
    .run_tests = run_utils_tests
};

void register_utils_tests(void) {
    register_test_suite(&utils_suite);
}