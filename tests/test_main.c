/**
 * @file test_main.c
 * @brief Test suite registration
 * 
 * @copyright MIT License
 */

#include "test_framework.h"

/* Declare test registration functions */
extern void register_core_tests(void);
extern void register_tensor_tests(void);
extern void register_ops_tests(void);
extern void register_io_tests(void);
extern void register_conv_tests(void);
extern void register_kron_tests(void);
extern void register_autograd_tests(void);
extern void register_utils_tests(void);

void register_all_test_suites(void) {
    register_core_tests();
    register_tensor_tests();
    register_ops_tests();
    register_io_tests();
    register_conv_tests();
    register_kron_tests();
    register_autograd_tests();
    register_utils_tests();
}