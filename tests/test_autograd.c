/**
 * @file test_autograd.c
 * @brief Tests for automatic differentiation
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <math.h>

// Test tape creation and destruction
static int test_tape_creation(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    vsla_tape_free(tape);
    return 1;
}

// Test basic operation recording
static int test_operation_recording(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    // Create test tensors
    uint64_t shape[] = {2};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !c) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
        return 0;
    }
    
    // Record an addition operation
    vsla_tensor_t* inputs[] = {a, b};
    vsla_error_t err = vsla_tape_record(tape, VSLA_OP_ADD, inputs, 2, c, NULL, 0);
    
    if (err != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
        return 0;
    }
    
    // Check that operation was recorded
    if (tape->num_ops != 1) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
        return 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
    return 1;
}

// Test gradient setting and getting
static int test_gradient_management(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    uint64_t shape[] = {2};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* grad = vsla_ones(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!tensor || !grad) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    // Set gradient
    vsla_error_t err = vsla_set_gradient(tape, tensor, grad);
    if (err != VSLA_SUCCESS) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    // Get gradient back
    vsla_tensor_t* retrieved_grad = vsla_get_gradient(tape, tensor);
    if (!retrieved_grad) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    // Verify gradient values
    for (int i = 0; i < 2; i++) {
        double val;
        uint64_t idx = i;
        if (vsla_get_f64(retrieved_grad, &idx, &val) != VSLA_SUCCESS ||
            fabs(val - 1.0) > 1e-15) {
            vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
            return 0;
        }
    }
    
    vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
    return 1;
}

// Test gradient clearing
static int test_gradient_clearing(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    uint64_t shape[] = {2};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* grad = vsla_ones(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!tensor || !grad) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    // Set gradient
    vsla_set_gradient(tape, tensor, grad);
    
    // Clear gradients
    vsla_error_t err = vsla_clear_gradients(tape);
    if (err != VSLA_SUCCESS) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    // Gradient should now be NULL
    vsla_tensor_t* retrieved_grad = vsla_get_gradient(tape, tensor);
    if (retrieved_grad != NULL) {
        vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
        return 0;
    }
    
    vsla_free(tensor); vsla_free(grad); vsla_tape_free(tape);
    return 1;
}

// Test simple addition backward pass
static int test_addition_backward(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    uint64_t shape[] = {2};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !c) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
        return 0;
    }
    
    // Set values: a = [1, 2], b = [3, 4]
    uint64_t idx0 = 0, idx1 = 1;
    vsla_set_f64(a, &idx0, 1.0); vsla_set_f64(a, &idx1, 2.0);
    vsla_set_f64(b, &idx0, 3.0); vsla_set_f64(b, &idx1, 4.0);
    
    // Compute c = a + b
    vsla_add(c, a, b);
    
    // Record operation
    vsla_tensor_t* inputs[] = {a, b};
    vsla_tape_record(tape, VSLA_OP_ADD, inputs, 2, c, NULL, 0);
    
    // Set output gradient to ones (like loss gradient)
    vsla_tensor_t* grad_c = vsla_ones(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!grad_c) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_tape_free(tape);
        return 0;
    }
    
    vsla_set_gradient(tape, c, grad_c);
    
    // Perform backward pass
    vsla_error_t err = vsla_backward(tape);
    if (err != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_free(grad_c); vsla_tape_free(tape);
        return 0;
    }
    
    // Check gradients (should both be [1, 1])
    vsla_tensor_t* grad_a = vsla_get_gradient(tape, a);
    vsla_tensor_t* grad_b = vsla_get_gradient(tape, b);
    
    int success = 1;
    if (grad_a) {
        for (int i = 0; i < 2; i++) {
            double val;
            uint64_t idx = i;
            if (vsla_get_f64(grad_a, &idx, &val) != VSLA_SUCCESS ||
                fabs(val - 1.0) > 1e-14) {
                success = 0;
                break;
            }
        }
    } else {
        success = 0;
    }
    
    if (grad_b && success) {
        for (int i = 0; i < 2; i++) {
            double val;
            uint64_t idx = i;
            if (vsla_get_f64(grad_b, &idx, &val) != VSLA_SUCCESS ||
                fabs(val - 1.0) > 1e-14) {
                success = 0;
                break;
            }
        }
    } else {
        success = 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(c); vsla_free(grad_c); vsla_tape_free(tape);
    return success;
}

// Test scaling backward pass
static int test_scaling_backward(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    uint64_t shape[] = {2};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b) {
        vsla_free(a); vsla_free(b); vsla_tape_free(tape);
        return 0;
    }
    
    // Set values: a = [2, 3]
    uint64_t idx0 = 0, idx1 = 1;
    vsla_set_f64(a, &idx0, 2.0); vsla_set_f64(a, &idx1, 3.0);
    
    // Compute b = 5 * a
    double scalar = 5.0;
    vsla_scale(b, a, scalar);
    
    // Record operation
    vsla_tensor_t* inputs[] = {a};
    vsla_tape_record(tape, VSLA_OP_SCALE, inputs, 1, b, &scalar, sizeof(double));
    
    // Set output gradient
    vsla_tensor_t* grad_b = vsla_ones(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!grad_b) {
        vsla_free(a); vsla_free(b); vsla_tape_free(tape);
        return 0;
    }
    
    vsla_set_gradient(tape, b, grad_b);
    
    // Perform backward pass
    vsla_error_t err = vsla_backward(tape);
    if (err != VSLA_SUCCESS) {
        vsla_free(a); vsla_free(b); vsla_free(grad_b); vsla_tape_free(tape);
        return 0;
    }
    
    // Check gradient (should be [5, 5])
    vsla_tensor_t* grad_a = vsla_get_gradient(tape, a);
    
    int success = 1;
    if (grad_a) {
        for (int i = 0; i < 2; i++) {
            double val;
            uint64_t idx = i;
            if (vsla_get_f64(grad_a, &idx, &val) != VSLA_SUCCESS ||
                fabs(val - 5.0) > 1e-14) {
                success = 0;
                break;
            }
        }
    } else {
        success = 0;
    }
    
    vsla_free(a); vsla_free(b); vsla_free(grad_b); vsla_tape_free(tape);
    return success;
}

// Test error handling
static int test_autograd_error_handling(void) {
    // Test NULL pointer errors
    if (vsla_tape_new() == NULL) return 0;  // Should succeed
    
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    // Test recording with NULL output
    if (vsla_tape_record(tape, VSLA_OP_ADD, NULL, 0, NULL, NULL, 0) == VSLA_SUCCESS) {
        vsla_tape_free(tape);
        return 0;  // Should fail
    }
    
    // Test gradient operations with NULL tape
    if (vsla_get_gradient(NULL, NULL) != NULL) {
        vsla_tape_free(tape);
        return 0;  // Should return NULL
    }
    
    if (vsla_set_gradient(NULL, NULL, NULL) == VSLA_SUCCESS) {
        vsla_tape_free(tape);
        return 0;  // Should fail
    }
    
    vsla_tape_free(tape);
    return 1;
}

// Test multiple operations
static int test_multiple_operations(void) {
    vsla_tape_t* tape = vsla_tape_new();
    if (!tape) return 0;
    
    uint64_t shape[] = {2};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* d = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    if (!a || !b || !c || !d) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_free(d); vsla_tape_free(tape);
        return 0;
    }
    
    // Set values
    uint64_t idx0 = 0, idx1 = 1;
    vsla_set_f64(a, &idx0, 1.0); vsla_set_f64(a, &idx1, 2.0);
    vsla_set_f64(b, &idx0, 3.0); vsla_set_f64(b, &idx1, 4.0);
    
    // Compute c = a + b
    vsla_add(c, a, b);
    vsla_tensor_t* inputs1[] = {a, b};
    vsla_tape_record(tape, VSLA_OP_ADD, inputs1, 2, c, NULL, 0);
    
    // Compute d = c - a
    vsla_sub(d, c, a);
    vsla_tensor_t* inputs2[] = {c, a};
    vsla_tape_record(tape, VSLA_OP_SUB, inputs2, 2, d, NULL, 0);
    
    // Set output gradient
    vsla_tensor_t* grad_d = vsla_ones(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!grad_d) {
        vsla_free(a); vsla_free(b); vsla_free(c); vsla_free(d); vsla_tape_free(tape);
        return 0;
    }
    
    vsla_set_gradient(tape, d, grad_d);
    
    // Perform backward pass
    vsla_backward(tape);
    
    // Check that we have 2 operations recorded
    int success = (tape->num_ops == 2);
    
    vsla_free(a); vsla_free(b); vsla_free(c); vsla_free(d); vsla_free(grad_d); vsla_tape_free(tape);
    return success;
}

static void autograd_test_setup(void) {
    // Setup for autograd tests
}

static void autograd_test_teardown(void) {
    // Teardown for autograd tests
}

static void run_autograd_tests(void) {
    printf("Running Autograd tests:\n");
    
    RUN_TEST(test_tape_creation);
    RUN_TEST(test_operation_recording);
    RUN_TEST(test_gradient_management);
    RUN_TEST(test_gradient_clearing);
    RUN_TEST(test_addition_backward);
    RUN_TEST(test_scaling_backward);
    RUN_TEST(test_autograd_error_handling);
    RUN_TEST(test_multiple_operations);
}

static const test_suite_t autograd_suite = {
    .name = "autograd",
    .setup = autograd_test_setup,
    .teardown = autograd_test_teardown,
    .run_tests = run_autograd_tests
};

void register_autograd_tests(void) {
    register_test_suite(&autograd_suite);
}