/**
 * @file test_stacking_comprehensive.c
 * @brief Comprehensive tests for VSLA stacking operations (Section 5)
 * 
 * Tests all stacking functionality for mathematical correctness:
 * - Basic stacking with ambient promotion
 * - Window stacking with ring buffers  
 * - Pyramid stacking with hierarchical processing
 * - Mathematical properties and edge cases
 */

#include "vsla/vsla_unified.h"
#include "vsla/vsla_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-10

static int tests_passed = 0;
static int tests_failed = 0;

void report_test(const char* test_name, int passed) {
    if (passed) {
        printf("✅ %s\n", test_name);
        tests_passed++;
    } else {
        printf("❌ %s\n", test_name);
        tests_failed++;
    }
}

void test_basic_stacking() {
    printf("\n=== Testing Basic Stacking (Section 5.1) ===\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        report_test("Context creation", 0);
        return;
    }
    
    // Test 1: Stack homogeneous tensors
    uint64_t shape[] = {2};
    vsla_tensor_t* t1 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t2 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t3 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill tensors with test data
    uint64_t idx0[] = {0}, idx1[] = {1};
    vsla_set_f64(ctx, t1, idx0, 1.0); vsla_set_f64(ctx, t1, idx1, 2.0);
    vsla_set_f64(ctx, t2, idx0, 3.0); vsla_set_f64(ctx, t2, idx1, 4.0);
    vsla_set_f64(ctx, t3, idx0, 5.0); vsla_set_f64(ctx, t3, idx1, 6.0);
    
    const vsla_tensor_t* tensors[] = {t1, t2, t3};
    
    // Create output tensor for stacking (should be rank 2, shape [3, 2])
    uint64_t out_shape[] = {3, 2};
    vsla_tensor_t* stacked = vsla_tensor_create(ctx, 2, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    vsla_error_t err = vsla_stack(ctx, stacked, tensors, 3);
    report_test("Basic stacking operation", err == VSLA_SUCCESS);
    
    // Verify output shape
    uint64_t actual_shape[2];
    vsla_get_shape(stacked, actual_shape);
    report_test("Stacked shape [3,2]", actual_shape[0] == 3 && actual_shape[1] == 2);
    
    // Verify stacked values
    double val;
    uint64_t idx00[] = {0, 0}, idx01[] = {0, 1};
    uint64_t idx10[] = {1, 0}, idx11[] = {1, 1};
    uint64_t idx20[] = {2, 0}, idx21[] = {2, 1};
    
    vsla_get_f64(ctx, stacked, idx00, &val);
    int val00_correct = fabs(val - 1.0) < EPSILON;
    vsla_get_f64(ctx, stacked, idx01, &val);
    int val01_correct = fabs(val - 2.0) < EPSILON;
    vsla_get_f64(ctx, stacked, idx10, &val);
    int val10_correct = fabs(val - 3.0) < EPSILON;
    vsla_get_f64(ctx, stacked, idx11, &val);
    int val11_correct = fabs(val - 4.0) < EPSILON;
    vsla_get_f64(ctx, stacked, idx20, &val);
    int val20_correct = fabs(val - 5.0) < EPSILON;
    vsla_get_f64(ctx, stacked, idx21, &val);
    int val21_correct = fabs(val - 6.0) < EPSILON;
    
    report_test("Stacked values correct", 
                val00_correct && val01_correct && val10_correct && 
                val11_correct && val20_correct && val21_correct);
    
    vsla_tensor_free(t1);
    vsla_tensor_free(t2);
    vsla_tensor_free(t3);
    vsla_tensor_free(stacked);
    vsla_cleanup(ctx);
}

void test_heterogeneous_stacking() {
    printf("\n=== Testing Heterogeneous Stacking (Ambient Promotion) ===\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        report_test("Context creation", 0);
        return;
    }
    
    // Test ambient promotion: shapes [2], [3], [1] -> output [3, 3]
    uint64_t shape1[] = {2}, shape2[] = {3}, shape3[] = {1};
    vsla_tensor_t* t1 = vsla_tensor_create(ctx, 1, shape1, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t2 = vsla_tensor_create(ctx, 1, shape2, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t3 = vsla_tensor_create(ctx, 1, shape3, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with test data
    uint64_t idx[] = {0};
    vsla_set_f64(ctx, t1, idx, 10.0);
    idx[0] = 1; vsla_set_f64(ctx, t1, idx, 20.0);
    
    idx[0] = 0; vsla_set_f64(ctx, t2, idx, 30.0);
    idx[0] = 1; vsla_set_f64(ctx, t2, idx, 40.0);
    idx[0] = 2; vsla_set_f64(ctx, t2, idx, 50.0);
    
    idx[0] = 0; vsla_set_f64(ctx, t3, idx, 60.0);
    
    const vsla_tensor_t* tensors[] = {t1, t2, t3};
    
    // Create output with ambient shape [3, 3] (k=3, max_shape=3)
    uint64_t out_shape[] = {3, 3};
    vsla_tensor_t* stacked = vsla_tensor_create(ctx, 2, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    vsla_error_t err = vsla_stack(ctx, stacked, tensors, 3);
    report_test("Heterogeneous stacking operation", err == VSLA_SUCCESS);
    
    // Verify ambient promotion: t1[2,0,0], t2[3], t3[1,0,0]
    double val;
    uint64_t test_idx[] = {0, 0};
    
    // t1 slice: [10, 20, 0]
    test_idx[0] = 0; test_idx[1] = 0; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t1_0_correct = fabs(val - 10.0) < EPSILON;
    test_idx[1] = 1; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t1_1_correct = fabs(val - 20.0) < EPSILON;
    test_idx[1] = 2; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t1_2_correct = fabs(val - 0.0) < EPSILON;  // Zero extension
    
    // t2 slice: [30, 40, 50]
    test_idx[0] = 1; test_idx[1] = 0; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t2_0_correct = fabs(val - 30.0) < EPSILON;
    test_idx[1] = 1; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t2_1_correct = fabs(val - 40.0) < EPSILON;
    test_idx[1] = 2; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t2_2_correct = fabs(val - 50.0) < EPSILON;
    
    // t3 slice: [60, 0, 0]
    test_idx[0] = 2; test_idx[1] = 0; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t3_0_correct = fabs(val - 60.0) < EPSILON;
    test_idx[1] = 1; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t3_1_correct = fabs(val - 0.0) < EPSILON;  // Zero extension
    test_idx[1] = 2; vsla_get_f64(ctx, stacked, test_idx, &val);
    int t3_2_correct = fabs(val - 0.0) < EPSILON;  // Zero extension
    
    report_test("Ambient promotion values correct",
                t1_0_correct && t1_1_correct && t1_2_correct &&
                t2_0_correct && t2_1_correct && t2_2_correct &&
                t3_0_correct && t3_1_correct && t3_2_correct);
    
    vsla_tensor_free(t1);
    vsla_tensor_free(t2);
    vsla_tensor_free(t3);
    vsla_tensor_free(stacked);
    vsla_cleanup(ctx);
}

void test_window_stacking() {
    printf("\n=== Testing Window Stacking (Section 5.2) ===\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        report_test("Context creation", 0);
        return;
    }
    
    // Create window with size 3 for rank-1 F64 tensors
    vsla_window_t* window = vsla_window_create(ctx, 3, 1, VSLA_DTYPE_F64);
    report_test("Window creation", window != NULL);
    
    // Create test tensors
    uint64_t shape[] = {2};
    vsla_tensor_t* t1 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t2 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t3 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* t4 = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill tensors
    uint64_t idx[] = {0};
    vsla_set_f64(ctx, t1, idx, 1.0); idx[0] = 1; vsla_set_f64(ctx, t1, idx, 2.0);
    idx[0] = 0; vsla_set_f64(ctx, t2, idx, 3.0); idx[0] = 1; vsla_set_f64(ctx, t2, idx, 4.0);
    idx[0] = 0; vsla_set_f64(ctx, t3, idx, 5.0); idx[0] = 1; vsla_set_f64(ctx, t3, idx, 6.0);
    idx[0] = 0; vsla_set_f64(ctx, t4, idx, 7.0); idx[0] = 1; vsla_set_f64(ctx, t4, idx, 8.0);
    
    // Push first two tensors (window not full)
    vsla_tensor_t* result1 = vsla_window_push(window, t1);
    vsla_tensor_t* result2 = vsla_window_push(window, t2);
    report_test("Window not full returns NULL", result1 == NULL && result2 == NULL);
    
    // Push third tensor (window full, should return stacked result)
    vsla_tensor_t* result3 = vsla_window_push(window, t3);
    report_test("Window full returns stacked tensor", result3 != NULL);
    
    if (result3) {
        // Verify stacked result has shape [3, 2]
        uint64_t out_shape[2];
        vsla_get_shape(result3, out_shape);
        report_test("Window output shape [3,2]", out_shape[0] == 3 && out_shape[1] == 2);
        
        // Verify values from stacked window
        double val;
        uint64_t test_idx[] = {0, 0};
        vsla_get_f64(ctx, result3, test_idx, &val);
        int first_val_correct = fabs(val - 1.0) < EPSILON;
        
        test_idx[0] = 2; test_idx[1] = 1;
        vsla_get_f64(ctx, result3, test_idx, &val);
        int last_val_correct = fabs(val - 6.0) < EPSILON;
        
        report_test("Window stacked values correct", first_val_correct && last_val_correct);
        
        vsla_tensor_free(result3);
    }
    
    // Push fourth tensor (new window cycle)
    vsla_tensor_t* result4 = vsla_window_push(window, t4);
    report_test("New window cycle returns NULL", result4 == NULL);
    
    vsla_window_destroy(window);
    vsla_tensor_free(t1);
    vsla_tensor_free(t2);
    vsla_tensor_free(t3);
    vsla_tensor_free(t4);
    vsla_cleanup(ctx);
}

void test_pyramid_stacking() {
    printf("\n=== Testing Pyramid Stacking (Section 5.2) ===\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        report_test("Context creation", 0);
        return;
    }
    
    // Create 2-level pyramid with window size 2
    vsla_pyramid_t* pyramid = vsla_pyramid_create(ctx, 2, 2, 1, VSLA_DTYPE_F64, false);
    report_test("Pyramid creation", pyramid != NULL);
    
    // Create test tensors
    uint64_t shape[] = {1};
    vsla_tensor_t* tensors[8];
    for (int i = 0; i < 8; i++) {
        tensors[i] = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
        uint64_t idx[] = {0};
        vsla_set_f64(ctx, tensors[i], idx, (double)(i + 1));
    }
    
    // Feed tensors through pyramid
    int results_received = 0;
    vsla_tensor_t* final_results[4];
    
    for (int i = 0; i < 8; i++) {
        vsla_tensor_t* result = vsla_pyramid_push(pyramid, tensors[i]);
        if (result) {
            final_results[results_received++] = result;
            
            // Verify result properties
            uint8_t result_rank = vsla_get_rank(result);
            report_test("Pyramid result has correct rank", result_rank == 3); // rank 1 -> 2 -> 3
            
            if (result_rank == 3) {
                uint64_t result_shape[3];
                vsla_get_shape(result, result_shape);
                report_test("Final pyramid shape [2,2,1]", 
                           result_shape[0] == 2 && result_shape[1] == 2 && result_shape[2] == 1);
            }
        }
    }
    
    report_test("Pyramid produces expected number of results", results_received > 0);
    
    // Test flushing
    size_t flush_count;
    vsla_tensor_t** flushed = vsla_pyramid_flush(pyramid, &flush_count);
    report_test("Pyramid flush operation", flushed != NULL || flush_count == 0);
    
    if (flushed) {
        for (size_t i = 0; i < flush_count; i++) {
            vsla_tensor_free(flushed[i]);
        }
        free(flushed);
    }
    
    for (int i = 0; i < results_received; i++) {
        vsla_tensor_free(final_results[i]);
    }
    
    vsla_pyramid_destroy(pyramid);
    for (int i = 0; i < 8; i++) {
        vsla_tensor_free(tensors[i]);
    }
    vsla_cleanup(ctx);
}

void test_empty_tensor_stacking() {
    printf("\n=== Testing Empty Tensor Stacking ===\n");
    
    vsla_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        report_test("Context creation", 0);
        return;
    }
    
    // Test stacking with empty tensor (shape [0])
    uint64_t empty_shape[] = {0};
    uint64_t normal_shape[] = {2};
    
    vsla_tensor_t* empty_tensor = vsla_tensor_create(ctx, 1, empty_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* normal_tensor = vsla_tensor_create(ctx, 1, normal_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    uint64_t idx[] = {0};
    vsla_set_f64(ctx, normal_tensor, idx, 42.0);
    idx[0] = 1; vsla_set_f64(ctx, normal_tensor, idx, 84.0);
    
    const vsla_tensor_t* tensors[] = {empty_tensor, normal_tensor};
    
    // Output should have ambient shape [2, 2]
    uint64_t out_shape[] = {2, 2};
    vsla_tensor_t* stacked = vsla_tensor_create(ctx, 2, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    vsla_error_t err = vsla_stack(ctx, stacked, tensors, 2);
    report_test("Empty tensor stacking", err == VSLA_SUCCESS);
    
    // Verify empty slice is zeros, normal slice has values
    double val;
    uint64_t test_idx[] = {0, 0};
    vsla_get_f64(ctx, stacked, test_idx, &val);
    int empty_zero = fabs(val - 0.0) < EPSILON;
    
    test_idx[0] = 1; test_idx[1] = 0;
    vsla_get_f64(ctx, stacked, test_idx, &val);
    int normal_value = fabs(val - 42.0) < EPSILON;
    
    report_test("Empty tensor produces zeros in stack", empty_zero && normal_value);
    
    vsla_tensor_free(empty_tensor);
    vsla_tensor_free(normal_tensor);
    vsla_tensor_free(stacked);
    vsla_cleanup(ctx);
}

int main() {
    printf("🏗️  VSLA Stacking Operations Test Suite\n");
    printf("=======================================\n");
    printf("Testing Section 5: Structural Operators\n\n");
    
    test_basic_stacking();
    test_heterogeneous_stacking(); 
    test_window_stacking();
    test_pyramid_stacking();
    test_empty_tensor_stacking();
    
    printf("\n=======================================\n");
    printf("📊 Stacking Test Summary: %d/%d tests passed\n", 
           tests_passed, tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("✅ All stacking tests passed! Section 5 implementation complete.\n");
        return 0;
    } else {
        printf("❌ %d tests failed. Section 5 needs attention.\n", tests_failed);
        return 1;
    }
}