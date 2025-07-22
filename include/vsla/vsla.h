/**
 * @file vsla.h
 * @brief Variable-Shape Linear Algebra (VSLA) Library - Production-Ready Public API
 * 
 * This is the single, complete public API for VSLA. All users should include only
 * this header. The library provides:
 * 
 * - Context-based resource management
 * - Unified tensor operations across CPU/GPU backends
 * - Mathematical correctness with ambient promotion semantics
 * - Variable-shape linear algebra operations
 * - High-performance stacking and structural operators
 * 
 * @version 3.1
 * @copyright MIT License
 * @author VSLA Development Team
 */

#ifndef VSLA_H
#define VSLA_H

#ifdef __cplusplus
extern "C" {
#endif

/* === Core Types and Constants === */
#include "vsla_core.h"

/* === Unified API - Single Point of Control === */
/* This is the ONLY interface users should use */
#include "vsla_unified.h"

/* Note: Version constants are defined in vsla_core.h */

/**
 * @brief Quick start example
 * 
 * @code
 * // Initialize VSLA context
 * vsla_context_t* ctx = vsla_init(NULL);
 * 
 * // Create tensors
 * uint64_t shape[] = {3, 2};
 * vsla_tensor_t* a = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
 * vsla_tensor_t* b = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
 * vsla_tensor_t* result = vsla_tensor_create(ctx, 2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
 * 
 * // Perform operations
 * vsla_fill(ctx, a, 2.0);
 * vsla_fill(ctx, b, 3.0);
 * vsla_add(ctx, result, a, b);  // result = a + b
 * 
 * // Cleanup
 * vsla_tensor_free(a);
 * vsla_tensor_free(b);
 * vsla_tensor_free(result);
 * vsla_cleanup(ctx);
 * @endcode
 */

/**
 * @brief Error handling best practices
 * 
 * All VSLA functions that can fail return vsla_error_t. Always check return values:
 * 
 * @code
 * vsla_error_t err = vsla_add(ctx, result, a, b);
 * if (err != VSLA_SUCCESS) {
 *     // Handle error
 *     printf("Addition failed: %d\n", err);
 *     return -1;
 * }
 * @endcode
 */

#ifdef __cplusplus
}
#endif

#endif /* VSLA_H */
