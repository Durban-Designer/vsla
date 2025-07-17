/**
 * @file vsla_kron.h
 * @brief Model B operations - Kronecker product semiring
 * 
 * @copyright MIT License
 */

#ifndef VSLA_KRON_H
#define VSLA_KRON_H

#include "vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Kronecker product of two tensors (Model B multiplication)
 * 
 * Computes the Kronecker product: for vectors v and w,
 * v ⊗ w = (v₁w₁, v₁w₂, ..., v₁wₙ, v₂w₁, ..., vₘwₙ)
 * 
 * The output dimension is d1 * d2 where d1 and d2 are the input dimensions.
 * Note: This operation is non-commutative.
 * 
 * @param out Output tensor (pre-allocated with dimension d1*d2)
 * @param a First input tensor (must be Model B)
 * @param b Second input tensor (must be Model B)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_kron_basic(vsla_tensor_t* out, const vsla_tensor_t* a, 
                       const vsla_tensor_t* b);

/**
 * @brief Naive Kronecker product implementation
 * 
 * Direct implementation with O(d1*d2) complexity.
 * 
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_kron_naive(vsla_tensor_t* out, const vsla_tensor_t* a, 
                             const vsla_tensor_t* b);

/**
 * @brief Tiled Kronecker product implementation
 * 
 * Cache-friendly tiled implementation for better performance on large tensors.
 * 
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @param tile_size Size of tiles for blocking (0 for auto)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_kron_tiled(vsla_tensor_t* out, const vsla_tensor_t* a, 
                             const vsla_tensor_t* b, size_t tile_size);

/**
 * @brief Matrix multiplication for Model B
 * 
 * Performs matrix multiplication where each element is a Model B tensor
 * and multiplication uses Kronecker product.
 * 
 * @param out Output matrix of tensors
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_matmul_kron(vsla_tensor_t** out, vsla_tensor_t** A, 
                              vsla_tensor_t** B, size_t m, size_t k, size_t n);

/**
 * @brief Convert Model B tensor to monoid algebra representation
 * 
 * Maps tensor elements to basis elements e_i in the monoid algebra ℝ[ℕ₊,×].
 * 
 * @param tensor Input tensor (Model B)
 * @param coeffs Output coefficients (pre-allocated)
 * @param indices Output indices for basis elements (pre-allocated)
 * @param max_terms Maximum number of terms to extract
 * @param num_terms Actual number of non-zero terms
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_to_monoid_algebra(const vsla_tensor_t* tensor, double* coeffs,
                                    uint64_t* indices, size_t max_terms, 
                                    size_t* num_terms);

/**
 * @brief Create Model B tensor from monoid algebra representation
 * 
 * @param coeffs Coefficient array
 * @param indices Basis element indices
 * @param num_terms Number of terms
 * @param dtype Data type for tensor
 * @return New tensor or NULL on error
 */
vsla_tensor_t* vsla_from_monoid_algebra(const double* coeffs, 
                                        const uint64_t* indices,
                                        size_t num_terms, vsla_dtype_t dtype);

/**
 * @brief Backward pass for Kronecker product (for autograd)
 * 
 * Computes gradients with respect to inputs given output gradient.
 * 
 * @param grad_a Gradient w.r.t. first input (pre-allocated)
 * @param grad_b Gradient w.r.t. second input (pre-allocated)
 * @param grad_out Gradient of output
 * @param a Forward pass first input
 * @param b Forward pass second input
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_kron_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                                const vsla_tensor_t* grad_out,
                                const vsla_tensor_t* a, const vsla_tensor_t* b);

/**
 * @brief Check if Kronecker product is commutative for given tensors
 * 
 * Returns true if a ⊗ b = b ⊗ a for the given tensors.
 * This happens when deg(a) = 1 or deg(b) = 1 or both are scalar multiples.
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return 1 if commutative, 0 otherwise
 */
int vsla_kron_is_commutative(const vsla_tensor_t* a, const vsla_tensor_t* b);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_KRON_H */