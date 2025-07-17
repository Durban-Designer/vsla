/**
 * @file vsla_conv.h
 * @brief Model A operations - Convolution semiring
 * 
 * @copyright MIT License
 */

#ifndef VSLA_CONV_H
#define VSLA_CONV_H

#include "vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convolution of two tensors (Model A multiplication)
 * 
 * Computes the discrete convolution of two tensors. For vectors v and w:
 * (v * w)_k = sum_{i+j=k+1} v_i * w_j
 * 
 * The output dimension is d1 + d2 - 1 where d1 and d2 are the input dimensions.
 * Uses FFT for efficiency when available.
 * 
 * @param out Output tensor (pre-allocated with correct dimensions)
 * @param a First input tensor (must be Model A)
 * @param b Second input tensor (must be Model A)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_conv_basic(vsla_tensor_t* out, const vsla_tensor_t* a, 
                             const vsla_tensor_t* b);

/**
 * @brief Direct convolution (no FFT)
 * 
 * Computes convolution using the direct O(n*m) algorithm.
 * Useful for small tensors or when FFT is not available.
 * 
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_conv_direct(vsla_tensor_t* out, const vsla_tensor_t* a, 
                              const vsla_tensor_t* b);

/**
 * @brief FFT-based convolution
 * 
 * Uses Fast Fourier Transform for O(n log n) convolution.
 * Falls back to radix-2 implementation if FFTW is not available.
 * 
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_conv_fft(vsla_tensor_t* out, const vsla_tensor_t* a, 
                           const vsla_tensor_t* b);

/**
 * @brief Matrix multiplication for Model A
 * 
 * Performs matrix multiplication where each element is a Model A tensor
 * and multiplication uses convolution.
 * 
 * @param out Output matrix of tensors
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_matmul_conv(vsla_tensor_t** out, vsla_tensor_t** A, 
                              vsla_tensor_t** B, size_t m, size_t k, size_t n);

/**
 * @brief Compute polynomial coefficients from Model A tensor
 * 
 * Extracts the polynomial representation where tensor elements
 * are coefficients of x^0, x^1, x^2, ...
 * 
 * @param tensor Input tensor (Model A)
 * @param coeffs Output coefficient array (pre-allocated)
 * @param max_degree Maximum degree to extract
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_to_polynomial(const vsla_tensor_t* tensor, double* coeffs, 
                                size_t max_degree);

/**
 * @brief Create Model A tensor from polynomial coefficients
 * 
 * @param coeffs Coefficient array
 * @param degree Polynomial degree
 * @param dtype Data type for tensor
 * @return New tensor or NULL on error
 */
vsla_tensor_t* vsla_from_polynomial(const double* coeffs, size_t degree, 
                                    vsla_dtype_t dtype);

/**
 * @brief Backward pass for convolution (for autograd)
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
vsla_error_t vsla_conv_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out,
                               const vsla_tensor_t* a, const vsla_tensor_t* b);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_CONV_H */