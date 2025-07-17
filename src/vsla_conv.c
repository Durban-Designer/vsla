/**
 * @file vsla_conv.c
 * @brief Model A operations - Convolution semiring
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "vsla/vsla_conv.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>

// Complex number typedef for FFT implementation
typedef double complex cplx;

// Helper function to validate convolution inputs
static vsla_error_t validate_conv_inputs(const vsla_tensor_t* out, 
                                        const vsla_tensor_t* a, 
                                        const vsla_tensor_t* b) {
    if (!out || !a || !b) return VSLA_ERROR_NULL_POINTER;
    if (a->model != VSLA_MODEL_A || b->model != VSLA_MODEL_A) {
        return VSLA_ERROR_INVALID_MODEL;
    }
    if (a->dtype != b->dtype || out->dtype != a->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    if (a->rank != b->rank || out->rank != a->rank) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    // Check output dimensions: should be a.shape + b.shape - 1
    for (uint8_t i = 0; i < a->rank; i++) {
        uint64_t expected_dim = a->shape[i] + b->shape[i] - 1;
        if (out->shape[i] != expected_dim) {
            return VSLA_ERROR_DIMENSION_MISMATCH;
        }
    }
    
    return VSLA_SUCCESS;
}


// Helper to set value at multi-dimensional indices
static vsla_error_t set_value_at_indices(vsla_tensor_t* tensor, 
                                         const uint64_t* indices, 
                                         double value) {
    return vsla_set_f64(tensor, indices, value);
}

// Helper to get value at multi-dimensional indices
static vsla_error_t get_value_at_indices(const vsla_tensor_t* tensor, 
                                         const uint64_t* indices, 
                                         double* value) {
    return vsla_get_f64(tensor, indices, value);
}

// Check if indices are within bounds
static int indices_in_bounds(const vsla_tensor_t* tensor, const uint64_t* indices) {
    for (uint8_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->shape[i]) {
            return 0;
        }
    }
    return 1;
}

vsla_error_t vsla_conv_direct(vsla_tensor_t* out, const vsla_tensor_t* a, 
                              const vsla_tensor_t* b) {
    vsla_error_t err = validate_conv_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // Zero the output tensor
    err = vsla_fill(out, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    // For 1D case (most common)
    if (a->rank == 1) {
        for (uint64_t i = 0; i < a->shape[0]; i++) {
            for (uint64_t j = 0; j < b->shape[0]; j++) {
                uint64_t k = i + j;
                if (k < out->shape[0]) {
                    double a_val, b_val, out_val;
                    
                    err = get_value_at_indices(a, &i, &a_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    err = get_value_at_indices(b, &j, &b_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    err = get_value_at_indices(out, &k, &out_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    err = set_value_at_indices(out, &k, out_val + a_val * b_val);
                    if (err != VSLA_SUCCESS) return err;
                }
            }
        }
        return VSLA_SUCCESS;
    }
    
    // General multi-dimensional convolution
    // This is computationally intensive but mathematically correct
    uint64_t* a_indices = malloc(a->rank * sizeof(uint64_t));
    uint64_t* b_indices = malloc(b->rank * sizeof(uint64_t));
    uint64_t* out_indices = malloc(out->rank * sizeof(uint64_t));
    
    if (!a_indices || !b_indices || !out_indices) {
        free(a_indices);
        free(b_indices);
        free(out_indices);
        return VSLA_ERROR_MEMORY;
    }
    
    // Initialize indices
    memset(a_indices, 0, a->rank * sizeof(uint64_t));
    
    // Nested loops for all dimensions
    int done = 0;
    while (!done) {
        memset(b_indices, 0, b->rank * sizeof(uint64_t));
        int b_done = 0;
        
        while (!b_done) {
            // Calculate output indices
            for (uint8_t d = 0; d < a->rank; d++) {
                out_indices[d] = a_indices[d] + b_indices[d];
            }
            
            // Check if output indices are valid
            if (indices_in_bounds(out, out_indices)) {
                double a_val, b_val, out_val;
                
                err = get_value_at_indices(a, a_indices, &a_val);
                if (err != VSLA_SUCCESS) goto cleanup;
                
                err = get_value_at_indices(b, b_indices, &b_val);
                if (err != VSLA_SUCCESS) goto cleanup;
                
                err = get_value_at_indices(out, out_indices, &out_val);
                if (err != VSLA_SUCCESS) goto cleanup;
                
                err = set_value_at_indices(out, out_indices, out_val + a_val * b_val);
                if (err != VSLA_SUCCESS) goto cleanup;
            }
            
            // Increment b_indices
            int carry = 1;
            for (int d = b->rank - 1; d >= 0 && carry; d--) {
                b_indices[d]++;
                if (b_indices[d] < b->shape[d]) {
                    carry = 0;
                } else {
                    b_indices[d] = 0;
                }
            }
            if (carry) b_done = 1;
        }
        
        // Increment a_indices
        int carry = 1;
        for (int d = a->rank - 1; d >= 0 && carry; d--) {
            a_indices[d]++;
            if (a_indices[d] < a->shape[d]) {
                carry = 0;
            } else {
                a_indices[d] = 0;
            }
        }
        if (carry) done = 1;
    }
    
cleanup:
    free(a_indices);
    free(b_indices);
    free(out_indices);
    return err;
}

// Simple radix-2 FFT implementation
static void fft_radix2(cplx* x, size_t n, int inverse) {
    if (n <= 1) return;
    
    // Bit-reversal permutation
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        for (size_t k = 1; k < n; k <<= 1) {
            j = (j << 1) | ((i & k) ? 1 : 0);
        }
        if (i < j) {
            cplx temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
    
    // Cooley-Tukey FFT
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? 2.0 : -2.0) * M_PI / len;
        cplx wlen = cos(angle) + I * sin(angle);
        
        for (size_t i = 0; i < n; i += len) {
            cplx w = 1.0;
            for (size_t j = 0; j < len / 2; j++) {
                cplx u = x[i + j];
                cplx v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    
    if (inverse) {
        for (size_t i = 0; i < n; i++) {
            x[i] /= n;
        }
    }
}

vsla_error_t vsla_conv_fft(vsla_tensor_t* out, const vsla_tensor_t* a, 
                           const vsla_tensor_t* b) {
    vsla_error_t err = validate_conv_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // For now, only implement 1D FFT convolution
    if (a->rank != 1) {
        // Fall back to direct convolution for multi-dimensional
        return vsla_conv_direct(out, a, b);
    }
    
    size_t a_len = a->shape[0];
    size_t b_len = b->shape[0];
    size_t conv_len = a_len + b_len - 1;
    
    // Find next power of 2 for FFT size
    size_t fft_size = 1;
    while (fft_size < conv_len) fft_size <<= 1;
    
    // Allocate FFT buffers
    cplx* a_fft = calloc(fft_size, sizeof(cplx));
    cplx* b_fft = calloc(fft_size, sizeof(cplx));
    
    if (!a_fft || !b_fft) {
        free(a_fft);
        free(b_fft);
        return VSLA_ERROR_MEMORY;
    }
    
    // Copy input data to FFT buffers
    for (size_t i = 0; i < a_len; i++) {
        double val;
        err = get_value_at_indices(a, &i, &val);
        if (err != VSLA_SUCCESS) {
            free(a_fft);
            free(b_fft);
            return err;
        }
        a_fft[i] = val;
    }
    
    for (size_t i = 0; i < b_len; i++) {
        double val;
        err = get_value_at_indices(b, &i, &val);
        if (err != VSLA_SUCCESS) {
            free(a_fft);
            free(b_fft);
            return err;
        }
        b_fft[i] = val;
    }
    
    // Forward FFT
    fft_radix2(a_fft, fft_size, 0);
    fft_radix2(b_fft, fft_size, 0);
    
    // Point-wise multiplication
    for (size_t i = 0; i < fft_size; i++) {
        a_fft[i] *= b_fft[i];
    }
    
    // Inverse FFT
    fft_radix2(a_fft, fft_size, 1);
    
    // Copy result to output tensor
    for (size_t i = 0; i < conv_len && i < out->shape[0]; i++) {
        err = set_value_at_indices(out, &i, creal(a_fft[i]));
        if (err != VSLA_SUCCESS) {
            free(a_fft);
            free(b_fft);
            return err;
        }
    }
    
    free(a_fft);
    free(b_fft);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_conv(vsla_tensor_t* out, const vsla_tensor_t* a, 
                       const vsla_tensor_t* b) {
    vsla_error_t err = validate_conv_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // Use FFT for larger 1D convolutions, direct for small or multi-dimensional
    if (a->rank == 1 && a->shape[0] * b->shape[0] > 64) {
        return vsla_conv_fft(out, a, b);
    } else {
        return vsla_conv_direct(out, a, b);
    }
}

vsla_error_t vsla_to_polynomial(const vsla_tensor_t* tensor, double* coeffs, 
                                size_t max_degree) {
    if (!tensor || !coeffs) return VSLA_ERROR_NULL_POINTER;
    if (tensor->model != VSLA_MODEL_A) return VSLA_ERROR_INVALID_MODEL;
    if (tensor->rank != 1) return VSLA_ERROR_INVALID_ARGUMENT;
    
    size_t degree = tensor->shape[0];
    if (degree > max_degree + 1) degree = max_degree + 1;
    
    for (size_t i = 0; i < degree; i++) {
        vsla_error_t err = get_value_at_indices(tensor, &i, &coeffs[i]);
        if (err != VSLA_SUCCESS) return err;
    }
    
    // Zero remaining coefficients
    for (size_t i = degree; i <= max_degree; i++) {
        coeffs[i] = 0.0;
    }
    
    return VSLA_SUCCESS;
}

vsla_tensor_t* vsla_from_polynomial(const double* coeffs, size_t degree, 
                                    vsla_dtype_t dtype) {
    if (!coeffs || degree == 0) return NULL;
    
    uint64_t shape = degree;
    vsla_tensor_t* tensor = vsla_new(1, &shape, VSLA_MODEL_A, dtype);
    if (!tensor) return NULL;
    
    for (size_t i = 0; i < degree; i++) {
        if (set_value_at_indices(tensor, &i, coeffs[i]) != VSLA_SUCCESS) {
            vsla_free(tensor);
            return NULL;
        }
    }
    
    return tensor;
}

vsla_error_t vsla_matmul_conv(vsla_tensor_t** out, vsla_tensor_t** A, 
                              vsla_tensor_t** B, size_t m, size_t k, size_t n) {
    if (!out || !A || !B) return VSLA_ERROR_NULL_POINTER;
    
    // For each output position (i,j), compute sum over k of A[i][k] * B[k][j]
    // where * is convolution
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            // Initialize output[i][j] to zero
            if (out[i * n + j]) {
                vsla_error_t err = vsla_fill(out[i * n + j], 0.0);
                if (err != VSLA_SUCCESS) return err;
            }
            
            for (size_t l = 0; l < k; l++) {
                vsla_tensor_t* a_elem = A[i * k + l];
                vsla_tensor_t* b_elem = B[l * n + j];
                
                if (!a_elem || !b_elem || !out[i * n + j]) {
                    return VSLA_ERROR_NULL_POINTER;
                }
                
                // Create temporary tensor for convolution result
                uint64_t conv_shape[8];  // Max 8 dimensions
                for (uint8_t d = 0; d < a_elem->rank; d++) {
                    conv_shape[d] = a_elem->shape[d] + b_elem->shape[d] - 1;
                }
                
                vsla_tensor_t* temp = vsla_new(a_elem->rank, conv_shape, 
                                              VSLA_MODEL_A, a_elem->dtype);
                if (!temp) return VSLA_ERROR_MEMORY;
                
                // Compute convolution
                vsla_error_t err = vsla_conv(temp, a_elem, b_elem);
                if (err != VSLA_SUCCESS) {
                    vsla_free(temp);
                    return err;
                }
                
                // Add to output[i][j] (requires addition with padding)
                err = vsla_add(out[i * n + j], out[i * n + j], temp);
                vsla_free(temp);
                if (err != VSLA_SUCCESS) return err;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

// Helper function to flip tensor elements (for convolution gradients)
static vsla_error_t flip_tensor_1d(vsla_tensor_t** flipped, const vsla_tensor_t* input) {
    if (!flipped || !input || input->rank != 1) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    *flipped = vsla_new(input->rank, input->shape, input->model, input->dtype);
    if (!*flipped) {
        return VSLA_ERROR_MEMORY;
    }
    
    size_t n = input->shape[0];
    size_t element_size = vsla_dtype_size(input->dtype);
    
    for (size_t i = 0; i < n; i++) {
        size_t src_offset = i * element_size;
        size_t dst_offset = (n - 1 - i) * element_size;
        memcpy((char*)(*flipped)->data + dst_offset, (char*)input->data + src_offset, element_size);
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_conv_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                               const vsla_tensor_t* grad_out,
                               const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!grad_a || !grad_b || !grad_out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For 1D convolution, the gradient computation involves:
    // grad_a = conv(grad_out, flip(b))
    // grad_b = conv(flip(a), grad_out)
    
    // Only support 1D convolution for now
    if (a->rank != 1 || b->rank != 1 || grad_out->rank != 1) {
        return VSLA_ERROR_NOT_IMPLEMENTED; // Multi-dimensional not supported yet
    }
    
    vsla_error_t err;
    vsla_tensor_t* b_flipped = NULL;
    vsla_tensor_t* a_flipped = NULL;
    
    // Create flipped versions
    err = flip_tensor_1d(&b_flipped, b);
    if (err != VSLA_SUCCESS) return err;
    
    err = flip_tensor_1d(&a_flipped, a);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_flipped);
        return err;
    }
    
    // Compute grad_a = conv(grad_out, flip(b))
    err = vsla_conv(grad_a, grad_out, b_flipped);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_flipped);
        vsla_free(a_flipped);
        return err;
    }
    
    // Compute grad_b = conv(flip(a), grad_out)
    err = vsla_conv(grad_b, a_flipped, grad_out);
    if (err != VSLA_SUCCESS) {
        vsla_free(b_flipped);
        vsla_free(a_flipped);
        return err;
    }
    
    vsla_free(b_flipped);
    vsla_free(a_flipped);
    return VSLA_SUCCESS;
}