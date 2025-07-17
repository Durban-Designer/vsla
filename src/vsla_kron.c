/**
 * @file vsla_kron.c
 * @brief Model B operations - Kronecker product semiring
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "vsla/vsla_kron.h"
#include "vsla/vsla_tensor.h"
#include "vsla/vsla_core.h"
#include "vsla/vsla_backend_cpu.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Helper function to validate Kronecker inputs
static vsla_error_t validate_kron_inputs(const vsla_tensor_t* out, 
                                        const vsla_tensor_t* a, 
                                        const vsla_tensor_t* b) {
    if (!out || !a || !b) return VSLA_ERROR_NULL_POINTER;
    if (a->model != VSLA_MODEL_B || b->model != VSLA_MODEL_B) {
        return VSLA_ERROR_INVALID_MODEL;
    }
    if (a->dtype != b->dtype || out->dtype != a->dtype) {
        return VSLA_ERROR_INVALID_DTYPE;
    }
    if (a->rank != b->rank || out->rank != a->rank) {
        return VSLA_ERROR_DIMENSION_MISMATCH;
    }
    
    // Check output dimensions: should be a.shape * b.shape
    for (uint8_t i = 0; i < a->rank; i++) {
        uint64_t expected_dim = a->shape[i] * b->shape[i];
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

vsla_error_t vsla_kron_naive(vsla_tensor_t* out, const vsla_tensor_t* a, 
                             const vsla_tensor_t* b) {
    vsla_error_t err = validate_kron_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // Zero the output tensor
    err = vsla_fill_basic(out, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    // For 1D case (most common)
    if (a->rank == 1) {
        for (uint64_t i = 0; i < a->shape[0]; i++) {
            for (uint64_t j = 0; j < b->shape[0]; j++) {
                uint64_t out_idx = i * b->shape[0] + j;
                if (out_idx < out->shape[0]) {
                    double a_val, b_val;
                    
                    err = get_value_at_indices(a, &i, &a_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    err = get_value_at_indices(b, &j, &b_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    err = set_value_at_indices(out, &out_idx, a_val * b_val);
                    if (err != VSLA_SUCCESS) return err;
                }
            }
        }
        return VSLA_SUCCESS;
    }
    
    // General multi-dimensional Kronecker product
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
            // Calculate output indices: out[i,j] = a[i] * b[j]
            for (uint8_t d = 0; d < a->rank; d++) {
                out_indices[d] = a_indices[d] * b->shape[d] + b_indices[d];
            }
            
            // Check if output indices are valid
            if (indices_in_bounds(out, out_indices)) {
                double a_val, b_val;
                
                err = get_value_at_indices(a, a_indices, &a_val);
                if (err != VSLA_SUCCESS) goto cleanup;
                
                err = get_value_at_indices(b, b_indices, &b_val);
                if (err != VSLA_SUCCESS) goto cleanup;
                
                err = set_value_at_indices(out, out_indices, a_val * b_val);
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

vsla_error_t vsla_kron_tiled(vsla_tensor_t* out, const vsla_tensor_t* a, 
                             const vsla_tensor_t* b, size_t tile_size) {
    vsla_error_t err = validate_kron_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // Auto-determine tile size if not provided
    if (tile_size == 0) {
        // Use sqrt of cache size as heuristic (assuming 32KB L1 cache)
        tile_size = 64;
    }
    
    // For small tensors or when tiling is not beneficial, use naive implementation
    if (a->rank != 1 || a->shape[0] < tile_size || b->shape[0] < tile_size) {
        return vsla_kron_naive(out, a, b);
    }
    
    // Zero the output tensor
    err = vsla_fill_basic(out, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    // Tiled implementation for 1D case
    uint64_t a_size = a->shape[0];
    uint64_t b_size = b->shape[0];
    
    for (uint64_t a_tile = 0; a_tile < a_size; a_tile += tile_size) {
        uint64_t a_end = (a_tile + tile_size < a_size) ? a_tile + tile_size : a_size;
        
        for (uint64_t b_tile = 0; b_tile < b_size; b_tile += tile_size) {
            uint64_t b_end = (b_tile + tile_size < b_size) ? b_tile + tile_size : b_size;
            
            // Process tile
            for (uint64_t i = a_tile; i < a_end; i++) {
                double a_val;
                err = get_value_at_indices(a, &i, &a_val);
                if (err != VSLA_SUCCESS) return err;
                
                for (uint64_t j = b_tile; j < b_end; j++) {
                    double b_val;
                    err = get_value_at_indices(b, &j, &b_val);
                    if (err != VSLA_SUCCESS) return err;
                    
                    uint64_t out_idx = i * b_size + j;
                    err = set_value_at_indices(out, &out_idx, a_val * b_val);
                    if (err != VSLA_SUCCESS) return err;
                }
            }
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_kron_basic(vsla_tensor_t* out, const vsla_tensor_t* a, 
                       const vsla_tensor_t* b) {
    vsla_error_t err = validate_kron_inputs(out, a, b);
    if (err != VSLA_SUCCESS) return err;
    
    // Use tiled implementation for larger 1D tensors, naive for small or multi-dimensional
    if (a->rank == 1 && a->shape[0] * b->shape[0] > 4096) {
        return vsla_kron_tiled(out, a, b, 0);  // Auto tile size
    } else {
        return vsla_kron_naive(out, a, b);
    }
}

vsla_error_t vsla_to_monoid_algebra(const vsla_tensor_t* tensor, double* coeffs,
                                    uint64_t* indices, size_t max_terms, 
                                    size_t* num_terms) {
    if (!tensor || !coeffs || !indices || !num_terms) return VSLA_ERROR_NULL_POINTER;
    if (tensor->model != VSLA_MODEL_B) return VSLA_ERROR_INVALID_MODEL;
    if (tensor->rank != 1) return VSLA_ERROR_INVALID_ARGUMENT;
    
    size_t terms = 0;
    uint64_t tensor_size = tensor->shape[0];
    
    for (uint64_t i = 0; i < tensor_size && terms < max_terms; i++) {
        double val;
        vsla_error_t err = get_value_at_indices(tensor, &i, &val);
        if (err != VSLA_SUCCESS) return err;
        
        if (fabs(val) > 1e-15) {  // Non-zero coefficient
            coeffs[terms] = val;
            indices[terms] = i + 1;  // Basis elements e_1, e_2, ...
            terms++;
        }
    }
    
    *num_terms = terms;
    return VSLA_SUCCESS;
}

vsla_tensor_t* vsla_from_monoid_algebra(const double* coeffs, 
                                        const uint64_t* indices,
                                        size_t num_terms, vsla_dtype_t dtype) {
    if (!coeffs || !indices || num_terms == 0) return NULL;
    
    // Find maximum index to determine tensor size
    uint64_t max_idx = 0;
    for (size_t i = 0; i < num_terms; i++) {
        if (indices[i] > max_idx) max_idx = indices[i];
    }
    
    if (max_idx == 0) return NULL;
    
    uint64_t shape = max_idx;  // Basis elements are 1-indexed, so size is max_idx
    vsla_tensor_t* tensor = vsla_new(1, &shape, VSLA_MODEL_B, dtype);
    if (!tensor) return NULL;
    
    // Initialize to zero
    if (vsla_cpu_fill(tensor, 0.0) != VSLA_SUCCESS) {
        vsla_free(tensor);
        return NULL;
    }
    
    // Set coefficients
    for (size_t i = 0; i < num_terms; i++) {
        if (indices[i] == 0 || indices[i] > max_idx) {
            vsla_free(tensor);
            return NULL;
        }
        
        uint64_t tensor_idx = indices[i] - 1;  // Convert to 0-indexed
        if (set_value_at_indices(tensor, &tensor_idx, coeffs[i]) != VSLA_SUCCESS) {
            vsla_free(tensor);
            return NULL;
        }
    }
    
    return tensor;
}

vsla_error_t vsla_matmul_kron(vsla_tensor_t** out, vsla_tensor_t** A, 
                              vsla_tensor_t** B, size_t m, size_t k, size_t n) {
    if (!out || !A || !B) return VSLA_ERROR_NULL_POINTER;
    
    // For each output position (i,j), compute sum over k of A[i][k] ⊗ B[k][j]
    // where ⊗ is Kronecker product
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            // Initialize output[i][j] to zero
            if (out[i * n + j]) {
                vsla_error_t err = vsla_fill_basic(out[i * n + j], 0.0);
                if (err != VSLA_SUCCESS) return err;
            }
            
            for (size_t l = 0; l < k; l++) {
                vsla_tensor_t* a_elem = A[i * k + l];
                vsla_tensor_t* b_elem = B[l * n + j];
                
                if (!a_elem || !b_elem || !out[i * n + j]) {
                    return VSLA_ERROR_NULL_POINTER;
                }
                
                // Create temporary tensor for Kronecker product result
                uint64_t kron_shape[8];  // Max 8 dimensions
                for (uint8_t d = 0; d < a_elem->rank; d++) {
                    kron_shape[d] = a_elem->shape[d] * b_elem->shape[d];
                }
                
                vsla_tensor_t* temp = vsla_new(a_elem->rank, kron_shape, 
                                              VSLA_MODEL_B, a_elem->dtype);
                if (!temp) return VSLA_ERROR_MEMORY;
                
                // Compute Kronecker product
                vsla_error_t err = vsla_kron_basic(temp, a_elem, b_elem);
                if (err != VSLA_SUCCESS) {
                    vsla_free(temp);
                    return err;
                }
                
                // Add to output[i][j] (requires addition with padding)
                err = vsla_cpu_add(out[i * n + j], out[i * n + j], temp);
                vsla_free(temp);
                if (err != VSLA_SUCCESS) return err;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

int vsla_kron_is_commutative(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!a || !b) return 0;
    if (a->model != VSLA_MODEL_B || b->model != VSLA_MODEL_B) return 0;
    if (a->rank != b->rank) return 0;
    
    // Check if either tensor has degree 1 (single element)
    int a_is_scalar = 1, b_is_scalar = 1;
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != 1) a_is_scalar = 0;
        if (b->shape[i] != 1) b_is_scalar = 0;
    }
    
    if (a_is_scalar || b_is_scalar) return 1;
    
    // Check if both are 1D vectors of length 1
    if (a->rank == 1 && a->shape[0] == 1 && b->rank == 1 && b->shape[0] == 1) {
        return 1;
    }
    
    // For general case, Kronecker product is typically non-commutative
    // Could implement full check by computing both a⊗b and b⊗a and comparing,
    // but this is expensive and rarely true for non-trivial cases
    return 0;
}

vsla_error_t vsla_kron_backward(vsla_tensor_t* grad_a, vsla_tensor_t* grad_b,
                                const vsla_tensor_t* grad_out,
                                const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!grad_a || !grad_b || !grad_out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // For Kronecker product backward pass on 1D tensors:
    // If c = a ⊗ b, then:
    // grad_a[i] = sum_j(grad_out[i*len_b + j] * b[j])
    // grad_b[j] = sum_i(grad_out[i*len_b + j] * a[i])
    
    // Only support 1D tensors for now
    if (a->rank != 1 || b->rank != 1 || grad_out->rank != 1) {
        return VSLA_ERROR_NOT_IMPLEMENTED; // Multi-dimensional not supported yet
    }
    
    size_t len_a = a->shape[0];
    size_t len_b = b->shape[0];
    
    // Verify grad_out has the expected size
    if (grad_out->shape[0] != len_a * len_b) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Initialize gradients to zero (using existing grad_a and grad_b)
    vsla_error_t err = vsla_fill_basic(grad_a, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    err = vsla_cpu_fill(grad_b, 0.0);
    if (err != VSLA_SUCCESS) return err;
    
    if (a->dtype == VSLA_DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* grad_out_data = (float*)grad_out->data;
        float* grad_a_data = (float*)grad_a->data;
        float* grad_b_data = (float*)grad_b->data;
        
        // Compute grad_a[i] = sum_j(grad_out[i*len_b + j] * b[j])
        for (size_t i = 0; i < len_a; i++) {
            for (size_t j = 0; j < len_b; j++) {
                grad_a_data[i] += grad_out_data[i * len_b + j] * b_data[j];
            }
        }
        
        // Compute grad_b[j] = sum_i(grad_out[i*len_b + j] * a[i])
        for (size_t j = 0; j < len_b; j++) {
            for (size_t i = 0; i < len_a; i++) {
                grad_b_data[j] += grad_out_data[i * len_b + j] * a_data[i];
            }
        }
    } else if (a->dtype == VSLA_DTYPE_F64) {
        double* a_data = (double*)a->data;
        double* b_data = (double*)b->data;
        double* grad_out_data = (double*)grad_out->data;
        double* grad_a_data = (double*)grad_a->data;
        double* grad_b_data = (double*)grad_b->data;
        
        // Compute grad_a[i] = sum_j(grad_out[i*len_b + j] * b[j])
        for (size_t i = 0; i < len_a; i++) {
            for (size_t j = 0; j < len_b; j++) {
                grad_a_data[i] += grad_out_data[i * len_b + j] * b_data[j];
            }
        }
        
        // Compute grad_b[j] = sum_i(grad_out[i*len_b + j] * a[i])
        for (size_t j = 0; j < len_b; j++) {
            for (size_t i = 0; i < len_a; i++) {
                grad_b_data[j] += grad_out_data[i * len_b + j] * a_data[i];
            }
        }
    } else {
        return VSLA_ERROR_INVALID_ARGUMENT; // Unsupported data type
    }
    
    return VSLA_SUCCESS;
}