/**
 * @file competitor_backends.c
 * @brief Implementation of competitor backends for fair performance comparisons
 */

#include "competitor_backends.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Helper function to calculate linear index from multi-dimensional index
static size_t linear_index(const competitor_tensor_t* tensor, const size_t* idx) {
    size_t linear = 0;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        linear += idx[i] * tensor->strides[i];
    }
    return linear;
}

// Helper to calculate strides from shape
static void calculate_strides(size_t* strides, const size_t* shape, uint8_t rank) {
    strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Helper to calculate total size
static size_t calculate_total_size(const size_t* shape, uint8_t rank) {
    size_t total = 1;
    for (uint8_t i = 0; i < rank; i++) {
        total *= shape[i];
    }
    return total;
}

// === COMPETITOR CONTEXT MANAGEMENT ===

competitor_context_t* competitor_init(competitor_backend_t backend) {
    competitor_context_t* ctx = (competitor_context_t*)malloc(sizeof(competitor_context_t));
    if (!ctx) return NULL;
    
    ctx->backend = backend;
    ctx->backend_data = NULL;
    
    // Backend-specific initialization
    switch (backend) {
        case COMPETITOR_RAW_C:
        case COMPETITOR_SIMPLE_BLAS:
        case COMPETITOR_DENSE_TENSOR:
            // No special initialization needed
            break;
        case COMPETITOR_SPARSE_MATRIX:
            // Could initialize sparse matrix library here
            break;
    }
    
    return ctx;
}

void competitor_cleanup(competitor_context_t* ctx) {
    if (!ctx) return;
    
    // Backend-specific cleanup
    if (ctx->backend_data) {
        free(ctx->backend_data);
    }
    
    free(ctx);
}

// === TENSOR MANAGEMENT ===

competitor_tensor_t* competitor_tensor_create(competitor_context_t* ctx, uint8_t rank, const size_t* shape) {
    if (!ctx || !shape || rank == 0) return NULL;
    
    competitor_tensor_t* tensor = (competitor_tensor_t*)malloc(sizeof(competitor_tensor_t));
    if (!tensor) return NULL;
    
    // Allocate shape and strides arrays
    tensor->shape = (size_t*)malloc(rank * sizeof(size_t));
    tensor->strides = (size_t*)malloc(rank * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and calculate strides
    memcpy(tensor->shape, shape, rank * sizeof(size_t));
    calculate_strides(tensor->strides, shape, rank);
    
    tensor->rank = rank;
    tensor->total_size = calculate_total_size(shape, rank);
    tensor->backend = ctx->backend;
    
    // Allocate data
    tensor->data = (double*)calloc(tensor->total_size, sizeof(double));
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    return tensor;
}

void competitor_tensor_free(competitor_tensor_t* tensor) {
    if (!tensor) return;
    
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

void competitor_tensor_fill(competitor_tensor_t* tensor, double value) {
    if (!tensor || !tensor->data) return;
    
    for (size_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] = value;
    }
}

void competitor_tensor_set(competitor_tensor_t* tensor, const size_t* idx, double value) {
    if (!tensor || !tensor->data || !idx) return;
    
    size_t linear = linear_index(tensor, idx);
    if (linear < tensor->total_size) {
        tensor->data[linear] = value;
    }
}

double competitor_tensor_get(competitor_tensor_t* tensor, const size_t* idx) {
    if (!tensor || !tensor->data || !idx) return 0.0;
    
    size_t linear = linear_index(tensor, idx);
    if (linear < tensor->total_size) {
        return tensor->data[linear];
    }
    return 0.0;
}

// === ARITHMETIC OPERATIONS ===

int competitor_add(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    // Check compatible shapes (simplified - assumes same shape for now)
    if (a->total_size != b->total_size || a->total_size != out->total_size) {
        return -1;
    }
    
    switch (ctx->backend) {
        case COMPETITOR_RAW_C:
        case COMPETITOR_SIMPLE_BLAS:
        case COMPETITOR_DENSE_TENSOR:
            // Simple vectorized addition
            for (size_t i = 0; i < out->total_size; i++) {
                out->data[i] = a->data[i] + b->data[i];
            }
            break;
            
        case COMPETITOR_SPARSE_MATRIX:
            // Could implement sparse addition here
            for (size_t i = 0; i < out->total_size; i++) {
                out->data[i] = a->data[i] + b->data[i];
            }
            break;
    }
    
    return 0;
}

int competitor_sub(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    if (a->total_size != b->total_size || a->total_size != out->total_size) {
        return -1;
    }
    
    for (size_t i = 0; i < out->total_size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    
    return 0;
}

int competitor_mul(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    if (a->total_size != b->total_size || a->total_size != out->total_size) {
        return -1;
    }
    
    for (size_t i = 0; i < out->total_size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    return 0;
}

int competitor_div(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    if (a->total_size != b->total_size || a->total_size != out->total_size) {
        return -1;
    }
    
    for (size_t i = 0; i < out->total_size; i++) {
        out->data[i] = (b->data[i] != 0.0) ? a->data[i] / b->data[i] : 0.0;
    }
    
    return 0;
}

// === ADVANCED OPERATIONS ===

int competitor_conv(competitor_context_t* ctx, competitor_tensor_t* out,
                   const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    // Simple O(n²) convolution for comparison
    if (a->rank != 1 || b->rank != 1 || out->rank != 1) return -1;
    
    size_t a_size = a->shape[0];
    size_t b_size = b->shape[0];
    size_t out_size = out->shape[0];
    
    if (out_size != a_size + b_size - 1) return -1;
    
    // Zero output
    competitor_tensor_fill(out, 0.0);
    
    // Direct convolution
    for (size_t i = 0; i < a_size; i++) {
        for (size_t j = 0; j < b_size; j++) {
            out->data[i + j] += a->data[i] * b->data[j];
        }
    }
    
    return 0;
}

int competitor_kron(competitor_context_t* ctx, competitor_tensor_t* out,
                   const competitor_tensor_t* a, const competitor_tensor_t* b) {
    if (!ctx || !out || !a || !b) return -1;
    
    // Simple Kronecker product for 1D tensors
    if (a->rank != 1 || b->rank != 1 || out->rank != 1) return -1;
    
    size_t a_size = a->shape[0];
    size_t b_size = b->shape[0];
    size_t out_size = out->shape[0];
    
    if (out_size != a_size * b_size) return -1;
    
    for (size_t i = 0; i < a_size; i++) {
        for (size_t j = 0; j < b_size; j++) {
            out->data[i * b_size + j] = a->data[i] * b->data[j];
        }
    }
    
    return 0;
}

// === REDUCTION OPERATIONS ===

int competitor_sum(competitor_context_t* ctx, const competitor_tensor_t* tensor, double* result) {
    if (!ctx || !tensor || !result) return -1;
    
    double sum = 0.0;
    for (size_t i = 0; i < tensor->total_size; i++) {
        sum += tensor->data[i];
    }
    
    *result = sum;
    return 0;
}

int competitor_dot(competitor_context_t* ctx, const competitor_tensor_t* a,
                  const competitor_tensor_t* b, double* result) {
    if (!ctx || !a || !b || !result) return -1;
    
    if (a->total_size != b->total_size) return -1;
    
    double dot = 0.0;
    for (size_t i = 0; i < a->total_size; i++) {
        dot += a->data[i] * b->data[i];
    }
    
    *result = dot;
    return 0;
}

// === STACKING OPERATIONS ===

int competitor_stack(competitor_context_t* ctx, competitor_tensor_t* out,
                    const competitor_tensor_t** tensors, size_t num_tensors, uint8_t axis) {
    if (!ctx || !out || !tensors || num_tensors == 0) return -1;
    
    // Simple 1D stacking (concatenation)
    if (axis != 0) return -1;  // Only support axis 0 for now
    
    size_t offset = 0;
    for (size_t t = 0; t < num_tensors; t++) {
        const competitor_tensor_t* tensor = tensors[t];
        if (!tensor) return -1;
        
        memcpy(out->data + offset, tensor->data, tensor->total_size * sizeof(double));
        offset += tensor->total_size;
    }
    
    return 0;
}

// === UTILITY FUNCTIONS ===

size_t competitor_tensor_size(const competitor_tensor_t* tensor) {
    return tensor ? tensor->total_size : 0;
}

void competitor_tensor_print(const competitor_tensor_t* tensor, const char* name) {
    if (!tensor || !name) return;
    
    printf("%s: shape=[", name);
    for (uint8_t i = 0; i < tensor->rank; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->rank - 1) printf(", ");
    }
    printf("], size=%zu\n", tensor->total_size);
    
    // Print first few elements
    printf("  data: [");
    size_t print_count = (tensor->total_size < 10) ? tensor->total_size : 10;
    for (size_t i = 0; i < print_count; i++) {
        printf("%.3f", tensor->data[i]);
        if (i < print_count - 1) printf(", ");
    }
    if (tensor->total_size > 10) printf(", ...");
    printf("]\n");
}

// === SPARSE OPERATIONS ===

sparse_vector_t* sparse_vector_create(size_t capacity) {
    sparse_vector_t* vec = (sparse_vector_t*)malloc(sizeof(sparse_vector_t));
    if (!vec) return NULL;
    
    vec->values = (double*)malloc(capacity * sizeof(double));
    vec->indices = (size_t*)malloc(capacity * sizeof(size_t));
    
    if (!vec->values || !vec->indices) {
        free(vec->values);
        free(vec->indices);
        free(vec);
        return NULL;
    }
    
    vec->nnz = 0;
    vec->capacity = capacity;
    
    return vec;
}

void sparse_vector_free(sparse_vector_t* vec) {
    if (!vec) return;
    
    free(vec->values);
    free(vec->indices);
    free(vec);
}

int sparse_add(sparse_vector_t* out, const sparse_vector_t* a, const sparse_vector_t* b) {
    if (!out || !a || !b) return -1;
    
    // Simple merge-based sparse addition
    size_t ia = 0, ib = 0, iout = 0;
    
    while (ia < a->nnz && ib < b->nnz && iout < out->capacity) {
        if (a->indices[ia] < b->indices[ib]) {
            out->indices[iout] = a->indices[ia];
            out->values[iout] = a->values[ia];
            ia++;
        } else if (a->indices[ia] > b->indices[ib]) {
            out->indices[iout] = b->indices[ib];
            out->values[iout] = b->values[ib];
            ib++;
        } else {
            out->indices[iout] = a->indices[ia];
            out->values[iout] = a->values[ia] + b->values[ib];
            ia++;
            ib++;
        }
        iout++;
    }
    
    // Add remaining elements
    while (ia < a->nnz && iout < out->capacity) {
        out->indices[iout] = a->indices[ia];
        out->values[iout] = a->values[ia];
        ia++;
        iout++;
    }
    
    while (ib < b->nnz && iout < out->capacity) {
        out->indices[iout] = b->indices[ib];
        out->values[iout] = b->values[ib];
        ib++;
        iout++;
    }
    
    out->nnz = iout;
    return 0;
}