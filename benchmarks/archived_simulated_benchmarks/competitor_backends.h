/**
 * @file competitor_backends.h
 * @brief Competitor backend interfaces for fair performance comparisons
 * 
 * Provides identical APIs to VSLA but implemented using:
 * - Raw C arrays with manual memory management
 * - Simple BLAS-like operations
 * - Traditional dense tensor approaches
 * - Sparse matrix libraries (when available)
 */

#ifndef COMPETITOR_BACKENDS_H
#define COMPETITOR_BACKENDS_H

#include <stddef.h>
#include <stdint.h>

// Competitor backend types
typedef enum {
    COMPETITOR_RAW_C,           // Raw C arrays with manual operations
    COMPETITOR_SIMPLE_BLAS,     // Simple BLAS-like operations
    COMPETITOR_DENSE_TENSOR,    // Traditional dense tensor library
    COMPETITOR_SPARSE_MATRIX    // Sparse matrix operations
} competitor_backend_t;

// Generic tensor structure for competitors
typedef struct {
    double* data;
    size_t* shape;
    size_t* strides;
    uint8_t rank;
    size_t total_size;
    competitor_backend_t backend;
} competitor_tensor_t;

// Context for competitor operations
typedef struct {
    competitor_backend_t backend;
    void* backend_data;  // Backend-specific context
} competitor_context_t;

// === COMPETITOR CONTEXT MANAGEMENT ===
competitor_context_t* competitor_init(competitor_backend_t backend);
void competitor_cleanup(competitor_context_t* ctx);

// === TENSOR MANAGEMENT ===
competitor_tensor_t* competitor_tensor_create(competitor_context_t* ctx, uint8_t rank, const size_t* shape);
void competitor_tensor_free(competitor_tensor_t* tensor);
void competitor_tensor_fill(competitor_tensor_t* tensor, double value);
void competitor_tensor_set(competitor_tensor_t* tensor, const size_t* idx, double value);
double competitor_tensor_get(competitor_tensor_t* tensor, const size_t* idx);

// === ARITHMETIC OPERATIONS ===
int competitor_add(competitor_context_t* ctx, competitor_tensor_t* out, 
                  const competitor_tensor_t* a, const competitor_tensor_t* b);
int competitor_sub(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b);
int competitor_mul(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b);
int competitor_div(competitor_context_t* ctx, competitor_tensor_t* out,
                  const competitor_tensor_t* a, const competitor_tensor_t* b);

// === ADVANCED OPERATIONS ===
int competitor_conv(competitor_context_t* ctx, competitor_tensor_t* out,
                   const competitor_tensor_t* a, const competitor_tensor_t* b);
int competitor_kron(competitor_context_t* ctx, competitor_tensor_t* out,
                   const competitor_tensor_t* a, const competitor_tensor_t* b);

// === REDUCTION OPERATIONS ===
int competitor_sum(competitor_context_t* ctx, const competitor_tensor_t* tensor, double* result);
int competitor_dot(competitor_context_t* ctx, const competitor_tensor_t* a, 
                  const competitor_tensor_t* b, double* result);

// === STACKING OPERATIONS ===
int competitor_stack(competitor_context_t* ctx, competitor_tensor_t* out,
                    const competitor_tensor_t** tensors, size_t num_tensors, uint8_t axis);

// === UTILITY FUNCTIONS ===
size_t competitor_tensor_size(const competitor_tensor_t* tensor);
void competitor_tensor_print(const competitor_tensor_t* tensor, const char* name);

// === SPARSE OPERATIONS (when available) ===
typedef struct {
    double* values;
    size_t* indices;
    size_t nnz;  // Number of non-zeros
    size_t capacity;
} sparse_vector_t;

sparse_vector_t* sparse_vector_create(size_t capacity);
void sparse_vector_free(sparse_vector_t* vec);
int sparse_add(sparse_vector_t* out, const sparse_vector_t* a, const sparse_vector_t* b);

#endif // COMPETITOR_BACKENDS_H