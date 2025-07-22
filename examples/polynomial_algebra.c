/**
 * @file polynomial_algebra.c
 * @brief E2E Polynomial Algebra with Variable-Degree Operations using the new API.
 *
 * This example demonstrates polynomial operations using VSLA's convolution
 * semiring for variable-degree polynomials without requiring degree padding.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Polynomial structure
typedef struct {
    vsla_tensor_t* coeffs;  // Coefficients tensor
    int degree;
} polynomial_t;

// Create a polynomial from a coefficient array
static polynomial_t* create_polynomial(vsla_context_t* ctx, const double* coeffs, int degree) {
    polynomial_t* poly = (polynomial_t*)malloc(sizeof(polynomial_t));
    uint64_t shape[] = {degree + 1};
    poly->coeffs = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    poly->degree = degree;

    // In a real application, you would copy the coefficients to the tensor here.
    // vsla_tensor_set_data(ctx, poly->coeffs, coeffs);

    return poly;
}

// Free a polynomial
static void free_polynomial(polynomial_t* poly) {
    if (!poly) return;
    vsla_tensor_free(poly->coeffs);
    free(poly);
}

// Polynomial multiplication using VSLA convolution
static polynomial_t* poly_multiply(vsla_context_t* ctx, const polynomial_t* a, const polynomial_t* b) {
    int result_degree = a->degree + b->degree;
    uint64_t result_shape[] = {result_degree + 1};

    polynomial_t* result = (polynomial_t*)malloc(sizeof(polynomial_t));
    result->coeffs = vsla_tensor_create(ctx, 1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    result->degree = result_degree;

    vsla_conv(ctx, result->coeffs, a->coeffs, b->coeffs);

    return result;
}

// Polynomial addition
static polynomial_t* poly_add(vsla_context_t* ctx, const polynomial_t* a, const polynomial_t* b) {
    int max_degree = (a->degree > b->degree) ? a->degree : b->degree;
    uint64_t result_shape[] = {max_degree + 1};

    polynomial_t* result = (polynomial_t*)malloc(sizeof(polynomial_t));
    result->coeffs = vsla_tensor_create(ctx, 1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    result->degree = max_degree;

    vsla_add(ctx, result->coeffs, a->coeffs, b->coeffs);

    return result;
}

int main(void) {
    printf("=== VSLA Polynomial Algebra Example (New API) ===\n");

    // Initialize VSLA context
    vsla_config_t config = { .backend_selection = VSLA_BACKEND_AUTO };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    // Create two polynomials
    double p_coeffs[] = {1.0, -1.0, 3.0, 2.0};
    polynomial_t* p = create_polynomial(ctx, p_coeffs, 3);

    double q_coeffs[] = {-3.0, 2.0, 1.0};
    polynomial_t* q = create_polynomial(ctx, q_coeffs, 2);

    // Perform polynomial multiplication
    polynomial_t* pq = poly_multiply(ctx, p, q);
    printf("P(x) * Q(x) has degree %d\n", pq->degree);

    // Perform polynomial addition
    polynomial_t* p_plus_q = poly_add(ctx, p, q);
    printf("P(x) + Q(x) has degree %d\n", p_plus_q->degree);

    // Clean up
    free_polynomial(p);
    free_polynomial(q);
    free_polynomial(pq);
    free_polynomial(p_plus_q);
    vsla_cleanup(ctx);

    printf("\n✓ Polynomial algebra example completed successfully!\n");

    return 0;
}
