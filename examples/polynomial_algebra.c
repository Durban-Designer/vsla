/**
 * @file polynomial_algebra.c
 * @brief E2E Polynomial Algebra with Variable-Degree Operations
 * 
 * This example demonstrates polynomial operations using VSLA's convolution
 * semiring for variable-degree polynomials without requiring degree padding.
 * 
 * Use Case: Symbolic computation and control systems with variable-order polynomials
 * Problem: Traditional frameworks require fixed-size arrays and manual degree tracking
 * VSLA Solution: Natural polynomial representation using convolution semiring
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_unified.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <complex.h>

#define MAX_POLY_DEGREE 1024
#define TOLERANCE 1e-10

// Polynomial structure
typedef struct {
    vsla_unified_tensor_t* coeffs;  // Coefficients tensor
    int degree;                      // Actual degree (not padded)
    char name[32];                   // Polynomial name for display
} polynomial_t;

// Root finding result
typedef struct {
    double complex* roots;
    int num_roots;
    double* residuals;
} roots_result_t;

// Create polynomial from coefficient array
static polynomial_t* create_polynomial(vsla_unified_context_t* ctx,
                                       const double* coeffs,
                                       int degree,
                                       const char* name) {
    polynomial_t* poly = malloc(sizeof(polynomial_t));
    if (!poly) return NULL;
    
    // Remove leading zeros to find actual degree
    int actual_degree = degree;
    while (actual_degree > 0 && fabs(coeffs[actual_degree]) < TOLERANCE) {
        actual_degree--;
    }
    
    poly->degree = actual_degree;
    strncpy(poly->name, name, sizeof(poly->name) - 1);
    poly->name[sizeof(poly->name) - 1] = '\0';
    
    // Create VSLA tensor with exact degree (no padding)
    uint64_t shape[] = {actual_degree + 1};
    poly->coeffs = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!poly->coeffs) {
        free(poly);
        return NULL;
    }
    
    // Copy coefficients
    double* tensor_data = (double*)vsla_tensor_data_mut(poly->coeffs, NULL);
    for (int i = 0; i <= actual_degree; i++) {
        tensor_data[i] = coeffs[i];
    }
    
    printf("Created polynomial '%s': degree %d\n", name, actual_degree);
    return poly;
}

// Create polynomial from roots (for testing)
static polynomial_t* create_polynomial_from_roots(vsla_unified_context_t* ctx,
                                                   const double complex* roots,
                                                   int num_roots,
                                                   const char* name) {
    printf("Creating polynomial from %d roots...\n", num_roots);
    
    // Start with polynomial (x - roots[0])
    double initial_coeffs[] = {-creal(roots[0]), 1.0};
    polynomial_t* result = create_polynomial(ctx, initial_coeffs, 1, "temp");
    if (!result) return NULL;
    
    // Multiply by (x - roots[i]) for each additional root
    for (int i = 1; i < num_roots; i++) {
        double factor_coeffs[] = {-creal(roots[i]), 1.0};
        polynomial_t* factor = create_polynomial(ctx, factor_coeffs, 1, "factor");
        if (!factor) break;
        
        // Multiply polynomials using convolution
        polynomial_t* new_result = poly_multiply(ctx, result, factor, "temp_product");
        
        // Replace result
        vsla_tensor_free(result->coeffs);
        free(result);
        vsla_tensor_free(factor->coeffs);
        free(factor);
        
        result = new_result;
        if (!result) break;
    }
    
    if (result) {
        strncpy(result->name, name, sizeof(result->name) - 1);
        result->name[sizeof(result->name) - 1] = '\0';
    }
    
    return result;
}

// Free polynomial
static void free_polynomial(polynomial_t* poly) {
    if (!poly) return;
    if (poly->coeffs) vsla_tensor_free(poly->coeffs);
    free(poly);
}

// Print polynomial in readable format
static void print_polynomial(const polynomial_t* poly) {
    if (!poly || !poly->coeffs) return;
    
    const double* coeffs = (const double*)vsla_tensor_data(poly->coeffs, NULL);
    printf("%s(x) = ", poly->name);
    
    bool first_term = true;
    for (int i = poly->degree; i >= 0; i--) {
        if (fabs(coeffs[i]) < TOLERANCE) continue;
        
        if (!first_term) {
            printf(coeffs[i] >= 0 ? " + " : " - ");
        } else if (coeffs[i] < 0) {
            printf("-");
        }
        
        double coeff = fabs(coeffs[i]);
        
        if (i == 0) {
            printf("%.3g", coeff);
        } else if (i == 1) {
            if (fabs(coeff - 1.0) < TOLERANCE) {
                printf("x");
            } else {
                printf("%.3g*x", coeff);
            }
        } else {
            if (fabs(coeff - 1.0) < TOLERANCE) {
                printf("x^%d", i);
            } else {
                printf("%.3g*x^%d", coeff, i);
            }
        }
        
        first_term = false;
    }
    
    if (first_term) printf("0");
    printf("\n");
}

// Polynomial multiplication using VSLA convolution
static polynomial_t* poly_multiply(vsla_unified_context_t* ctx,
                                   const polynomial_t* a,
                                   const polynomial_t* b,
                                   const char* result_name) {
    printf("  Multiplying %s (deg %d) * %s (deg %d)\n", 
           a->name, a->degree, b->name, b->degree);
    
    // Result degree is sum of input degrees
    int result_degree = a->degree + b->degree;
    
    // Create output polynomial
    uint64_t result_shape[] = {result_degree + 1};
    vsla_unified_tensor_t* result_coeffs = vsla_tensor_create(ctx, 1, result_shape,
                                                               VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!result_coeffs) return NULL;
    
    clock_t start = clock();
    
    // VSLA convolution automatically selects optimal algorithm (FFT for large polynomials)
    vsla_error_t err = vsla_conv(ctx, result_coeffs, a->coeffs, b->coeffs);
    
    clock_t end = clock();
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    if (err != VSLA_SUCCESS) {
        printf("  Convolution failed: %d\n", err);
        vsla_tensor_free(result_coeffs);
        return NULL;
    }
    
    printf("  Multiplication completed in %.3f ms\n", time_ms);
    
    // Create result polynomial
    polynomial_t* result = malloc(sizeof(polynomial_t));
    if (!result) {
        vsla_tensor_free(result_coeffs);
        return NULL;
    }
    
    result->coeffs = result_coeffs;
    result->degree = result_degree;
    strncpy(result->name, result_name, sizeof(result->name) - 1);
    result->name[sizeof(result->name) - 1] = '\0';
    
    return result;
}

// Polynomial addition
static polynomial_t* poly_add(vsla_unified_context_t* ctx,
                               const polynomial_t* a,
                               const polynomial_t* b,
                               const char* result_name) {
    printf("  Adding %s (deg %d) + %s (deg %d)\n",
           a->name, a->degree, b->name, b->degree);
    
    int max_degree = (a->degree > b->degree) ? a->degree : b->degree;
    
    // Create result polynomial
    polynomial_t* result = malloc(sizeof(polynomial_t));
    if (!result) return NULL;
    
    uint64_t result_shape[] = {max_degree + 1};
    result->coeffs = vsla_tensor_zeros(ctx, 1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!result->coeffs) {
        free(result);
        return NULL;
    }
    
    result->degree = max_degree;
    strncpy(result->name, result_name, sizeof(result->name) - 1);
    result->name[sizeof(result->name) - 1] = '\0';
    
    // Use VSLA addition with automatic variable-shape handling
    vsla_error_t err = vsla_add(ctx, result->coeffs, a->coeffs, b->coeffs);
    if (err != VSLA_SUCCESS) {
        printf("  Addition failed: %d\n", err);
        free_polynomial(result);
        return NULL;
    }
    
    return result;
}

// Polynomial evaluation using Horner's method
static double poly_evaluate(const polynomial_t* poly, double x) {
    if (!poly || !poly->coeffs) return 0.0;
    
    const double* coeffs = (const double*)vsla_tensor_data(poly->coeffs, NULL);
    
    // Horner's method: P(x) = a_n + x(a_{n-1} + x(a_{n-2} + ... + x*a_1 + a_0))
    double result = coeffs[poly->degree];
    for (int i = poly->degree - 1; i >= 0; i--) {
        result = result * x + coeffs[i];
    }
    
    return result;
}

// Polynomial derivative
static polynomial_t* poly_derivative(vsla_unified_context_t* ctx,
                                     const polynomial_t* poly,
                                     const char* result_name) {
    if (!poly || poly->degree == 0) {
        // Derivative of constant is zero
        double zero_coeffs[] = {0.0};
        return create_polynomial(ctx, zero_coeffs, 0, result_name);
    }
    
    printf("  Computing derivative of %s (deg %d)\n", poly->name, poly->degree);
    
    const double* coeffs = (const double*)vsla_tensor_data(poly->coeffs, NULL);
    
    // Derivative reduces degree by 1
    int result_degree = poly->degree - 1;
    double* deriv_coeffs = malloc((result_degree + 1) * sizeof(double));
    if (!deriv_coeffs) return NULL;
    
    // d/dx[a_n * x^n] = n * a_n * x^{n-1}
    for (int i = 0; i <= result_degree; i++) {
        deriv_coeffs[i] = (i + 1) * coeffs[i + 1];
    }
    
    polynomial_t* result = create_polynomial(ctx, deriv_coeffs, result_degree, result_name);
    free(deriv_coeffs);
    
    return result;
}

// Newton-Raphson root finding
static double find_root_newton(const polynomial_t* poly,
                               const polynomial_t* derivative,
                               double initial_guess,
                               int max_iterations) {
    double x = initial_guess;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        double f = poly_evaluate(poly, x);
        double fp = poly_evaluate(derivative, x);
        
        if (fabs(fp) < TOLERANCE) {
            printf("    Derivative too small at x=%.6f\n", x);
            break;
        }
        
        double delta = f / fp;
        x = x - delta;
        
        if (fabs(delta) < TOLERANCE) {
            printf("    Root found: x=%.8f (iterations: %d)\n", x, iter + 1);
            return x;
        }
    }
    
    printf("    Newton-Raphson did not converge from x=%.6f\n", initial_guess);
    return x;
}

// Comprehensive polynomial test suite
static void polynomial_test_suite(vsla_unified_context_t* ctx) {
    printf("\n=== Polynomial Algebra Test Suite ===\n");
    
    // Test 1: Basic polynomial operations
    printf("\n--- Test 1: Basic Operations ---\n");
    
    // P(x) = 2x^3 + 3x^2 - x + 1
    double p_coeffs[] = {1.0, -1.0, 3.0, 2.0};
    polynomial_t* P = create_polynomial(ctx, p_coeffs, 3, "P");
    
    // Q(x) = x^2 + 2x - 3
    double q_coeffs[] = {-3.0, 2.0, 1.0};
    polynomial_t* Q = create_polynomial(ctx, q_coeffs, 2, "Q");
    
    print_polynomial(P);
    print_polynomial(Q);
    
    // Test polynomial multiplication (convolution)
    polynomial_t* PQ = poly_multiply(ctx, P, Q, "P*Q");
    if (PQ) {
        print_polynomial(PQ);
        printf("  Product degree: %d (expected: %d)\n", PQ->degree, P->degree + Q->degree);
    }
    
    // Test polynomial addition
    polynomial_t* P_plus_Q = poly_add(ctx, P, Q, "P+Q");
    if (P_plus_Q) {
        print_polynomial(P_plus_Q);
    }
    
    // Test 2: Variable-degree polynomial multiplication
    printf("\n--- Test 2: Variable-Degree Multiplication ---\n");
    
    // Create polynomials of different degrees
    polynomial_t* polys[5];
    int degrees[] = {2, 5, 10, 25, 50};
    
    for (int i = 0; i < 5; i++) {
        double* coeffs = malloc((degrees[i] + 1) * sizeof(double));
        for (int j = 0; j <= degrees[i]; j++) {
            coeffs[j] = 1.0 / (j + 1); // Decreasing coefficients
        }
        
        char name[16];
        snprintf(name, sizeof(name), "poly_%d", degrees[i]);
        polys[i] = create_polynomial(ctx, coeffs, degrees[i], name);
        free(coeffs);
    }
    
    // Multiply polynomials in sequence
    polynomial_t* product = polys[0];
    for (int i = 1; i < 5; i++) {
        polynomial_t* new_product = poly_multiply(ctx, product, polys[i], "product");
        if (i > 1) free_polynomial(product);
        product = new_product;
        if (!product) break;
    }
    
    if (product) {
        printf("Final product degree: %d\n", product->degree);
        if (product != polys[0]) free_polynomial(product);
    }
    
    // Test 3: Root finding demonstration
    printf("\n--- Test 3: Root Finding ---\n");
    
    // Create polynomial with known roots: (x-1)(x-2)(x+3) = x^3 - 7x + 6
    double roots_poly_coeffs[] = {6.0, -7.0, 0.0, 1.0};
    polynomial_t* roots_poly = create_polynomial(ctx, roots_poly_coeffs, 3, "roots_test");
    print_polynomial(roots_poly);
    
    // Compute derivative for Newton-Raphson
    polynomial_t* derivative = poly_derivative(ctx, roots_poly, "derivative");
    if (derivative) {
        print_polynomial(derivative);
        
        // Find roots using different starting points
        printf("Finding roots:\n");
        double guesses[] = {0.5, 1.5, -2.5};
        for (int i = 0; i < 3; i++) {
            find_root_newton(roots_poly, derivative, guesses[i], 50);
        }
        
        free_polynomial(derivative);
    }
    
    // Test 4: Performance benchmarking
    printf("\n--- Test 4: Performance Benchmarking ---\n");
    
    int bench_degrees[] = {10, 50, 100, 250, 500, 1000};
    int num_bench = sizeof(bench_degrees) / sizeof(bench_degrees[0]);
    
    printf("Degree | Multiply Time | Add Time | Memory\n");
    printf("-------|---------------|----------|--------\n");
    
    for (int i = 0; i < num_bench; i++) {
        int deg = bench_degrees[i];
        
        // Create random polynomials
        double* coeffs1 = malloc((deg + 1) * sizeof(double));
        double* coeffs2 = malloc((deg + 1) * sizeof(double));
        
        for (int j = 0; j <= deg; j++) {
            coeffs1[j] = (double)rand() / RAND_MAX;
            coeffs2[j] = (double)rand() / RAND_MAX;
        }
        
        polynomial_t* bench1 = create_polynomial(ctx, coeffs1, deg, "bench1");
        polynomial_t* bench2 = create_polynomial(ctx, coeffs2, deg, "bench2");
        
        // Benchmark multiplication
        clock_t start = clock();
        polynomial_t* mult_result = poly_multiply(ctx, bench1, bench2, "mult_bench");
        clock_t end = clock();
        double mult_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        
        // Benchmark addition
        start = clock();
        polynomial_t* add_result = poly_add(ctx, bench1, bench2, "add_bench");
        end = clock();
        double add_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        
        // Get memory stats
        vsla_stats_t stats;
        vsla_get_stats(ctx, &stats);
        
        printf("%6d | %11.2f ms | %6.2f ms | %6zu MB\n",
               deg, mult_time, add_time, stats.memory_used_mb);
        
        // Cleanup
        free(coeffs1);
        free(coeffs2);
        free_polynomial(bench1);
        free_polynomial(bench2);
        if (mult_result) free_polynomial(mult_result);
        if (add_result) free_polynomial(add_result);
    }
    
    // Cleanup test polynomials
    free_polynomial(P);
    free_polynomial(Q);
    if (PQ) free_polynomial(PQ);
    if (P_plus_Q) free_polynomial(P_plus_Q);
    for (int i = 1; i < 5; i++) {
        free_polynomial(polys[i]);
    }
    free_polynomial(roots_poly);
}

int main(void) {
    printf("=== VSLA Polynomial Algebra Example ===\n");
    printf("Demonstrating variable-degree polynomial operations using convolution semiring\n\n");
    
    // Initialize VSLA
    vsla_unified_context_t* ctx = vsla_init(NULL);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }
    
    // Get runtime information
    vsla_backend_t backend;
    char device_name[256];
    double memory_gb;
    vsla_get_runtime_info(ctx, &backend, device_name, &memory_gb);
    
    printf("Runtime: Backend=%d, Device=%s, Memory=%.1fGB\n", 
           backend, device_name, memory_gb);
    
    // Run comprehensive test suite
    polynomial_test_suite(ctx);
    
    // Final performance statistics
    printf("\n=== Final Performance Statistics ===\n");
    vsla_stats_t stats;
    vsla_get_stats(ctx, &stats);
    
    printf("Total operations: %lu\n", stats.total_operations);
    printf("GPU operations: %lu (%.1f%%)\n", stats.gpu_operations,
           100.0 * stats.gpu_operations / stats.total_operations);
    printf("CPU operations: %lu (%.1f%%)\n", stats.cpu_operations,
           100.0 * stats.cpu_operations / stats.total_operations);
    printf("Peak memory usage: %zu MB\n", stats.peak_memory_mb);
    printf("Total computation time: %.2f ms\n", stats.total_time_ms);
    
    // Cleanup
    vsla_cleanup(ctx);
    
    printf("\n✓ Polynomial algebra example completed successfully!\n");
    printf("VSLA's convolution semiring provided natural polynomial operations\n");
    printf("with automatic degree handling and performance optimization.\n");
    
    return 0;
}