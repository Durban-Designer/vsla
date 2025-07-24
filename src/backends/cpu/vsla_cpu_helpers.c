/**
 * @file vsla_cpu_helpers.c
 * @brief VSLA CPU backend helper functions following v3.1 specification
 * 
 * Implements the helper functions defined in section 2.3 of the spec:
 * - Overflow guards
 * - Stride computation  
 * - Index calculations
 * - Bounds checking
 */

#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <stdint.h>
#include <stdbool.h>

// Overflow guard as specified in section 2.3
bool mul_ov(uint64_t a, uint64_t b) {
    return b && a > UINT64_MAX / b;
}

// Helper functions from section 2.2
bool vsla_is_empty(const vsla_tensor_t* t) {
    if (!t) return true;
    for (uint8_t i = 0; i < t->rank; i++) {
        if (t->shape[i] == 0) return true;
    }
    return false;
}

uint64_t vsla_logical_elems(const vsla_tensor_t* t) {
    if (!t || vsla_is_empty(t)) return 0;
    
    uint64_t product = 1;
    for (uint8_t i = 0; i < t->rank; i++) {
        if (mul_ov(product, t->shape[i])) {
            return UINT64_MAX; // Overflow indicator
        }
        product *= t->shape[i];
    }
    return product;
}

uint64_t vsla_capacity_elems(const vsla_tensor_t* t) {
    if (!t || vsla_is_empty(t)) return 0;
    
    uint64_t product = 1;
    for (uint8_t i = 0; i < t->rank; i++) {
        if (mul_ov(product, t->cap[i])) {
            return UINT64_MAX; // Overflow indicator
        }
        product *= t->cap[i];
    }
    return product;
}

// Stride computation from section 2.3
void compute_strides(const vsla_tensor_t* t, uint64_t* s) {
    uint64_t acc = 1;
    for (int j = t->rank - 1; j >= 0; --j) {
        s[j] = acc;
        acc *= t->cap[j];
    }
}

uint64_t vsla_offset(const vsla_tensor_t* t, const uint64_t* idx) {
    uint64_t strides[VSLA_MAX_RANK];
    compute_strides(t, strides);
    uint64_t off = 0;
    for (int j = 0; j < t->rank; ++j) {
        off += idx[j] * strides[j];
    }
    return off;
}

// Unravel function from section 2.3
void unravel(uint64_t lin, const uint64_t* shape, uint8_t rank, uint64_t* out) {
    for (int j = rank - 1; j >= 0; --j) {
        uint64_t s = shape[j];
        out[j] = (s ? lin % s : 0);
        lin /= (s ? s : 1); // Division by 1 avoids /0 for zero-sized dims
    }
}

// Bounds checking for arithmetic operations
bool in_bounds(const vsla_tensor_t* t, const uint64_t* idx) {
    for (int d = 0; d < t->rank; ++d) {
        if (idx[d] >= t->shape[d]) return false;
    }
    return true;
}

// Check if two tensors have equal shapes
bool shapes_equal(const vsla_tensor_t* a, const vsla_tensor_t* b) {
    if (!a || !b) return false;
    if (a->rank != b->rank) return false;
    
    for (uint8_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}