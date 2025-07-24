/**
 * @file vsla_cpu_fft.c
 * @brief VSLA CPU FFT implementation following v3.2 specification
 * 
 * Implements radix-2 iterative Cooley-Tukey FFT for convolution operations.
 * No external dependencies - pure C implementation respecting VSLA semantics.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

// Complex number type (avoiding C99 complex.h for portability)
typedef struct {
    double re;
    double im;
} c64;

// FFT plan structure
typedef struct vsla_fft_plan {
    size_t L;                    // FFT length (power of 2)
    c64* twiddles;              // Twiddle factors, length L/2
    uint32_t* bitrev;           // Bit reversal indices, length L
    struct vsla_fft_plan* next; // For LRU cache linked list
} vsla_fft_plan_t;

// Global FFT plan cache (simple implementation, thread-safety for later)
static vsla_fft_plan_t* plan_cache = NULL;
static const size_t MAX_CACHED_PLANS = 16;

// Helper: compute next power of 2
static inline size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// Helper: count trailing zeros (for log2 of power of 2)
static inline int log2_pow2(size_t n) {
    int count = 0;
    while ((n & 1) == 0) {
        n >>= 1;
        count++;
    }
    return count;
}

// Helper: bit reversal
static uint32_t bit_reverse(uint32_t x, int log2n) {
    uint32_t result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Create FFT plan
static vsla_fft_plan_t* create_fft_plan(size_t L) {
    // Verify L is power of 2
    if (L == 0 || (L & (L - 1)) != 0) {
        return NULL;
    }
    
    vsla_fft_plan_t* plan = (vsla_fft_plan_t*)calloc(1, sizeof(vsla_fft_plan_t));
    if (!plan) return NULL;
    
    plan->L = L;
    
    // Allocate twiddle factors
    plan->twiddles = (c64*)calloc(L / 2, sizeof(c64));
    if (!plan->twiddles) {
        free(plan);
        return NULL;
    }
    
    // Compute twiddle factors W_L^k = exp(-2πi·k/L)
    const double pi = 3.14159265358979323846;
    for (size_t k = 0; k < L / 2; k++) {
        double angle = -2.0 * pi * k / L;
        plan->twiddles[k].re = cos(angle);
        plan->twiddles[k].im = sin(angle);
    }
    
    // Allocate and compute bit reversal indices
    plan->bitrev = (uint32_t*)calloc(L, sizeof(uint32_t));
    if (!plan->bitrev) {
        free(plan->twiddles);
        free(plan);
        return NULL;
    }
    
    int log2L = log2_pow2(L);
    for (size_t i = 0; i < L; i++) {
        plan->bitrev[i] = bit_reverse(i, log2L);
    }
    
    return plan;
}

// Destroy FFT plan
static void destroy_fft_plan(vsla_fft_plan_t* plan) {
    if (!plan) return;
    free(plan->twiddles);
    free(plan->bitrev);
    free(plan);
}

// Get or create FFT plan from cache
static vsla_fft_plan_t* plan_get_or_make(size_t L) {
    // Search in cache
    vsla_fft_plan_t* prev = NULL;
    vsla_fft_plan_t* curr = plan_cache;
    
    while (curr) {
        if (curr->L == L) {
            // Move to front (LRU)
            if (prev) {
                prev->next = curr->next;
                curr->next = plan_cache;
                plan_cache = curr;
            }
            return curr;
        }
        prev = curr;
        curr = curr->next;
    }
    
    // Not found, create new plan
    vsla_fft_plan_t* new_plan = create_fft_plan(L);
    if (!new_plan) return NULL;
    
    // Add to cache front
    new_plan->next = plan_cache;
    plan_cache = new_plan;
    
    // Evict oldest if cache too large
    size_t count = 0;
    curr = plan_cache;
    while (curr && count < MAX_CACHED_PLANS - 1) {
        count++;
        prev = curr;
        curr = curr->next;
    }
    if (curr) {
        prev->next = NULL;
        while (curr) {
            vsla_fft_plan_t* next = curr->next;
            destroy_fft_plan(curr);
            curr = next;
        }
    }
    
    return new_plan;
}

// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
static inline c64 c64_mul(c64 a, c64 b) {
    c64 result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

// Complex addition
static inline c64 c64_add(c64 a, c64 b) {
    c64 result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    return result;
}

// Complex subtraction
static inline c64 c64_sub(c64 a, c64 b) {
    c64 result;
    result.re = a.re - b.re;
    result.im = a.im - b.im;
    return result;
}

// Forward FFT (in-place)
static void fft_forward_inplace(c64* data, const vsla_fft_plan_t* plan) {
    size_t L = plan->L;
    
    // Bit reversal reordering
    for (size_t i = 0; i < L; i++) {
        uint32_t j = plan->bitrev[i];
        if (i < j) {
            c64 temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    // Cooley-Tukey decimation-in-time
    int log2L = log2_pow2(L);
    
    for (int s = 1; s <= log2L; s++) {
        size_t m = 1UL << s;        // Current FFT size
        size_t half_m = m >> 1;     // Half of current size
        
        // Process each FFT of size m
        for (size_t k = 0; k < L; k += m) {
            // Butterfly operations
            for (size_t j = 0; j < half_m; j++) {
                // Twiddle factor index
                size_t twiddle_idx = j * (L / m);
                c64 w = plan->twiddles[twiddle_idx];
                
                // Butterfly
                c64 t = c64_mul(w, data[k + j + half_m]);
                c64 u = data[k + j];
                
                data[k + j] = c64_add(u, t);
                data[k + j + half_m] = c64_sub(u, t);
            }
        }
    }
}

// Inverse FFT (in-place) - same as forward but with conjugated twiddles and 1/L scaling
static void fft_inverse_inplace(c64* data, const vsla_fft_plan_t* plan) {
    size_t L = plan->L;
    
    // Bit reversal reordering
    for (size_t i = 0; i < L; i++) {
        uint32_t j = plan->bitrev[i];
        if (i < j) {
            c64 temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    // Cooley-Tukey with conjugated twiddles
    int log2L = log2_pow2(L);
    
    for (int s = 1; s <= log2L; s++) {
        size_t m = 1UL << s;
        size_t half_m = m >> 1;
        
        for (size_t k = 0; k < L; k += m) {
            for (size_t j = 0; j < half_m; j++) {
                size_t twiddle_idx = j * (L / m);
                c64 w = plan->twiddles[twiddle_idx];
                // Conjugate for inverse
                w.im = -w.im;
                
                c64 t = c64_mul(w, data[k + j + half_m]);
                c64 u = data[k + j];
                
                data[k + j] = c64_add(u, t);
                data[k + j + half_m] = c64_sub(u, t);
            }
        }
    }
    
    // Scale by 1/L
    double scale = 1.0 / L;
    for (size_t i = 0; i < L; i++) {
        data[i].re *= scale;
        data[i].im *= scale;
    }
}

// VSLA-aware convolution using FFT
int vsla_conv_fft(double* out, const double* A, size_t m, const double* B, size_t n) {
    // Empty operand check
    if (m == 0 || n == 0) {
        return 0; // Success - empty result
    }
    
    // Compute FFT length
    size_t out_len = m + n - 1;
    size_t L = next_pow2(out_len);
    
    // Get or create FFT plan
    vsla_fft_plan_t* plan = plan_get_or_make(L);
    if (!plan) {
        return -1; // Allocation failure
    }
    
    // Allocate temporary complex arrays
    c64* fa = (c64*)calloc(L, sizeof(c64));
    c64* fb = (c64*)calloc(L, sizeof(c64));
    if (!fa || !fb) {
        free(fa);
        free(fb);
        return -1;
    }
    
    // Load and zero-extend inputs
    for (size_t i = 0; i < L; i++) {
        fa[i].re = (i < m) ? A[i] : 0.0;
        fa[i].im = 0.0;
        fb[i].re = (i < n) ? B[i] : 0.0;
        fb[i].im = 0.0;
    }
    
    // Forward FFT
    fft_forward_inplace(fa, plan);
    fft_forward_inplace(fb, plan);
    
    // Pointwise multiplication in frequency domain
    for (size_t i = 0; i < L; i++) {
        c64 product = c64_mul(fa[i], fb[i]);
        fa[i] = product;
    }
    
    // Inverse FFT
    fft_inverse_inplace(fa, plan);
    
    // Copy real parts to output
    for (size_t k = 0; k < out_len; k++) {
        out[k] = fa[k].re;
    }
    
    // Optional: magnitude-based shrink heuristic
    // Find max magnitude
    double max_mag = 0.0;
    for (size_t k = 0; k < out_len; k++) {
        double mag = fabs(out[k]);
        if (mag > max_mag) max_mag = mag;
    }
    
    // Zero out small coefficients
    const double threshold = 32.0 * 2.220446049250313e-16 * max_mag; // 32 * DBL_EPSILON
    for (size_t k = 0; k < out_len; k++) {
        if (fabs(out[k]) < threshold) {
            out[k] = 0.0;
        }
    }
    
    // Clean up
    free(fa);
    free(fb);
    
    return 0; // Success
}

// Clean up global plan cache (call at program exit)
void vsla_fft_cleanup(void) {
    vsla_fft_plan_t* curr = plan_cache;
    while (curr) {
        vsla_fft_plan_t* next = curr->next;
        destroy_fft_plan(curr);
        curr = next;
    }
    plan_cache = NULL;
}