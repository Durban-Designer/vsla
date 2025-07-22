/**
 * @file vsla_window.h
 * @brief VSLA window stacking structure definition
 * 
 * Internal header for window stacking implementation
 */

#ifndef VSLA_WINDOW_H
#define VSLA_WINDOW_H

#include "../vsla_core.h"
#include "../vsla_tensor.h"

/**
 * @brief Window structure for stacking operations (Section 5.2) - MODERN VERSION
 * 
 * Maintains ring buffer collecting w tensors then emitting S_w
 * Now includes context for proper tensor creation
 */
struct vsla_window_s {
    vsla_tensor_t** buf;        // Ring buffer of tensors
    size_t fill;                // Current fill level
    size_t window_size;         // Window size w
    uint8_t base_rank;          // Rank of input tensors
    vsla_dtype_t dtype;         // Data type of tensors
    vsla_context_t* ctx;        // Context for unified tensor creation
};

/**
 * @brief Pyramid structure for hierarchical stacking (Section 5.2)
 * 
 * Array of windows[L] that feed results recursively upward
 */
typedef struct vsla_pyramid_s {
    vsla_window_t** windows;    // Array of windows at each level
    size_t levels;              // Number of pyramid levels L
    size_t window_size;         // Window size for each level
    uint8_t base_rank;          // Rank of input tensors
    vsla_dtype_t dtype;         // Data type of tensors
    bool discard_partials;      // Flushing policy: true=discard, false=pad
} vsla_pyramid_t;

// Note: Implementation functions are internal - use vsla_unified.h API

#endif /* VSLA_WINDOW_H */