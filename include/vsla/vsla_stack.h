/**
 * @file vsla_stack.h
 * @brief VSLA Stacking Operator and Tensor Pyramid Construction
 * 
 * Implements the stacking operator Σ that turns a collection of rank-r tensors
 * into a rank-(r+1) tensor, and the window-stacking operator Ω for building
 * tensor pyramids from streaming data.
 * 
 * Mathematical Foundation:
 * - Σ_k : (𝕋_r)^k → 𝕋_{r+1} - Stacks k tensors along a new leading axis
 * - Ω_w : Stream(𝕋_r) → Stream(𝕋_{r+1}) - Sliding window with step w
 * 
 * Algebraic Properties:
 * - Associativity (nested levels)
 * - Neutral-zero absorption  
 * - Distributivity over +, ⊙
 * - Forms strict monoidal category (𝕋_r, +, Σ)
 * 
 * @copyright MIT License
 */

#ifndef VSLA_STACK_H
#define VSLA_STACK_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Stacking operator configuration
 */
typedef struct {
    bool copy_data;          /**< Copy data or create view (if possible) */
    int axis;               /**< Axis along which to stack (0 = new leading axis) */
    bool preserve_sparsity; /**< Preserve zero-structure when possible */
} vsla_stack_config_t;

/**
 * @brief Window accumulator state for streaming operations
 */
typedef struct vsla_window_state vsla_window_state_t;

/**
 * @brief Pyramid builder for recursive aggregation
 */
typedef struct vsla_pyramid_builder vsla_pyramid_builder_t;

/**
 * @brief Window statistics structure
 */
typedef struct {
    size_t current_count;
    size_t total_processed;
    size_t windows_emitted;
} vsla_window_stats_t;

// === Core Stacking Operations ===

/**
 * @brief Stacking operator Σ_k: (𝕋_r)^k → 𝕋_{r+1}
 * 
 * Stacks k tensors along a new leading axis. Automatically pads inputs to
 * common ambient shape, then concatenates along fresh axis.
 * 
 * Mathematical definition:
 * Given A^(1),...,A^(k) ∈ 𝕋_r, choose ambient shape 𝐧 = (max_i n_j^(i))
 * Result[i,𝐣] = A^(i)[𝐣] for 1≤i≤k, 𝐣≤𝐧, else 0
 * 
 * @param out Output tensor of rank r+1 (must be pre-allocated)
 * @param tensors Array of k input tensors (all rank r)
 * @param k Number of tensors to stack
 * @param config Stacking configuration (NULL for defaults)
 * @return VSLA_SUCCESS on success
 * 
 * @note Satisfies associativity, zero-absorption, distributivity
 * @note Time: Θ(N) if shapes equal, Θ(N + k·Δ) with padding
 * @note Space: Θ(k) pointer table + padding overhead
 * 
 * @code
 * // Stack 3 vectors into a matrix
 * vsla_tensor_t* vecs[3] = {v1, v2, v3};
 * vsla_tensor_t* matrix = vsla_zeros(2, (uint64_t[]){3, max_len}, 
 *                                    VSLA_MODEL_A, VSLA_DTYPE_F64);
 * vsla_stack(matrix, vecs, 3, NULL);
 * @endcode
 */
vsla_error_t vsla_stack(vsla_tensor_t* out,
                        vsla_tensor_t* const* tensors,
                        size_t k,
                        const vsla_stack_config_t* config);

/**
 * @brief Convenience function to create stacked tensor
 * 
 * Automatically determines output shape and allocates result tensor.
 * 
 * @param tensors Array of k input tensors  
 * @param k Number of tensors to stack
 * @param config Stacking configuration (NULL for defaults)
 * @return New stacked tensor or NULL on error
 */
vsla_tensor_t* vsla_stack_create(vsla_tensor_t* const* tensors,
                                 size_t k, 
                                 const vsla_stack_config_t* config);

/**
 * @brief Stack tensors along specified axis
 * 
 * Generalizes stacking to any axis position (not just leading).
 * For axis=0, equivalent to vsla_stack().
 * 
 * @param out Output tensor
 * @param tensors Input tensors
 * @param k Number of tensors
 * @param axis Axis position (0=leading, -1=trailing, etc.)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_stack_axis(vsla_tensor_t* out,
                             vsla_tensor_t* const* tensors,
                             size_t k,
                             int axis);

// === Window-Stacking Operations (Ω operator) ===

/**
 * @brief Create window accumulator for streaming data
 * 
 * Implements the window-stacking operator Ω_w that slides a window of 
 * length w with step w (non-overlapping) over a stream.
 * 
 * Mathematical definition:
 * Ω_w(X^(t))_s = Σ_w(X^(sw), ..., X^(sw+w-1)) ∈ 𝕋_{r+1}
 * 
 * @param window_size Window length w
 * @param template_tensor Example tensor for determining rank/type
 * @return New window state or NULL on error
 */
vsla_window_state_t* vsla_window_create(size_t window_size,
                                        const vsla_tensor_t* template_tensor);

/**
 * @brief Free window accumulator
 * 
 * @param state Window state to free
 */
void vsla_window_free(vsla_window_state_t* state);

/**
 * @brief Accumulate tensor in sliding window
 * 
 * Adds tensor to current window. When window is full (count == w),
 * emits a stacked tensor and resets for next window.
 * 
 * @param state Window accumulator state  
 * @param tensor New tensor to add
 * @param output Output stacked tensor (only valid when function returns true)
 * @return true if window is full and output is ready, false otherwise
 * 
 * @code
 * vsla_window_state_t* win = vsla_window_create(4, template);
 * vsla_tensor_t* result;
 * 
 * for (int t = 0; t < stream_length; t++) {
 *     if (vsla_window_accum(win, stream[t], &result)) {
 *         // Process batched result (rank increased by 1)
 *         process_batch(result);
 *         vsla_free(result);
 *     }
 * }
 * @endcode
 */
bool vsla_window_accum(vsla_window_state_t* state,
                       const vsla_tensor_t* tensor,
                       vsla_tensor_t** output);

/**
 * @brief Flush partial window
 * 
 * Forces output of current window even if not full (pads with zeros).
 * Useful for end-of-stream processing.
 * 
 * @param state Window state
 * @param output Output tensor (may be smaller than window_size)
 * @return true if any tensors were in buffer, false if empty
 */
bool vsla_window_flush(vsla_window_state_t* state,
                       vsla_tensor_t** output);

/**
 * @brief Get window accumulator statistics
 * 
 * @param state Window state  
 * @param current_count Output for current buffer size
 * @param total_processed Output for total tensors processed
 * @param windows_emitted Output for number of complete windows emitted
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_window_stats(const vsla_window_state_t* state,
                               size_t* current_count,
                               size_t* total_processed, 
                               size_t* windows_emitted);

// === Tensor Pyramid Construction ===

/**
 * @brief Create pyramid builder for recursive aggregation
 * 
 * Builds d-level tensor pyramids by composing Ω repeatedly:
 * 𝕋_r →^{Ω_{w₁}} 𝕋_{r+1} →^{Ω_{w₂}} 𝕋_{r+2} → ... →^{Ω_{wₐ}} 𝕋_{r+d}
 * 
 * @param levels Number of pyramid levels d
 * @param window_sizes Array of window sizes [w₁, w₂, ..., wₐ]
 * @param template_tensor Example tensor for base level
 * @return New pyramid builder or NULL on error
 */
vsla_pyramid_builder_t* vsla_pyramid_create(size_t levels,
                                             const size_t* window_sizes,
                                             const vsla_tensor_t* template_tensor);

/**
 * @brief Free pyramid builder
 * 
 * @param builder Pyramid builder to free
 */
void vsla_pyramid_free(vsla_pyramid_builder_t* builder);

/**
 * @brief Add tensor to pyramid base level
 * 
 * Feeds tensor into level 0, which may trigger cascading aggregations
 * up the pyramid as windows fill.
 * 
 * @param builder Pyramid builder
 * @param tensor Input tensor (rank r)
 * @param level_outputs Array to receive outputs from any completed levels
 * @param max_outputs Size of level_outputs array
 * @param num_outputs Number of outputs actually produced
 * @return VSLA_SUCCESS on success
 * 
 * @code
 * vsla_pyramid_builder_t* pyr = vsla_pyramid_create(3, 
 *     (size_t[]){4, 3, 2}, template);
 * 
 * vsla_tensor_t* outputs[3];
 * size_t count;
 * 
 * for (int t = 0; t < stream_length; t++) {
 *     vsla_pyramid_add(pyr, stream[t], outputs, 3, &count);
 *     for (size_t i = 0; i < count; i++) {
 *         printf("Level %zu output: rank %d\n", i, outputs[i]->rank);
 *         vsla_free(outputs[i]);
 *     }
 * }
 * @endcode
 */
vsla_error_t vsla_pyramid_add(vsla_pyramid_builder_t* builder,
                              const vsla_tensor_t* tensor,
                              vsla_tensor_t** level_outputs,
                              size_t max_outputs,
                              size_t* num_outputs);

/**
 * @brief Flush all pyramid levels
 * 
 * Forces output of partial windows at all levels. Useful for
 * end-of-stream processing.
 * 
 * @param builder Pyramid builder
 * @param level_outputs Array to receive outputs from all levels
 * @param max_outputs Size of level_outputs array  
 * @param num_outputs Number of outputs actually produced
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_pyramid_flush(vsla_pyramid_builder_t* builder,
                                vsla_tensor_t** level_outputs,
                                size_t max_outputs,
                                size_t* num_outputs);

/**
 * @brief Get pyramid builder statistics
 * 
 * @param builder Pyramid builder
 * @param level_stats Array of window stats for each level
 * @param num_levels Size of level_stats array
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_pyramid_stats(const vsla_pyramid_builder_t* builder,
                                vsla_window_stats_t* level_stats,
                                size_t num_levels);

// === Utility Functions ===

/**
 * @brief Compute common ambient shape for stacking
 * 
 * Determines the maximum extent along each axis that accommodates
 * all input tensors after zero-padding.
 * 
 * @param tensors Input tensors
 * @param k Number of tensors
 * @param ambient_shape Output shape (caller allocates)
 * @param rank Output rank (should be same for all inputs)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_stack_ambient_shape(vsla_tensor_t* const* tensors,
                                      size_t k,
                                      uint64_t* ambient_shape,
                                      uint8_t* rank);

/**
 * @brief Check if tensors can be stacked efficiently
 * 
 * Returns true if all tensors already share the same shape,
 * enabling O(1) view construction instead of O(N) copy.
 * 
 * @param tensors Input tensors
 * @param k Number of tensors
 * @return true if shapes are identical, false otherwise
 */
bool vsla_stack_shapes_compatible(vsla_tensor_t* const* tensors, size_t k);

/**
 * @brief Unstack tensor along specified axis
 * 
 * Inverse operation of stacking. Splits rank-(r+1) tensor into
 * k rank-r tensors along specified axis.
 * 
 * @param tensor Input tensor to unstack
 * @param axis Axis to split along (typically 0)
 * @param outputs Array to receive unstacked tensors
 * @param max_outputs Size of outputs array
 * @param num_outputs Actual number of tensors produced
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_unstack(const vsla_tensor_t* tensor,
                          int axis,
                          vsla_tensor_t** outputs,
                          size_t max_outputs,
                          size_t* num_outputs);

// === Default Configuration ===

/**
 * @brief Get default stacking configuration
 * 
 * @return Default configuration struct
 */
vsla_stack_config_t vsla_stack_default_config(void);

#ifdef __cplusplus
}
#endif

#endif // VSLA_STACK_H