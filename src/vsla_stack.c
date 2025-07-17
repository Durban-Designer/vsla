/**
 * @file vsla_stack.c
 * @brief Implementation of VSLA Stacking Operator and Tensor Pyramids
 * 
 * @copyright MIT License
 */

#include "vsla/vsla_stack.h"
#include "vsla/vsla_ops.h"
#include "vsla/vsla_core.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// === Window State Structure ===

struct vsla_window_state {
    size_t window_size;          // Window length w
    size_t current_count;        // Current buffer size
    size_t total_processed;      // Total tensors seen
    size_t windows_emitted;      // Complete windows emitted
    vsla_tensor_t** buffer;      // Tensor buffer
    uint8_t rank;               // Expected tensor rank
    vsla_model_t model;         // Expected tensor model
    vsla_dtype_t dtype;         // Expected tensor dtype
};

// === Pyramid Builder Structure ===

struct vsla_pyramid_builder {
    size_t num_levels;           // Number of pyramid levels
    vsla_window_state_t** levels; // Array of window states
    size_t* window_sizes;        // Window sizes for each level
};

// === Window Statistics Structure ===

typedef struct {
    size_t current_count;
    size_t total_processed;
    size_t windows_emitted;
} vsla_window_stats_t;

// === Default Configuration ===

vsla_stack_config_t vsla_stack_default_config(void) {
    vsla_stack_config_t config = {
        .copy_data = true,
        .axis = 0,
        .preserve_sparsity = true
    };
    return config;
}

// === Static Helper Functions ===

/**
 * @brief Recursively copy multi-dimensional block with zero-padding
 * 
 * @param dst_data Destination data pointer
 * @param src_data Source data pointer  
 * @param dst_shape Destination tensor shape
 * @param src_shape Source tensor shape
 * @param rank Number of dimensions
 * @param dtype_size Size of data type in bytes
 * @param block_idx Block index in stacked dimension
 * @param dim Current dimension being processed
 * @return VSLA_SUCCESS on success
 */
static vsla_error_t vsla_stack_copy_block(void* dst_data, 
                                          const void* src_data,
                                          const uint64_t* dst_shape,
                                          const uint64_t* src_shape,
                                          uint8_t rank,
                                          size_t dtype_size,
                                          size_t block_idx,
                                          uint8_t dim) {
    if (dim == rank) {
        // Base case: copy single element
        memcpy(dst_data, src_data, dtype_size);
        return VSLA_SUCCESS;
    }
    
    // Calculate strides for current dimension
    uint64_t dst_stride = dtype_size;
    uint64_t src_stride = dtype_size;
    
    // Calculate strides correctly - only iterate through remaining dimensions
    for (uint8_t i = dim + 1; i < rank; i++) {
        dst_stride *= dst_shape[i + 1]; // +1 because output has extra stacking dimension
        src_stride *= src_shape[i];
    }
    
    // Copy elements along current dimension
    uint64_t copy_count = src_shape[dim]; // Only copy up to source size
    
    for (uint64_t i = 0; i < copy_count; i++) {
        // Calculate offsets
        uint64_t dst_offset = i * dst_stride;
        uint64_t src_offset = i * src_stride;
        
        // Add block offset for stacking dimension (only at outermost level)
        if (dim == 0) {
            uint64_t block_stride = dtype_size;
            for (uint8_t j = 1; j <= rank; j++) { // Changed < to <= to include all dimensions
                block_stride *= dst_shape[j];
            }
            dst_offset += block_idx * block_stride;
        }
        
        // Recursive call for next dimension
        vsla_error_t err = vsla_stack_copy_block(
            (char*)dst_data + dst_offset,
            (const char*)src_data + src_offset,
            dst_shape, src_shape, rank, dtype_size,
            block_idx, dim + 1
        );
        
        if (err != VSLA_SUCCESS) {
            return err;
        }
    }
    
    return VSLA_SUCCESS;
}

// === Utility Functions ===

vsla_error_t vsla_stack_ambient_shape(vsla_tensor_t* const* tensors,
                                      size_t k,
                                      uint64_t* ambient_shape,
                                      uint8_t* rank) {
    if (!tensors || k == 0 || !ambient_shape || !rank) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Get rank from first tensor
    *rank = tensors[0]->rank;
    
    // Initialize ambient shape with first tensor
    for (uint8_t i = 0; i < *rank; i++) {
        ambient_shape[i] = tensors[0]->shape[i];
    }
    
    // Find maximum extent along each axis
    for (size_t t = 1; t < k; t++) {
        if (tensors[t]->rank != *rank) {
            return VSLA_ERROR_SHAPE_MISMATCH;
        }
        
        for (uint8_t i = 0; i < *rank; i++) {
            if (tensors[t]->shape[i] > ambient_shape[i]) {
                ambient_shape[i] = tensors[t]->shape[i];
            }
        }
    }
    
    return VSLA_SUCCESS;
}

bool vsla_stack_shapes_compatible(vsla_tensor_t* const* tensors, size_t k) {
    if (!tensors || k <= 1) return true;
    
    uint8_t rank = tensors[0]->rank;
    
    for (size_t t = 1; t < k; t++) {
        if (tensors[t]->rank != rank) return false;
        
        for (uint8_t i = 0; i < rank; i++) {
            if (tensors[t]->shape[i] != tensors[0]->shape[i]) {
                return false;
            }
        }
    }
    
    return true;
}

// === Core Stacking Implementation ===

vsla_error_t vsla_stack(vsla_tensor_t* out,
                        vsla_tensor_t* const* tensors,
                        size_t k,
                        const vsla_stack_config_t* config) {
    if (!out || !tensors || k == 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    // Use default config if none provided
    vsla_stack_config_t default_config = vsla_stack_default_config();
    if (!config) config = &default_config;
    
    // Validate inputs have same rank and model
    uint8_t rank = tensors[0]->rank;
    vsla_model_t model = tensors[0]->model;
    vsla_dtype_t dtype = tensors[0]->dtype;
    
    for (size_t i = 1; i < k; i++) {
        if (tensors[i]->rank != rank || 
            tensors[i]->model != model ||
            tensors[i]->dtype != dtype) {
            return VSLA_ERROR_SHAPE_MISMATCH;
        }
    }
    
    // Validate output tensor
    if (out->rank != rank + 1 || out->model != model || out->dtype != dtype) {
        return VSLA_ERROR_SHAPE_MISMATCH;
    }
    
    // Compute ambient shape
    uint64_t ambient_shape[VSLA_MAX_RANK];
    uint8_t input_rank;
    vsla_error_t err = vsla_stack_ambient_shape(tensors, k, ambient_shape, &input_rank);
    if (err != VSLA_SUCCESS) return err;
    
    // Verify output shape: [k, ambient_shape...]
    if (out->shape[0] < k) {
        return VSLA_ERROR_SHAPE_MISMATCH;
    }
    for (uint8_t i = 0; i < rank; i++) {
        if (out->shape[i + 1] < ambient_shape[i]) {
            return VSLA_ERROR_SHAPE_MISMATCH;
        }
    }
    
    // Fill output tensor with zeros first
    vsla_fill(out, 0.0);
    
    // Copy each input tensor to appropriate block
    size_t dtype_size = vsla_dtype_size(dtype);
    
    for (size_t t = 0; t < k; t++) {
        const vsla_tensor_t* src = tensors[t];
        
        // Calculate strides for block copying
        uint64_t src_total = 1;
        for (uint8_t i = 0; i < rank; i++) {
            src_total *= src->shape[i];
        }
        
        if (src_total == 0) continue; // Skip empty tensors
        
        // Get data pointers
        const void* src_data = vsla_tensor_data(src, NULL);
        void* out_data = vsla_tensor_data_mut(out, NULL);
        
        if (!src_data || !out_data) {
            return VSLA_ERROR_MEMORY;
        }
        
        // Calculate output offset for this tensor block
        uint64_t out_block_offset = 0;
        uint64_t out_stride = 1;
        
        // Skip to block t in leading dimension
        out_block_offset = t;
        for (uint8_t i = 0; i < rank; i++) {
            out_stride *= out->shape[i + 1];
        }
        out_block_offset *= out_stride;
        
        // For tensors with same shape as ambient, can do single memcpy
        if (vsla_stack_shapes_compatible((vsla_tensor_t**)&src, 1)) {
            bool same_as_ambient = true;
            for (uint8_t i = 0; i < rank; i++) {
                if (src->shape[i] != ambient_shape[i]) {
                    same_as_ambient = false;
                    break;
                }
            }
            
            if (same_as_ambient) {
                char* dst = (char*)out_data + out_block_offset * dtype_size;
                memcpy(dst, src_data, src_total * dtype_size);
                continue;
            }
        }
        
        // Need element-wise copy with padding
        // Implement recursive multi-dimensional block copying
        if (rank == 1) {
            char* dst = (char*)out_data + out_block_offset * dtype_size;
            memcpy(dst, src_data, src->shape[0] * dtype_size);
            // Padding with zeros already done by vsla_fill above
        } else {
            // General multi-dimensional block copying with recursive approach
            vsla_error_t copy_err = vsla_stack_copy_block(
                out_data, src_data, 
                out->shape, src->shape, 
                rank, dtype_size, 
                t, 0  // block_index=t, current_dim=0
            );
            if (copy_err != VSLA_SUCCESS) {
                return copy_err;
            }
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_tensor_t* vsla_stack_create(vsla_tensor_t* const* tensors,
                                 size_t k,
                                 const vsla_stack_config_t* config) {
    if (!tensors || k == 0) return NULL;
    
    // Compute output shape
    uint64_t ambient_shape[VSLA_MAX_RANK];
    uint8_t rank;
    if (vsla_stack_ambient_shape(tensors, k, ambient_shape, &rank) != VSLA_SUCCESS) {
        return NULL;
    }
    
    // Create output shape: [k, ambient_shape...]
    uint64_t out_shape[VSLA_MAX_RANK + 1];
    out_shape[0] = k;
    for (uint8_t i = 0; i < rank; i++) {
        out_shape[i + 1] = ambient_shape[i];
    }
    
    // Create output tensor
    vsla_tensor_t* result = vsla_new(rank + 1, out_shape, 
                                     tensors[0]->model, tensors[0]->dtype);
    if (!result) return NULL;
    
    // Perform stacking
    if (vsla_stack(result, tensors, k, config) != VSLA_SUCCESS) {
        vsla_free(result);
        return NULL;
    }
    
    return result;
}

vsla_error_t vsla_stack_axis(vsla_tensor_t* out,
                             vsla_tensor_t* const* tensors,
                             size_t k,
                             int axis) {
    // For now, only support axis = 0 (leading axis stacking)
    if (axis != 0) {
        return VSLA_ERROR_NOT_IMPLEMENTED;
    }
    
    return vsla_stack(out, tensors, k, NULL);
}

// === Window-Stacking Implementation ===

vsla_window_state_t* vsla_window_create(size_t window_size,
                                        const vsla_tensor_t* template_tensor) {
    if (window_size == 0 || !template_tensor) return NULL;
    
    vsla_window_state_t* state = calloc(1, sizeof(vsla_window_state_t));
    if (!state) return NULL;
    
    state->window_size = window_size;
    state->current_count = 0;
    state->total_processed = 0;
    state->windows_emitted = 0;
    state->rank = template_tensor->rank;
    state->model = template_tensor->model;
    state->dtype = template_tensor->dtype;
    
    // Allocate buffer for window_size tensors
    state->buffer = calloc(window_size, sizeof(vsla_tensor_t*));
    if (!state->buffer) {
        free(state);
        return NULL;
    }
    
    return state;
}

void vsla_window_free(vsla_window_state_t* state) {
    if (!state) return;
    
    // Free any tensors remaining in buffer
    for (size_t i = 0; i < state->current_count; i++) {
        if (state->buffer[i]) {
            vsla_free(state->buffer[i]);
        }
    }
    
    free(state->buffer);
    free(state);
}

bool vsla_window_accum(vsla_window_state_t* state,
                       const vsla_tensor_t* tensor,
                       vsla_tensor_t** output) {
    if (!state || !tensor || !output) return false;
    
    // Validate tensor matches expected type
    if (tensor->rank != state->rank || 
        tensor->model != state->model ||
        tensor->dtype != state->dtype) {
        return false;
    }
    
    // Copy tensor into buffer (take ownership)
    state->buffer[state->current_count] = vsla_copy(tensor);
    if (!state->buffer[state->current_count]) return false;
    
    state->current_count++;
    state->total_processed++;
    
    // Check if window is full
    if (state->current_count >= state->window_size) {
        // Create stacked tensor
        *output = vsla_stack_create(state->buffer, state->window_size, NULL);
        if (!*output) return false;
        
        // Free buffer tensors and reset
        for (size_t i = 0; i < state->window_size; i++) {
            vsla_free(state->buffer[i]);
            state->buffer[i] = NULL;
        }
        
        state->current_count = 0;
        state->windows_emitted++;
        
        return true;
    }
    
    return false;
}

bool vsla_window_flush(vsla_window_state_t* state,
                       vsla_tensor_t** output) {
    if (!state || !output || state->current_count == 0) return false;
    
    // Create stacked tensor from partial window
    *output = vsla_stack_create(state->buffer, state->current_count, NULL);
    if (!*output) return false;
    
    // Free buffer tensors and reset
    for (size_t i = 0; i < state->current_count; i++) {
        vsla_free(state->buffer[i]);
        state->buffer[i] = NULL;
    }
    
    state->current_count = 0;
    
    return true;
}

vsla_error_t vsla_window_stats(const vsla_window_state_t* state,
                               size_t* current_count,
                               size_t* total_processed,
                               size_t* windows_emitted) {
    if (!state) return VSLA_ERROR_INVALID_ARGUMENT;
    
    if (current_count) *current_count = state->current_count;
    if (total_processed) *total_processed = state->total_processed;
    if (windows_emitted) *windows_emitted = state->windows_emitted;
    
    return VSLA_SUCCESS;
}

// === Pyramid Builder Implementation ===

vsla_pyramid_builder_t* vsla_pyramid_create(size_t levels,
                                             const size_t* window_sizes,
                                             const vsla_tensor_t* template_tensor) {
    if (levels == 0 || !window_sizes || !template_tensor) return NULL;
    
    vsla_pyramid_builder_t* builder = calloc(1, sizeof(vsla_pyramid_builder_t));
    if (!builder) return NULL;
    
    builder->num_levels = levels;
    
    // Allocate arrays
    builder->levels = calloc(levels, sizeof(vsla_window_state_t*));
    builder->window_sizes = calloc(levels, sizeof(size_t));
    
    if (!builder->levels || !builder->window_sizes) {
        vsla_pyramid_free(builder);
        return NULL;
    }
    
    // Copy window sizes
    memcpy(builder->window_sizes, window_sizes, levels * sizeof(size_t));
    
    // Create window states for each level
    const vsla_tensor_t* level_template = template_tensor;
    
    for (size_t i = 0; i < levels; i++) {
        builder->levels[i] = vsla_window_create(window_sizes[i], level_template);
        if (!builder->levels[i]) {
            vsla_pyramid_free(builder);
            return NULL;
        }
        
        // Template for next level has rank increased by 1
        // (This is conceptual - we would need to create an actual template)
        // For now, we'll handle this in the add function
    }
    
    return builder;
}

void vsla_pyramid_free(vsla_pyramid_builder_t* builder) {
    if (!builder) return;
    
    if (builder->levels) {
        for (size_t i = 0; i < builder->num_levels; i++) {
            if (builder->levels[i]) {
                vsla_window_free(builder->levels[i]);
            }
        }
        free(builder->levels);
    }
    
    free(builder->window_sizes);
    free(builder);
}

vsla_error_t vsla_pyramid_add(vsla_pyramid_builder_t* builder,
                              const vsla_tensor_t* tensor,
                              vsla_tensor_t** level_outputs,
                              size_t max_outputs,
                              size_t* num_outputs) {
    if (!builder || !tensor || !level_outputs || !num_outputs) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    *num_outputs = 0;
    const vsla_tensor_t* current_input = tensor;
    
    // Process through each level
    for (size_t level = 0; level < builder->num_levels && level < max_outputs; level++) {
        vsla_tensor_t* level_output = NULL;
        
        bool window_complete = vsla_window_accum(builder->levels[level], 
                                                 current_input, &level_output);
        
        if (window_complete) {
            level_outputs[*num_outputs] = level_output;
            (*num_outputs)++;
            
            // Output from this level becomes input to next level
            current_input = level_output;
        } else {
            // Window not complete, stop cascade
            break;
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_pyramid_flush(vsla_pyramid_builder_t* builder,
                                vsla_tensor_t** level_outputs,
                                size_t max_outputs,
                                size_t* num_outputs) {
    if (!builder || !level_outputs || !num_outputs) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    *num_outputs = 0;
    
    // Flush each level that has partial data
    for (size_t level = 0; level < builder->num_levels && level < max_outputs; level++) {
        vsla_tensor_t* level_output = NULL;
        
        if (vsla_window_flush(builder->levels[level], &level_output)) {
            level_outputs[*num_outputs] = level_output;
            (*num_outputs)++;
        }
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_pyramid_stats(const vsla_pyramid_builder_t* builder,
                                vsla_window_stats_t* level_stats,
                                size_t num_levels) {
    if (!builder || !level_stats) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    size_t levels_to_report = (num_levels < builder->num_levels) ? 
                              num_levels : builder->num_levels;
    
    for (size_t i = 0; i < levels_to_report; i++) {
        vsla_window_stats(builder->levels[i],
                          &level_stats[i].current_count,
                          &level_stats[i].total_processed,
                          &level_stats[i].windows_emitted);
    }
    
    return VSLA_SUCCESS;
}

// === Unstack Implementation ===

vsla_error_t vsla_unstack(const vsla_tensor_t* tensor,
                          int axis,
                          vsla_tensor_t** outputs,
                          size_t max_outputs,
                          size_t* num_outputs) {
    if (!tensor || !outputs || !num_outputs || axis != 0) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    
    if (tensor->rank == 0) {
        return VSLA_ERROR_SHAPE_MISMATCH;
    }
    
    size_t split_count = tensor->shape[0];
    if (split_count > max_outputs) {
        split_count = max_outputs;
    }
    
    *num_outputs = split_count;
    
    // Create output shape (remove leading dimension)
    uint64_t output_shape[VSLA_MAX_RANK];
    for (uint8_t i = 1; i < tensor->rank; i++) {
        output_shape[i - 1] = tensor->shape[i];
    }
    
    // Extract each slice along axis 0
    for (size_t i = 0; i < split_count; i++) {
        outputs[i] = vsla_new(tensor->rank - 1, output_shape, 
                              tensor->model, tensor->dtype);
        if (!outputs[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                vsla_free(outputs[j]);
            }
            return VSLA_ERROR_MEMORY;
        }
        
        // Copy data slice (simplified for 1D case)
        if (tensor->rank == 2) {
            const void* src_data = vsla_tensor_data(tensor, NULL);
            void* dst_data = vsla_tensor_data_mut(outputs[i], NULL);
            
            size_t dtype_size = vsla_dtype_size(tensor->dtype);
            size_t slice_size = tensor->shape[1] * dtype_size;
            size_t src_offset = i * slice_size;
            
            memcpy(dst_data, (const char*)src_data + src_offset, slice_size);
        } else {
            // TODO: Implement general multi-dimensional unstacking
            vsla_free(outputs[i]);
            return VSLA_ERROR_NOT_IMPLEMENTED;
        }
    }
    
    return VSLA_SUCCESS;
}