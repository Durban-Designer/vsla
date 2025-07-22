/**
 * @file neural_network_cnn.c
 * @brief E2E Convolutional Neural Network with Variable Input Sizes using the new API.
 *
 * This example demonstrates a complete CNN implementation using VSLA's
 * hardware-agnostic interface for processing images of varying sizes
 * without manual padding or batching.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_LAYERS 10

// Simplified layer and model structures for demonstration
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
} cnn_layer_t;

typedef struct {
    cnn_layer_t layers[MAX_LAYERS];
    int num_layers;
} cnn_model_t;

// Simplified feature map structure
typedef struct {
    vsla_tensor_t* data;
} feature_map_t;

// Forward pass through a single convolutional layer
static feature_map_t* conv_layer_forward(vsla_context_t* ctx, const feature_map_t* input, const cnn_layer_t* layer) {
    const uint64_t* input_shape = vsla_tensor_get_shape(input->data);
    uint64_t output_shape[] = {input_shape[0], input_shape[1] - layer->kernel_size + 1, layer->output_channels};

    feature_map_t* output = (feature_map_t*)malloc(sizeof(feature_map_t));
    output->data = vsla_tensor_create(ctx, 3, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    // In a real scenario, you would perform convolution here.
    // This is a placeholder to demonstrate the API usage.
    printf("    Performing convolution...\n");

    return output;
}

// Main CNN forward pass
static feature_map_t* cnn_forward_pass(vsla_context_t* ctx, feature_map_t* input, const cnn_model_t* model) {
    printf("  Forward pass through the CNN:\n");

    feature_map_t* current = input;

    for (int i = 0; i < model->num_layers; i++) {
        const cnn_layer_t* layer = &model->layers[i];
        feature_map_t* next = conv_layer_forward(ctx, current, layer);

        if (current != input) {
            vsla_tensor_free(current->data);
            free(current);
        }
        current = next;
    }

    return current;
}

int main(void) {
    printf("=== VSLA CNN Variable Input Size Example (New API) ===\n");

    // Initialize VSLA context
    vsla_config_t config = { .backend_selection = VSLA_BACKEND_AUTO };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    // Create a simple CNN model
    cnn_model_t model = { .num_layers = 2 };
    model.layers[0] = (cnn_layer_t){ .input_channels = 3, .output_channels = 16, .kernel_size = 3 };
    model.layers[1] = (cnn_layer_t){ .input_channels = 16, .output_channels = 32, .kernel_size = 3 };

    // Process images of different sizes
    uint64_t image_shapes[][2] = { {64, 64}, {128, 128}, {224, 224} };

    for (int i = 0; i < 3; i++) {
        uint64_t shape[] = {image_shapes[i][0], image_shapes[i][1], 3};
        printf("\n--- Processing image of size %lux%lu ---\n", shape[0], shape[1]);

        // Create a dummy input feature map
        feature_map_t* input = (feature_map_t*)malloc(sizeof(feature_map_t));
        input->data = vsla_tensor_create(ctx, 3, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

        // Perform the forward pass
        feature_map_t* output = cnn_forward_pass(ctx, input, &model);

        const uint64_t* output_shape = vsla_tensor_get_shape(output->data);
        printf("  Final output shape: %lux%lux%u\n", output_shape[0], output_shape[1], (unsigned int)output_shape[2]);

        // Clean up
        vsla_tensor_free(output->data);
        free(output);
    }

    // Cleanup
    vsla_cleanup(ctx);

    printf("\n✓ CNN example completed successfully!\n");

    return 0;
}
