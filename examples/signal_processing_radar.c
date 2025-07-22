/**
 * @file signal_processing_radar.c
 * @brief E2E Radar Signal Processing with Variable-Length Signals using the new API.
 *
 * This example demonstrates real-world radar signal processing using VSLA's
 * hardware-agnostic interface.
 */

#include "vsla/vsla.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#define BASE_PULSE_LENGTH 128
#define MAX_SIGNAL_LENGTH 2048

// Generate a radar pulse
static vsla_tensor_t* generate_radar_pulse(vsla_context_t* ctx, int pulse_length) {
    uint64_t shape[] = {pulse_length};
    vsla_tensor_t* pulse = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F32);
    // In a real application, you would generate a chirp waveform here.
    vsla_fill(ctx, pulse, 1.0f);
    return pulse;
}

// Matched filter processing using VSLA convolution
static vsla_tensor_t* matched_filter(vsla_context_t* ctx, vsla_tensor_t* signal, vsla_tensor_t* pulse) {
    const uint64_t* signal_shape = vsla_tensor_get_shape(signal);
    const uint64_t* pulse_shape = vsla_tensor_get_shape(pulse);
    uint64_t output_shape[] = {signal_shape[0] + pulse_shape[0] - 1};

    vsla_tensor_t* matched_output = vsla_tensor_create(ctx, 1, output_shape, VSLA_MODEL_A, VSLA_DTYPE_F32);

    // In a real application, you would use the conjugate of the pulse.
    vsla_conv(ctx, matched_output, signal, pulse);

    return matched_output;
}

int main(void) {
    printf("=== VSLA Radar Signal Processing Example (New API) ===\n");

    // Initialize VSLA context
    vsla_config_t config = { .backend_selection = VSLA_BACKEND_AUTO };
    vsla_context_t* ctx = vsla_init(&config);
    if (!ctx) {
        printf("Failed to initialize VSLA context\n");
        return 1;
    }

    // Generate a reference pulse and a received signal
    vsla_tensor_t* pulse = generate_radar_pulse(ctx, BASE_PULSE_LENGTH);
    vsla_tensor_t* signal = generate_radar_pulse(ctx, MAX_SIGNAL_LENGTH);

    // Perform matched filter processing
    vsla_tensor_t* matched_output = matched_filter(ctx, signal, pulse);

    if (matched_output) {
        const uint64_t* output_shape = vsla_tensor_get_shape(matched_output);
        printf("Matched filter output shape: [%lu]\n", output_shape[0]);
    }

    // Clean up
    vsla_tensor_free(pulse);
    vsla_tensor_free(signal);
    vsla_tensor_free(matched_output);
    vsla_cleanup(ctx);

    printf("\n✓ Radar signal processing example completed successfully!\n");

    return 0;
}
