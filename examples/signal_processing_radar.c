/**
 * @file signal_processing_radar.c
 * @brief E2E Radar Signal Processing with Variable-Length Signals
 * 
 * This example demonstrates real-world radar signal processing using VSLA's
 * hardware-agnostic interface. It processes radar returns of varying lengths
 * using automatic convolution algorithm selection and hardware optimization.
 * 
 * Use Case: Multi-target radar with varying range gates and pulse lengths
 * Problem: Traditional frameworks require manual padding/reshaping
 * VSLA Solution: Automatic variable-shape handling with optimal performance
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

#define MAX_TARGETS 10
#define BASE_PULSE_LENGTH 128
#define MAX_SIGNAL_LENGTH 2048
#define SAMPLE_RATE 10e6  // 10 MHz
#define LIGHT_SPEED 3e8

// Radar target structure
typedef struct {
    double range_km;        // Target range in km
    double velocity_mps;    // Target velocity in m/s
    double rcs_dbsm;        // Radar cross section in dBsm
    double azimuth_deg;     // Azimuth angle in degrees
    int pulse_length;       // Variable pulse length
} radar_target_t;

// Radar system parameters
typedef struct {
    double carrier_freq_hz;     // Carrier frequency
    double bandwidth_hz;        // Signal bandwidth
    double pulse_duration_s;    // Pulse duration
    double prf_hz;             // Pulse repetition frequency
    int num_pulses;            // Number of pulses in CPI
} radar_params_t;

// Generate radar pulse with variable length
static vsla_unified_tensor_t* generate_radar_pulse(vsla_unified_context_t* ctx,
                                                    int pulse_length,
                                                    double bandwidth,
                                                    double sample_rate) {
    printf("  Generating %d-sample radar pulse (BW=%.1f MHz)\n", 
           pulse_length, bandwidth/1e6);
    
    uint64_t shape[] = {pulse_length};
    vsla_unified_tensor_t* pulse = vsla_tensor_create(ctx, 1, shape, 
                                                       VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (!pulse) return NULL;
    
    // Generate linear chirp (LFM waveform)
    float complex* data = (float complex*)vsla_tensor_data_mut(pulse, NULL);
    
    double chirp_rate = bandwidth / (pulse_length / sample_rate);
    
    for (int i = 0; i < pulse_length; i++) {
        double t = i / sample_rate;
        double phase = 2.0 * M_PI * chirp_rate * t * t / 2.0;
        
        // Apply Hann window to reduce sidelobes
        double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (pulse_length - 1)));
        
        data[i] = window * (cos(phase) + I * sin(phase));
    }
    
    return pulse;
}

// Generate radar return signal with multiple targets
static vsla_unified_tensor_t* generate_radar_return(vsla_unified_context_t* ctx,
                                                     const radar_target_t* targets,
                                                     int num_targets,
                                                     const radar_params_t* params,
                                                     int signal_length) {
    printf("  Generating radar return with %d targets (length=%d)\n", 
           num_targets, signal_length);
    
    uint64_t shape[] = {signal_length};
    vsla_unified_tensor_t* signal = vsla_tensor_zeros(ctx, 1, shape, 
                                                       VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (!signal) return NULL;
    
    float complex* data = (float complex*)vsla_tensor_data_mut(signal, NULL);
    
    // Add target returns
    for (int t = 0; t < num_targets; t++) {
        const radar_target_t* target = &targets[t];
        
        // Calculate time delay based on range
        double round_trip_time = 2.0 * target->range_km * 1000.0 / LIGHT_SPEED;
        int delay_samples = (int)(round_trip_time * SAMPLE_RATE);
        
        // Calculate Doppler shift
        double doppler_shift = 2.0 * target->velocity_mps * params->carrier_freq_hz / LIGHT_SPEED;
        
        // Calculate signal amplitude from RCS
        double amplitude = pow(10.0, target->rcs_dbsm / 20.0) / (target->range_km * target->range_km);
        
        // Generate target pulse with variable length
        int target_pulse_length = target->pulse_length;
        
        printf("    Target %d: Range=%.1fkm, Velocity=%.1fm/s, RCS=%.1fdBsm, PulseLen=%d\n",
               t, target->range_km, target->velocity_mps, target->rcs_dbsm, target_pulse_length);
        
        // Add target return to signal
        for (int i = 0; i < target_pulse_length && (delay_samples + i) < signal_length; i++) {
            double t_sample = i / SAMPLE_RATE;
            double phase = 2.0 * M_PI * doppler_shift * t_sample;
            
            // Simple target model with phase shift
            float complex target_sample = amplitude * (cos(phase) + I * sin(phase));
            
            data[delay_samples + i] += target_sample;
        }
    }
    
    // Add noise
    for (int i = 0; i < signal_length; i++) {
        double noise_real = 0.1 * ((double)rand() / RAND_MAX - 0.5);
        double noise_imag = 0.1 * ((double)rand() / RAND_MAX - 0.5);
        data[i] += noise_real + I * noise_imag;
    }
    
    return signal;
}

// Matched filter processing using VSLA convolution
static vsla_unified_tensor_t* matched_filter_processing(vsla_unified_context_t* ctx,
                                                         vsla_unified_tensor_t* signal,
                                                         vsla_unified_tensor_t* reference_pulse) {
    printf("  Performing matched filter processing...\n");
    
    // Get signal and pulse dimensions
    uint8_t signal_rank, pulse_rank;
    const uint64_t *signal_shape, *pulse_shape;
    
    vsla_tensor_get_info(signal, &signal_rank, &signal_shape, NULL, NULL);
    vsla_tensor_get_info(reference_pulse, &pulse_rank, &pulse_shape, NULL, NULL);
    
    // Output size for convolution
    uint64_t output_shape[] = {signal_shape[0] + pulse_shape[0] - 1};
    vsla_unified_tensor_t* matched_output = vsla_tensor_create(ctx, 1, output_shape,
                                                                VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (!matched_output) return NULL;
    
    // Conjugate the reference pulse for matched filtering
    vsla_unified_tensor_t* conjugate_pulse = vsla_tensor_create(ctx, 1, pulse_shape,
                                                                 VSLA_MODEL_A, VSLA_DTYPE_F32);
    if (!conjugate_pulse) {
        vsla_tensor_free(matched_output);
        return NULL;
    }
    
    // Perform conjugation
    const float complex* pulse_data = (const float complex*)vsla_tensor_data(reference_pulse, NULL);
    float complex* conj_data = (float complex*)vsla_tensor_data_mut(conjugate_pulse, NULL);
    
    for (size_t i = 0; i < pulse_shape[0]; i++) {
        conj_data[i] = conjf(pulse_data[i]);
    }
    
    // VSLA automatically selects optimal convolution algorithm (FFT for large signals)
    clock_t start = clock();
    vsla_error_t err = vsla_conv(ctx, matched_output, signal, conjugate_pulse);
    clock_t end = clock();
    
    if (err != VSLA_SUCCESS) {
        printf("    Error in convolution: %d\n", err);
        vsla_tensor_free(matched_output);
        vsla_tensor_free(conjugate_pulse);
        return NULL;
    }
    
    double processing_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("    Convolution completed in %.2f ms\n", processing_time_ms);
    
    vsla_tensor_free(conjugate_pulse);
    return matched_output;
}

// Peak detection for target identification
static int detect_targets(vsla_unified_context_t* ctx,
                          vsla_unified_tensor_t* matched_filter_output,
                          double threshold,
                          radar_target_t* detected_targets) {
    printf("  Detecting targets with threshold %.2f...\n", threshold);
    
    const float complex* data = (const float complex*)vsla_tensor_data(matched_filter_output, NULL);
    
    uint8_t rank;
    const uint64_t* shape;
    vsla_tensor_get_info(matched_filter_output, &rank, &shape, NULL, NULL);
    
    int num_detected = 0;
    
    // Simple peak detection
    for (size_t i = 1; i < shape[0] - 1; i++) {
        double magnitude = cabsf(data[i]);
        double left_mag = cabsf(data[i-1]);
        double right_mag = cabsf(data[i+1]);
        
        // Local maximum above threshold
        if (magnitude > threshold && magnitude > left_mag && magnitude > right_mag) {
            if (num_detected < MAX_TARGETS) {
                detected_targets[num_detected].range_km = 
                    (i * LIGHT_SPEED) / (2.0 * SAMPLE_RATE) / 1000.0;
                detected_targets[num_detected].rcs_dbsm = 20.0 * log10(magnitude);
                detected_targets[num_detected].velocity_mps = 0.0; // Would need multiple pulses
                
                printf("    Target detected: Range=%.2fkm, Magnitude=%.2fdB\n",
                       detected_targets[num_detected].range_km,
                       detected_targets[num_detected].rcs_dbsm);
                
                num_detected++;
            }
        }
    }
    
    return num_detected;
}

// Performance benchmark comparing different signal sizes
static void benchmark_signal_processing(vsla_unified_context_t* ctx) {
    printf("\n=== Performance Benchmarking ===\n");
    
    int test_sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("Signal Size | Processing Time | Throughput (MOPS)\n");
    printf("------------|-----------------|------------------\n");
    
    for (int i = 0; i < num_tests; i++) {
        int signal_size = test_sizes[i];
        int pulse_size = 128;
        
        // Create test signals
        uint64_t signal_shape[] = {signal_size};
        uint64_t pulse_shape[] = {pulse_size};
        
        vsla_unified_tensor_t* signal = vsla_tensor_ones(ctx, 1, signal_shape, 
                                                          VSLA_MODEL_A, VSLA_DTYPE_F32);
        vsla_unified_tensor_t* pulse = vsla_tensor_ones(ctx, 1, pulse_shape,
                                                         VSLA_MODEL_A, VSLA_DTYPE_F32);
        
        uint64_t output_shape[] = {signal_size + pulse_size - 1};
        vsla_unified_tensor_t* output = vsla_tensor_create(ctx, 1, output_shape,
                                                            VSLA_MODEL_A, VSLA_DTYPE_F32);
        
        // Benchmark convolution
        clock_t start = clock();
        vsla_conv(ctx, output, signal, pulse);
        vsla_synchronize(ctx); // Ensure GPU operations complete
        clock_t end = clock();
        
        double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        double ops = (double)signal_size * pulse_size;
        double throughput_mops = ops / (time_ms * 1000.0);
        
        printf("%11d | %13.2f ms | %16.1f\n", signal_size, time_ms, throughput_mops);
        
        vsla_tensor_free(signal);
        vsla_tensor_free(pulse);
        vsla_tensor_free(output);
    }
}

int main(void) {
    printf("=== VSLA Radar Signal Processing Example ===\n");
    printf("Demonstrating variable-length signal processing with automatic hardware optimization\n\n");
    
    // Initialize VSLA with automatic configuration
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
    
    printf("Runtime: Backend=%d, Device=%s, Memory=%.1fGB\n\n", 
           backend, device_name, memory_gb);
    
    // Radar system parameters
    radar_params_t radar_params = {
        .carrier_freq_hz = 10e9,    // 10 GHz X-band
        .bandwidth_hz = 100e6,      // 100 MHz bandwidth
        .pulse_duration_s = 10e-6,  // 10 microseconds
        .prf_hz = 1000,             // 1 kHz PRF
        .num_pulses = 64            // 64 pulse CPI
    };
    
    // Define scenario with variable-length targets
    radar_target_t targets[5] = {
        {.range_km = 10.0, .velocity_mps = 50.0, .rcs_dbsm = 20.0, .azimuth_deg = 0.0, .pulse_length = 128},
        {.range_km = 25.0, .velocity_mps = -30.0, .rcs_dbsm = 15.0, .azimuth_deg = 45.0, .pulse_length = 256},
        {.range_km = 45.0, .velocity_mps = 100.0, .rcs_dbsm = 25.0, .azimuth_deg = 90.0, .pulse_length = 64},
        {.range_km = 75.0, .velocity_mps = 0.0, .rcs_dbsm = 10.0, .azimuth_deg = 180.0, .pulse_length = 512},
        {.range_km = 120.0, .velocity_mps = -75.0, .rcs_dbsm = 18.0, .azimuth_deg = 270.0, .pulse_length = 192}
    };
    int num_targets = 5;
    
    printf("=== Scenario Setup ===\n");
    printf("Radar: %.1f GHz, BW=%.0f MHz, PRF=%.0f Hz\n",
           radar_params.carrier_freq_hz/1e9, 
           radar_params.bandwidth_hz/1e6,
           radar_params.prf_hz);
    printf("Targets: %d with variable pulse lengths\n\n", num_targets);
    
    // Generate reference pulse (standard length)
    printf("=== Signal Generation ===\n");
    vsla_unified_tensor_t* reference_pulse = generate_radar_pulse(ctx, BASE_PULSE_LENGTH,
                                                                   radar_params.bandwidth_hz,
                                                                   SAMPLE_RATE);
    if (!reference_pulse) {
        printf("Failed to generate reference pulse\n");
        vsla_cleanup(ctx);
        return 1;
    }
    
    // Generate radar return signal
    vsla_unified_tensor_t* radar_return = generate_radar_return(ctx, targets, num_targets,
                                                                 &radar_params, MAX_SIGNAL_LENGTH);
    if (!radar_return) {
        printf("Failed to generate radar return\n");
        vsla_tensor_free(reference_pulse);
        vsla_cleanup(ctx);
        return 1;
    }
    
    printf("\n=== Signal Processing ===\n");
    
    // Perform matched filter processing
    // VSLA automatically selects FFT convolution for large signals
    vsla_unified_tensor_t* matched_output = matched_filter_processing(ctx, radar_return, reference_pulse);
    if (!matched_output) {
        printf("Failed in matched filter processing\n");
        vsla_tensor_free(reference_pulse);
        vsla_tensor_free(radar_return);
        vsla_cleanup(ctx);
        return 1;
    }
    
    // Target detection
    printf("\n=== Target Detection ===\n");
    radar_target_t detected_targets[MAX_TARGETS];
    int num_detected = detect_targets(ctx, matched_output, 0.1, detected_targets);
    
    printf("\nDetected %d targets:\n", num_detected);
    for (int i = 0; i < num_detected; i++) {
        printf("  Target %d: Range=%.2f km, Magnitude=%.1f dB\n",
               i+1, detected_targets[i].range_km, detected_targets[i].rcs_dbsm);
    }
    
    // Performance benchmarking
    benchmark_signal_processing(ctx);
    
    // Get performance statistics
    printf("\n=== Performance Statistics ===\n");
    vsla_stats_t stats;
    vsla_get_stats(ctx, &stats);
    
    printf("Total operations: %lu\n", stats.total_operations);
    printf("GPU operations: %lu (%.1f%%)\n", stats.gpu_operations, 
           100.0 * stats.gpu_operations / stats.total_operations);
    printf("CPU operations: %lu (%.1f%%)\n", stats.cpu_operations,
           100.0 * stats.cpu_operations / stats.total_operations);
    printf("Total processing time: %.2f ms\n", stats.total_time_ms);
    printf("GPU time: %.2f ms\n", stats.gpu_time_ms);
    printf("CPU time: %.2f ms\n", stats.cpu_time_ms);
    printf("Memory transfer time: %.2f ms\n", stats.transfer_time_ms);
    
    // Cleanup
    vsla_tensor_free(reference_pulse);
    vsla_tensor_free(radar_return);
    vsla_tensor_free(matched_output);
    vsla_cleanup(ctx);
    
    printf("\n✓ Radar signal processing completed successfully!\n");
    printf("VSLA automatically handled variable-length signals and optimized\n");
    printf("hardware usage for maximum performance.\n");
    
    return 0;
}