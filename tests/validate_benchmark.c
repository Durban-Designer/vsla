#include "include/vsla/vsla.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    printf("=== VSLA Benchmark Validation ===\n\n");
    
    vsla_init();
    
    // Test various sizes to validate correctness
    size_t test_sizes[][2] = {
        {64, 8},    // 64*8 = 512 > 64 -> FFT
        {128, 16},  // 128*16 = 2048 > 64 -> FFT
        {256, 32},  // 256*32 = 8192 > 64 -> FFT
        {8, 4},     // 8*4 = 32 < 64 -> Direct (for auto-select test)
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        size_t signal_size = test_sizes[t][0];
        size_t kernel_size = test_sizes[t][1];
        size_t output_size = signal_size + kernel_size - 1;
        
        printf("Testing signal_size=%zu, kernel_size=%zu (product=%zu)\n", 
               signal_size, kernel_size, signal_size * kernel_size);
        
        // Create test data
        vsla_tensor_t* signal = vsla_new(1, &signal_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* kernel = vsla_new(1, &kernel_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* result_auto = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* result_direct = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        vsla_tensor_t* result_fft = vsla_new(1, &output_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!signal || !kernel || !result_auto || !result_direct || !result_fft) {
            printf("ERROR: Failed to allocate tensors\n");
            return 1;
        }
        
        // Fill with test pattern (same as benchmark)
        for (size_t i = 0; i < signal_size; i++) {
            uint64_t idx = i;
            vsla_set_f64(signal, &idx, sin(2.0 * M_PI * i / signal_size));
        }
        
        for (size_t i = 0; i < kernel_size; i++) {
            uint64_t idx = i;
            vsla_set_f64(kernel, &idx, exp(-0.1 * i));
        }
        
        // Test all three methods
        vsla_error_t err1 = vsla_conv(result_auto, signal, kernel);        // Auto-select
        vsla_error_t err2 = vsla_conv_direct(result_direct, signal, kernel); // Direct
        vsla_error_t err3 = vsla_conv_fft(result_fft, signal, kernel);       // FFT
        
        if (err1 != VSLA_SUCCESS || err2 != VSLA_SUCCESS || err3 != VSLA_SUCCESS) {
            printf("ERROR: Convolution failed (auto=%d, direct=%d, fft=%d)\n", err1, err2, err3);
            return 1;
        }
        
        // Compare results
        double max_diff_auto_direct = 0.0;
        double max_diff_auto_fft = 0.0;
        double max_diff_direct_fft = 0.0;
        
        for (size_t i = 0; i < output_size; i++) {
            uint64_t idx = i;
            double val_auto, val_direct, val_fft;
            
            vsla_get_f64(result_auto, &idx, &val_auto);
            vsla_get_f64(result_direct, &idx, &val_direct);
            vsla_get_f64(result_fft, &idx, &val_fft);
            
            double diff_auto_direct = fabs(val_auto - val_direct);
            double diff_auto_fft = fabs(val_auto - val_fft);
            double diff_direct_fft = fabs(val_direct - val_fft);
            
            if (diff_auto_direct > max_diff_auto_direct) max_diff_auto_direct = diff_auto_direct;
            if (diff_auto_fft > max_diff_auto_fft) max_diff_auto_fft = diff_auto_fft;
            if (diff_direct_fft > max_diff_direct_fft) max_diff_direct_fft = diff_direct_fft;
        }
        
        printf("  Max differences:\n");
        printf("    Auto vs Direct: %.2e\n", max_diff_auto_direct);
        printf("    Auto vs FFT:    %.2e\n", max_diff_auto_fft);
        printf("    Direct vs FFT:  %.2e\n", max_diff_direct_fft);
        
        // Check if differences are within acceptable tolerance
        double tolerance = 1e-10;
        if (max_diff_direct_fft > tolerance) {
            printf("  WARNING: Direct and FFT results differ by more than %.2e\n", tolerance);
            
            // Print first few values for debugging
            printf("  First 5 values:\n");
            for (size_t i = 0; i < 5 && i < output_size; i++) {
                uint64_t idx = i;
                double val_direct, val_fft;
                vsla_get_f64(result_direct, &idx, &val_direct);
                vsla_get_f64(result_fft, &idx, &val_fft);
                printf("    [%zu] Direct: %.6f, FFT: %.6f, diff: %.2e\n", 
                       i, val_direct, val_fft, fabs(val_direct - val_fft));
            }
        } else {
            printf("  ✓ All algorithms agree within tolerance\n");
        }
        
        // Validate which algorithm vsla_conv chose
        if (signal_size * kernel_size > 64) {
            if (max_diff_auto_fft < tolerance) {
                printf("  ✓ Auto-select correctly chose FFT (product > 64)\n");
            } else {
                printf("  ERROR: Auto-select should have chosen FFT but results differ\n");
            }
        } else {
            if (max_diff_auto_direct < tolerance) {
                printf("  ✓ Auto-select correctly chose Direct (product <= 64)\n");
            } else {
                printf("  ERROR: Auto-select should have chosen Direct but results differ\n");
            }
        }
        
        vsla_free(signal);
        vsla_free(kernel);
        vsla_free(result_auto);
        vsla_free(result_direct);
        vsla_free(result_fft);
        
        printf("\n");
    }
    
    vsla_cleanup();
    
    printf("=== Validation Complete ===\n");
    return 0;
}