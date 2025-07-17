/**
 * @file vsla_fft_vendor.h
 * @brief Vendor-agnostic FFT wrapper interface for VSLA
 * 
 * This module provides a unified interface to various vendor FFT libraries
 * (cuFFT, rocFFT, Intel MKL) while maintaining the VSLA variable-shape
 * tensor abstraction.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_FFT_VENDOR_H
#define VSLA_FFT_VENDOR_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include "vsla_gpu.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FFT backend type
 */
typedef enum {
    VSLA_FFT_BACKEND_BUILTIN = 0,  /**< Built-in VSLA FFT implementation */
    VSLA_FFT_BACKEND_CUFFT = 1,    /**< NVIDIA cuFFT */
    VSLA_FFT_BACKEND_ROCFFT = 2,   /**< AMD rocFFT */
    VSLA_FFT_BACKEND_MKL = 3,      /**< Intel MKL FFT */
    VSLA_FFT_BACKEND_AUTO = 4      /**< Auto-detect best available backend */
} vsla_fft_backend_t;

/**
 * @brief FFT direction
 */
typedef enum {
    VSLA_FFT_FORWARD = -1,  /**< Forward FFT */
    VSLA_FFT_INVERSE = 1    /**< Inverse FFT */
} vsla_fft_direction_t;

/**
 * @brief FFT plan structure (opaque)
 */
typedef struct vsla_fft_plan vsla_fft_plan_t;

/**
 * @brief FFT configuration
 */
typedef struct {
    vsla_fft_backend_t backend;     /**< FFT backend to use */
    size_t max_workspace_size;      /**< Maximum workspace memory (0 = unlimited) */
    bool allow_gpu;                 /**< Allow GPU acceleration if available */
    bool measure_performance;       /**< Measure and optimize for performance */
    int device_id;                  /**< GPU device ID (-1 = auto) */
} vsla_fft_config_t;

/**
 * @brief FFT backend capabilities
 */
typedef struct {
    bool supports_gpu;              /**< Backend supports GPU acceleration */
    bool supports_double;           /**< Backend supports double precision */
    bool supports_single;           /**< Backend supports single precision */
    bool supports_multidim;         /**< Backend supports multi-dimensional FFT */
    bool supports_inplace;          /**< Backend supports in-place transforms */
    size_t max_1d_size;             /**< Maximum 1D FFT size (0 = unlimited) */
    const char* name;               /**< Backend name string */
    const char* version;            /**< Backend version string */
} vsla_fft_capabilities_t;

// FFT Backend Management
/**
 * @brief Initialize FFT subsystem
 * 
 * @param config FFT configuration (NULL for defaults)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_init(const vsla_fft_config_t* config);

/**
 * @brief Cleanup FFT subsystem
 */
void vsla_fft_cleanup(void);

/**
 * @brief Get available FFT backends
 * 
 * @param backends Array to fill with available backends
 * @param max_count Maximum number of backends to return
 * @param count Actual number of backends found
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_get_backends(vsla_fft_backend_t* backends, 
                                    size_t max_count, 
                                    size_t* count);

/**
 * @brief Get backend capabilities
 * 
 * @param backend FFT backend
 * @param caps Capabilities structure to fill
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_get_capabilities(vsla_fft_backend_t backend,
                                        vsla_fft_capabilities_t* caps);

/**
 * @brief Set active FFT backend
 * 
 * @param backend FFT backend to use
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_set_backend(vsla_fft_backend_t backend);

/**
 * @brief Get current FFT backend
 * 
 * @return Current FFT backend
 */
vsla_fft_backend_t vsla_fft_get_backend(void);

// FFT Planning
/**
 * @brief Create 1D FFT plan
 * 
 * @param size FFT size
 * @param dtype Data type (VSLA_DTYPE_F32 or VSLA_DTYPE_F64)
 * @param direction FFT direction
 * @return FFT plan or NULL on error
 */
vsla_fft_plan_t* vsla_fft_plan_1d(size_t size, 
                                   vsla_dtype_t dtype,
                                   vsla_fft_direction_t direction);

/**
 * @brief Create multi-dimensional FFT plan
 * 
 * @param rank Number of dimensions
 * @param sizes Size in each dimension
 * @param dtype Data type
 * @param direction FFT direction
 * @return FFT plan or NULL on error
 */
vsla_fft_plan_t* vsla_fft_plan_nd(uint8_t rank,
                                   const size_t* sizes,
                                   vsla_dtype_t dtype,
                                   vsla_fft_direction_t direction);

/**
 * @brief Destroy FFT plan
 * 
 * @param plan FFT plan to destroy
 */
void vsla_fft_plan_destroy(vsla_fft_plan_t* plan);

// FFT Execution - CPU
/**
 * @brief Execute 1D complex FFT
 * 
 * @param plan FFT plan
 * @param in_real Input real part
 * @param in_imag Input imaginary part
 * @param out_real Output real part
 * @param out_imag Output imaginary part
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_execute_1d(const vsla_fft_plan_t* plan,
                                  const void* in_real,
                                  const void* in_imag,
                                  void* out_real,
                                  void* out_imag);

/**
 * @brief Execute 1D real-to-complex FFT
 * 
 * @param plan FFT plan
 * @param in_real Input real data
 * @param out_real Output real part
 * @param out_imag Output imaginary part
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_execute_1d_r2c(const vsla_fft_plan_t* plan,
                                      const void* in_real,
                                      void* out_real,
                                      void* out_imag);

// FFT Execution - GPU
#ifdef VSLA_ENABLE_CUDA
/**
 * @brief Execute 1D FFT on GPU
 * 
 * @param plan FFT plan
 * @param gpu_in_real GPU input real part
 * @param gpu_in_imag GPU input imaginary part
 * @param gpu_out_real GPU output real part
 * @param gpu_out_imag GPU output imaginary part
 * @param stream CUDA stream (NULL for default)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_execute_1d_gpu(const vsla_fft_plan_t* plan,
                                      const void* gpu_in_real,
                                      const void* gpu_in_imag,
                                      void* gpu_out_real,
                                      void* gpu_out_imag,
                                      void* stream);
#endif

// High-level Convolution Functions
/**
 * @brief Perform FFT convolution using vendor libraries
 * 
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @param backend FFT backend to use (VSLA_FFT_BACKEND_AUTO for auto-select)
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_conv_fft_vendor(vsla_tensor_t* out,
                                   const vsla_tensor_t* a,
                                   const vsla_tensor_t* b,
                                   vsla_fft_backend_t backend);

/**
 * @brief Perform GPU FFT convolution using vendor libraries
 * 
 * @param out Output GPU tensor
 * @param a First input GPU tensor
 * @param b Second input GPU tensor
 * @param ctx GPU context
 * @param backend FFT backend to use
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_gpu_conv_fft_vendor(vsla_gpu_tensor_t* out,
                                       const vsla_gpu_tensor_t* a,
                                       const vsla_gpu_tensor_t* b,
                                       vsla_gpu_context_t* ctx,
                                       vsla_fft_backend_t backend);

// Utility Functions
/**
 * @brief Get optimal FFT size for convolution
 * 
 * @param signal_size Signal size
 * @param kernel_size Kernel size
 * @param backend FFT backend
 * @return Optimal FFT size
 */
size_t vsla_fft_get_optimal_size(size_t signal_size,
                                  size_t kernel_size,
                                  vsla_fft_backend_t backend);

/**
 * @brief Benchmark FFT backends for given size
 * 
 * @param size FFT size
 * @param dtype Data type
 * @param use_gpu Use GPU if available
 * @param results Array to store benchmark times (in microseconds)
 * @param backends Array of backends that were benchmarked
 * @param count Number of backends benchmarked
 * @return VSLA_SUCCESS on success
 */
vsla_error_t vsla_fft_benchmark(size_t size,
                                 vsla_dtype_t dtype,
                                 bool use_gpu,
                                 double* results,
                                 vsla_fft_backend_t* backends,
                                 size_t* count);

#ifdef __cplusplus
}
#endif

#endif // VSLA_FFT_VENDOR_H