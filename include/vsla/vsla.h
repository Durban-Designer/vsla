/**
 * @file vsla.h
 * @brief Main header file for the Variable-Shape Linear Algebra (VSLA) library
 * 
 * This library implements Variable-Shape Linear Algebra, a mathematical framework
 * where vector and matrix dimensions are treated as intrinsic data rather than
 * fixed constraints. The library provides two models:
 * - Model A: Convolution-based semiring (commutative)
 * - Model B: Kronecker product-based semiring (non-commutative)
 * 
 * @copyright MIT License
 */

#ifndef VSLA_H
#define VSLA_H

#include "vsla_core.h"
#include "vsla_tensor.h"
#include "vsla_ops.h"
#include "vsla_io.h"
#include "vsla_conv.h"
#include "vsla_kron.h"
#include "vsla_autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Library version information
 */
#define VSLA_VERSION_MAJOR 1
#define VSLA_VERSION_MINOR 0
#define VSLA_VERSION_PATCH 0
#define VSLA_VERSION_STRING "1.0.0"

/**
 * @brief Initialize the VSLA library
 * 
 * This function initializes the library, including setting up FFTW plans
 * if FFTW support is enabled. This is optional but recommended for optimal
 * performance with Model A operations.
 * 
 * @return VSLA_SUCCESS on success, error code otherwise
 */
vsla_error_t vsla_init(void);

/**
 * @brief Clean up the VSLA library
 * 
 * This function cleans up any global resources used by the library,
 * including FFTW plans if FFTW support is enabled.
 * 
 * @return VSLA_SUCCESS on success, error code otherwise
 */
vsla_error_t vsla_cleanup(void);

/**
 * @brief Get the version string of the library
 * 
 * @return Version string in the format "major.minor.patch"
 */
const char* vsla_version(void);

/**
 * @brief Check if FFTW support is compiled in
 * 
 * @return 1 if FFTW is available, 0 otherwise
 */
int vsla_has_fftw(void);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_H */