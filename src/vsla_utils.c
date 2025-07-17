/**
 * @file vsla_utils.c
 * @brief Utility functions and library initialization
 * 
 * @copyright MIT License
 */

#include "vsla/vsla.h"
#include <string.h>

const char* vsla_version(void) {
    return VSLA_VERSION_STRING;
}

int vsla_has_fftw(void) {
#ifdef USE_FFTW
    return 1;
#else
    return 0;
#endif
}

int vsla_has_gpu(void) {
#ifdef VSLA_ENABLE_CUDA
    return 1;
#else
    return 0;
#endif
}