/**
 * @file vsla_utils.c
 * @brief Utility functions and library initialization
 * 
 * @copyright MIT License
 */

#include "vsla/vsla.h"
#include <string.h>

static int g_initialized = 0;

vsla_error_t vsla_init(void) {
    if (g_initialized) {
        return VSLA_SUCCESS;
    }
    
    /* TODO: Initialize FFTW if available */
#ifdef USE_FFTW
    /* FFTW initialization would go here */
#endif
    
    g_initialized = 1;
    return VSLA_SUCCESS;
}

vsla_error_t vsla_cleanup(void) {
    if (!g_initialized) {
        return VSLA_SUCCESS;
    }
    
    /* TODO: Cleanup FFTW if available */
#ifdef USE_FFTW
    /* FFTW cleanup would go here */
#endif
    
    g_initialized = 0;
    return VSLA_SUCCESS;
}

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