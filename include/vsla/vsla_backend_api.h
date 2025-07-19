/**
 * @file vsla_backend_api.h
 * @brief API for backend discovery, negotiation, and management.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_BACKEND_API_H
#define VSLA_BACKEND_API_H

#include "vsla_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a backend implementation.
 */
typedef struct vsla_backend_instance vsla_backend_instance_t;

/**
 * @brief Get the number of available backends.
 *
 * @return The number of registered backends that can be used.
 */
int vsla_get_num_backends(void);

/**
 * @brief Get information about a specific backend.
 *
 * @param backend_index The index of the backend (from 0 to vsla_get_num_backends() - 1).
 * @param name_out Pointer to a char* to store the backend name.
 * @param capabilities_out Pointer to a uint32_t to store backend capability flags.
 * @return VSLA_SUCCESS or an error code.
 */
vsla_error_t vsla_get_backend_info(int backend_index, const char** name_out, uint32_t* capabilities_out);

/**
 * @brief Select and initialize a backend for use.
 *
 * @param backend_index The index of the backend to initialize.
 * @param instance_out Pointer to a vsla_backend_instance_t* to store the initialized backend instance.
 * @return VSLA_SUCCESS or an error code.
 */
vsla_error_t vsla_init_backend(int backend_index, vsla_backend_instance_t** instance_out);

/**
 * @brief Release a backend instance.
 *
 * @param instance The backend instance to release.
 * @return VSLA_SUCCESS or an error code.
 */
vsla_error_t vsla_release_backend(vsla_backend_instance_t* instance);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_BACKEND_API_H */
