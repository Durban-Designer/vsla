/**
 * @file vsla_context.h
 * @brief VSLA context structure definition
 * 
 * @copyright MIT License
 */

#ifndef VSLA_CONTEXT_H
#define VSLA_CONTEXT_H

#include "vsla_core.h"
#include "vsla_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief VSLA runtime context structure (forward declaration)
 * 
 * This structure manages the runtime state including backend selection,
 * device management, and performance statistics.
 */
typedef struct vsla_context vsla_context_t;

#ifdef __cplusplus
}
#endif

#endif /* VSLA_CONTEXT_H */