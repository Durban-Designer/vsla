/**
 * @file vsla.h
 * @brief Main header file for the Variable-Shape Linear Algebra (VSLA) library
 * 
 * This is the single public API for VSLA. All operations go through the
 * unified interface with explicit context management for clean, modern API design.
 * 
 * @copyright MIT License
 */

#ifndef VSLA_H
#define VSLA_H

/* Core types and error codes */
#include "vsla_core.h"

/* Unified interface - the single control point for all operations */
#include "vsla_unified.h"

/* Note: vsla_tensor.h is included by vsla_unified.h for the opaque tensor type */
/* Users should not use direct tensor operations - use the context-based API */

#endif /* VSLA_H */
