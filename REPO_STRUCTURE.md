# VSLA Repository Structure

This document describes the clean, unified architecture of the VSLA repository after legacy cleanup.

## Root Directory Structure

```
vsla/
├── bench/                    # (Outdated) Benchmarking framework - needs rewrite
├── docs/                     # Project documentation and papers
├── examples/                 # (Outdated) Usage examples - needs rewrite  
├── include/                  # Public C headers for unified API
├── python/                   # Python bindings (working)
├── src/                      # Core C implementation of unified API
├── tests/                    # Unit and integration tests
├── CMakeLists.txt           # Build configuration
├── README.md                # Main project documentation
├── STATUS.md                # Current development status
├── REPO_STRUCTURE.md        # This file
├── LICENSE                  # MIT license
└── setup.py                 # Python package setup
```

## Clean Architecture Overview

### Core Implementation (`src/` - 4 files)

**Unified Interface:**
- `src/vsla_unified.c` - Main unified interface, single entry point for all operations

**Core System:**
- `src/vsla_core.c` - Core utility functions (error strings, dtype utilities)
- `src/vsla_tensor.c` - Tensor property accessors for opaque API
- `src/vsla_utils.c` - Library version and feature detection

**Backend System:**
- `src/backends/vsla_backend_cpu.c` - CPU backend with context-aware wrapper functions
- `src/backends/cpu/*.c` - CPU implementation files (memory, arithmetic, linalg, reduction, advanced)

### Public Headers (`include/vsla/` - 9 files)

**Main API:**
- `vsla.h` - Single public header file for entire library
- `vsla_unified.h` - Unified interface declarations
- `vsla_core.h` - Core types, error codes, and enums
- `vsla_tensor.h` - Opaque tensor handle and accessor functions

**Backend Interface:**
- `vsla_backend.h` - Backend interface definition
- `vsla_backend_cpu.h` - CPU backend declarations

**Internal (Implementation Details):**
- `vsla_tensor_internal.h` - Internal tensor structure definition
- `vsla_context.h` - Context structure definitions
- `internal/` - Internal implementation headers

### Python Bindings (`python/` - Working)

```
python/
├── src/bindings.cpp         # pybind11 C++ bindings (updated for unified API)
├── vsla/__init__.py         # Python package initialization
├── setup.py                 # Python build configuration (working)
├── venv/                    # Virtual environment
└── _core.*.so              # Built C extension
```

## Architecture Principles

### Single Entry Point
- All operations route through `src/vsla_unified.c`
- Context-based API: `vsla_operation(ctx, ...)`
- Clean separation from implementation details

### Backend Abstraction
- Modular backend system via `vsla_backend.h` interface
- Context parameter passed to all backend functions
- Easy addition of new backends (CUDA, ROCm, etc.)

### Opaque Handle Design
- `vsla_tensor_t` is opaque for ABI stability
- Internal structure in `vsla_tensor_internal.h`
- Public accessors in `vsla_tensor.h`

### Modern C17 Standard
- Clean, standards-compliant codebase
- No legacy conflicts or outdated patterns
- Thread-safe design considerations

## Development Status

✅ **Core architecture complete and clean**
✅ **Python bindings working**
✅ **Build system functional**
⚠️ **CPU backend functions need implementation**
❌ **Examples and benchmarks need rewrite for new API**

## Development Workflow

1. **Building**: `mkdir build && cd build && cmake .. && make`
2. **Python**: `cd python && source venv/bin/activate && pip install -e .`
3. **Testing**: `ctest` from build directory
4. **Python Testing**: `python -c "import vsla; print('Works!')"`

## Legacy Cleanup Complete

**Removed legacy files (8 files):**
- Old autograd system (`vsla_autograd.c/.h`)
- Standalone I/O system (`vsla_io.c/.h`)
- Tensor adapter system (`vsla_tensor_adapter.c/.h`)
- Minimal GPU utilities (`vsla_tensor_utils.c/.h`)

The repository now has a clean, minimal architecture focused on the unified interface without legacy conflicts.