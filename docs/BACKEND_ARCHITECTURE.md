# VSLA Multi-Backend Architecture

## Overview

VSLA (Variable-Shape Linear Algebra) implements a flexible, extensible backend architecture that supports multiple GPU vendors and compute platforms. This document describes the design principles, implementation strategy, and migration path for adding new compute backends.

## Design Principles

### 1. **Graceful Degradation**
- Each backend is optional and independent
- Missing backends don't break the build
- Clear warnings when requested backends are unavailable
- Automatic fallback to available backends

### 2. **Unified API**
- Single consistent API across all backends
- Runtime backend selection and switching
- Transparent memory management across backends
- Backend-agnostic tensor operations

### 3. **Performance Optimization**
- Each backend uses vendor-specific optimal APIs
- Native memory formats (e.g., CUDA memory, Metal buffers)
- Vendor-specific optimizations (cuBLAS, rocBLAS, Metal Performance Shaders)
- Auto-detection of optimal compute architectures

### 4. **Build Flexibility**
- CMake options to enable/disable backends
- Independent backend detection and compilation
- Support for cross-compilation scenarios
- CI/CD friendly (builds succeed with partial backends)

## Supported Backends

| Backend | Status | Platform | Language | Primary Use Case |
|---------|--------|----------|----------|------------------|
| **CPU** | ✅ Current | All | C | Development, fallback, CPU-only systems |
| **CUDA** | ✅ Current | Linux/Windows | CUDA C | NVIDIA GPUs |
| **ROCm/HIP** | 🚧 Planned | Linux | HIP | AMD GPUs |
| **SYCL** | 🚧 Planned | All | C++/SYCL | Intel GPUs, cross-vendor |
| **Metal** | 🚧 Planned | macOS/iOS | Obj-C++/Metal | Apple Silicon |
| **OpenCL** | 🚧 Planned | All | C++/OpenCL | Cross-platform fallback |

## Architecture Overview

### Backend Interface Layer
```c
// Core backend interface (vsla/internal/vsla_backend.h)
typedef struct {
    const char* name;
    vsla_backend_type_t type;
    
    // Core operations
    vsla_error_t (*tensor_create)(vsla_tensor_t* tensor, ...);
    vsla_error_t (*tensor_copy)(vsla_tensor_t* dst, const vsla_tensor_t* src);
    
    // Mathematical operations  
    vsla_error_t (*add)(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*mul)(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    
    // Memory management
    vsla_error_t (*tensor_free)(vsla_tensor_t* tensor);
    
    // Backend-specific capabilities
    bool supports_unified_memory;
    bool supports_async_execution;
    size_t max_tensor_rank;
} vsla_backend_interface_t;
```

### Runtime Backend Selection
```c
// Context creation with backend preference
vsla_context_t* ctx = vsla_init_with_backend(VSLA_BACKEND_AUTO);

// Query available backends
vsla_backend_type_t* backends;
size_t count;
vsla_enumerate_backends(&backends, &count);

// Force specific backend
vsla_context_set_backend(ctx, VSLA_BACKEND_CUDA);
```

## CMake Backend Detection Strategy

### Current Implementation Pattern
```cmake
if(VSLA_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES native)  # Auto-detect GPU
        
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            # Enable CUDA backend
        endif()
    else()
        message(WARNING "CUDA requested but compiler not found")
    endif()
endif()
```

### Key CMake Features Used

1. **`check_language()`** - Safely probe for language support
2. **Conditional `enable_language()`** - Only enable when available
3. **`CMAKE_CUDA_ARCHITECTURES native`** - Auto-detect GPU architecture
4. **Backend-specific options** - `VSLA_BUILD_<BACKEND>` for user control
5. **Graceful warnings** - Clear messages when backends unavailable

## Backend Implementation Guide

### 1. Adding a New Backend (AMD ROCm Example)

#### Step 1: CMake Detection
```cmake
# --- AMD ROCm Backend ---
if(VSLA_BUILD_ROCM)
    find_package(hip QUIET)
    find_package(rocblas QUIET)
    
    if(hip_FOUND AND rocblas_FOUND)
        enable_language(HIP)
        message(STATUS "AMD ROCm backend enabled")
        
        list(APPEND VSLA_CORE_SOURCES
            src/backends/vsla_backend_rocm.cpp
            src/backends/rocm/vsla_hip_kernels.hip
        )
        
        add_compile_definitions(VSLA_BUILD_ROCM=1)
        set(VSLA_HAS_ROCM TRUE)
    else()
        message(WARNING "ROCm requested but not found")
    endif()
endif()
```

#### Step 2: Backend Interface Implementation
```c
// src/backends/vsla_backend_rocm.cpp
#include "vsla/internal/vsla_backend.h"
#include <hip/hip_runtime.h>

static vsla_error_t rocm_tensor_create(vsla_tensor_t* tensor, ...) {
    // ROCm-specific tensor creation using hipMalloc
    hipError_t err = hipMalloc(&tensor->gpu_data, size);
    return (err == hipSuccess) ? VSLA_SUCCESS : VSLA_ERROR_MEMORY;
}

static vsla_error_t rocm_add(vsla_tensor_t* out, 
                            const vsla_tensor_t* a, 
                            const vsla_tensor_t* b) {
    // Launch HIP kernel for addition
    dim3 grid, block;
    calculate_launch_params(&grid, &block, out->total_elements);
    
    hipLaunchKernelGGL(rocm_add_kernel, grid, block, 0, 0,
                       out->gpu_data, a->gpu_data, b->gpu_data, 
                       out->total_elements);
    
    return VSLA_SUCCESS;
}

// Backend interface registration
static const vsla_backend_interface_t rocm_backend = {
    .name = "AMD ROCm",
    .type = VSLA_BACKEND_ROCM,
    .tensor_create = rocm_tensor_create,
    .add = rocm_add,
    // ... other operations
};
```

#### Step 3: HIP Kernel Implementation  
```cpp
// src/backends/rocm/vsla_hip_kernels.hip
#include <hip/hip_runtime.h>

__global__ void rocm_add_kernel(double* out, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

### 2. Directory Structure
```
src/backends/
├── cpu/                           # CPU reference implementation
│   ├── vsla_cpu_arithmetic.c
│   ├── vsla_cpu_stacking.c
│   └── vsla_cpu_utils.c
├── cuda/                          # NVIDIA CUDA (current)
│   ├── vsla_gpu.cu
│   └── vsla_tensor_utils.c
├── rocm/                          # AMD ROCm/HIP (planned)
│   ├── vsla_hip_kernels.hip
│   └── vsla_rocm_utils.cpp
├── sycl/                          # Intel oneAPI SYCL (planned)
│   ├── vsla_sycl_kernels.cpp
│   └── vsla_sycl_utils.cpp
├── metal/                         # Apple Metal (planned)  
│   ├── vsla_metal_kernels.metal
│   └── vsla_metal_utils.mm
├── opencl/                        # OpenCL fallback (planned)
│   ├── vsla_opencl_kernels.cl
│   └── vsla_opencl_utils.cpp
├── vsla_backend_cpu.c             # Backend implementations
├── vsla_backend_cuda.c
├── vsla_backend_rocm.cpp          # Future backends
├── vsla_backend_sycl.cpp
├── vsla_backend_metal.mm
└── vsla_backend_opencl.cpp
```

## Migration Strategy

### Phase 1: Foundation (Current)
- ✅ CPU backend solid foundation
- ✅ CUDA backend with proper CMake detection
- ✅ Unified API design established
- ✅ Reference counting system (in progress)

### Phase 2: AMD Support
- 🚧 Add ROCm/HIP backend
- 🚧 Cross-validate against CUDA implementation
- 🚧 Performance benchmarking and optimization

### Phase 3: Intel Support
- 🚧 Add SYCL backend for Intel GPUs
- 🚧 Cross-vendor kernel compatibility testing
- 🚧 Integration with Intel toolchain

### Phase 4: Apple Support  
- 🚧 Add Metal backend for Apple Silicon
- 🚧 Optimize for unified memory architecture
- 🚧 iOS/macOS compatibility

### Phase 5: Universal Fallback
- 🚧 Add OpenCL backend as universal fallback
- 🚧 Runtime backend auto-selection logic
- 🚧 Comprehensive cross-platform testing

## Performance Considerations

### Memory Management
- **Unified Memory**: Where supported (CUDA, Metal)
- **Explicit Transfers**: For discrete GPUs (most desktop systems)
- **Memory Pools**: Reduce allocation overhead
- **Reference Counting**: Prevent memory leaks across backends

### Kernel Optimization
- **Auto-tuning**: Empirical optimization of block/grid sizes  
- **Vendor Libraries**: Leverage cuBLAS, rocBLAS, oneMKL, MPS
- **Mixed Precision**: FP16, BF16, INT8 support where available
- **Asynchronous Execution**: Overlap compute and memory transfers

### Benchmarking Strategy
```c
// Backend performance comparison
typedef struct {
    vsla_backend_type_t backend;
    double execution_time_ms;
    size_t memory_usage_bytes;
    double throughput_gflops;
} vsla_benchmark_result_t;

vsla_benchmark_result_t* results;
size_t count;
vsla_benchmark_operation(VSLA_OP_ADD, tensor_sizes, &results, &count);
```

## Testing Strategy

### Unit Tests Per Backend
- Functional correctness against CPU reference
- Numerical precision validation
- Memory leak detection (Valgrind, AddressSanitizer)
- Error handling and edge cases

### Integration Tests
- Multi-backend tensor operations
- Memory transfer between backends  
- Context switching performance
- Concurrent multi-GPU execution

### CI/CD Strategy
```yaml
# .github/workflows/backends.yml
matrix:
  include:
    - os: ubuntu-latest
      backend: CPU
      required: true
    - os: ubuntu-latest  
      backend: CUDA
      required: false  # Optional, may not be available
    - os: ubuntu-latest
      backend: ROCm
      required: false
    - os: macos-latest
      backend: Metal
      required: false
```

## API Stability Guarantees

### Stable APIs (v1.0+)
- Core tensor operations (add, mul, stack, etc.)
- Context management
- Error handling
- Memory management interface

### Backend-Specific Extensions
- Vendor-specific optimizations
- Advanced memory management
- Profiling integration
- Multi-GPU coordination

## Future Considerations

### Emerging Platforms
- **Qualcomm Adreno** (mobile GPUs)
- **ARM Mali** (embedded systems)
- **Google TPU** (machine learning)
- **Amazon Inferentia** (inference acceleration)

### Advanced Features
- **Multi-node scaling** (MPI, NCCL)
- **Dynamic compilation** (runtime kernel generation)
- **Auto-differentiation** (for ML workloads)
- **Sparse tensor support** (COO, CSR formats)

## Contributing Guidelines

### Adding New Backends
1. Create RFC document describing backend integration
2. Implement CMake detection following established patterns
3. Create backend interface implementation
4. Add comprehensive tests matching existing backends
5. Update documentation and examples
6. Performance benchmark against reference implementations

### Code Style
- Follow existing C/C++ conventions in codebase
- Use vendor-neutral naming where possible
- Document backend-specific limitations
- Provide clear error messages and warnings

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-22  
**Maintainer**: VSLA Core Team