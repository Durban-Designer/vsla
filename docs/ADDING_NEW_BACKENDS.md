# Adding New GPU Backends to VSLA

## Quick Reference Guide

This document provides a step-by-step guide for adding new GPU backends to the VSLA library, using the established patterns from the current CUDA implementation.

## Prerequisites

- Understanding of CMake build systems
- Familiarity with the target GPU backend's development environment
- Access to hardware or CI/CD for testing the new backend

## Step-by-Step Implementation

### Step 1: Update CMakeLists.txt

Add your backend following the established CUDA pattern:

```cmake
# --- YOUR_BACKEND Backend ---
option(VSLA_BUILD_YOUR_BACKEND "Build the YourBackend backend" OFF)  # Default OFF for new backends

if(VSLA_BUILD_YOUR_BACKEND)
    # Check for backend availability
    find_package(YourBackendToolkit QUIET)  # e.g., hip, IntelSYCL, etc.
    
    if(YourBackendToolkit_FOUND)
        # Enable language if needed (e.g., CUDA, HIP, CXX)
        enable_language(YOUR_BACKEND_LANGUAGE)
        
        # Set backend-specific compiler flags
        set(CMAKE_YOUR_BACKEND_ARCHITECTURES native)  # If supported
        
        message(STATUS "YourBackend found, enabling YourBackend backend.")
        list(APPEND VSLA_CORE_SOURCES 
            src/backends/vsla_backend_your_backend.c
            src/backends/your_backend/vsla_your_backend_kernels.ext
            src/backends/your_backend/vsla_your_backend_utils.c
        )
        add_compile_definitions(VSLA_BUILD_YOUR_BACKEND=1)
        set(VSLA_HAS_YOUR_BACKEND TRUE)
    else()
        message(WARNING "VSLA_BUILD_YOUR_BACKEND is ON but YourBackend was not found. YourBackend backend will not be built.")
    endif()
endif()

# Add to linking section:
if(VSLA_HAS_YOUR_BACKEND)
    target_link_libraries(vsla_static PRIVATE YourBackend::runtime YourBackend::math)
endif()
```

### Step 2: Create Directory Structure

```bash
mkdir -p src/backends/your_backend
touch src/backends/vsla_backend_your_backend.c
touch src/backends/your_backend/vsla_your_backend_kernels.ext  # .cu, .hip, .cpp, .metal, etc.
touch src/backends/your_backend/vsla_your_backend_utils.c
```

### Step 3: Implement Backend Interface

Create `src/backends/vsla_backend_your_backend.c`:

```c
#include "vsla/internal/vsla_backend.h"
#include "vsla/internal/vsla_tensor_internal.h"
#include <your_backend_runtime.h>

// Forward declarations for kernels
extern vsla_error_t your_backend_add_kernel_launch(vsla_tensor_t* out, 
                                                   const vsla_tensor_t* a, 
                                                   const vsla_tensor_t* b);

// Tensor creation
vsla_error_t your_backend_tensor_create(vsla_tensor_t* tensor, 
                                       uint8_t rank, 
                                       const uint64_t* shape, 
                                       vsla_dtype_t dtype) {
    // Calculate total size
    uint64_t total_elements = 1;
    for (uint8_t i = 0; i < rank; i++) {
        total_elements *= shape[i];
    }
    
    size_t dtype_size = vsla_dtype_size(dtype);
    size_t total_bytes = total_elements * dtype_size;
    
    // Allocate GPU memory using your backend's API
    your_backend_error_t err = your_backend_malloc(&tensor->gpu_data, total_bytes);
    if (err != YOUR_BACKEND_SUCCESS) {
        return VSLA_ERROR_MEMORY;
    }
    
    tensor->data_size = total_bytes;
    tensor->gpu_valid = true;
    tensor->cpu_valid = false;
    tensor->location = VSLA_BACKEND_YOUR_BACKEND;
    
    return VSLA_SUCCESS;
}

// Addition operation
vsla_error_t your_backend_add(vsla_tensor_t* out, 
                             const vsla_tensor_t* a, 
                             const vsla_tensor_t* b) {
    if (!out || !a || !b) {
        return VSLA_ERROR_NULL_POINTER;
    }
    
    // Validation (shapes, types, etc.)
    // ... validation code ...
    
    // Launch kernel
    return your_backend_add_kernel_launch(out, a, b);
}

// Memory cleanup
vsla_error_t your_backend_tensor_free(vsla_tensor_t* tensor) {
    if (tensor && tensor->gpu_data) {
        your_backend_free(tensor->gpu_data);
        tensor->gpu_data = NULL;
        tensor->gpu_valid = false;
    }
    return VSLA_SUCCESS;
}

// Backend registration (called from vsla_init)
#ifdef VSLA_BUILD_YOUR_BACKEND
void register_your_backend_backend() {
    vsla_backend_interface_t backend = {
        .name = "YourBackend",
        .type = VSLA_BACKEND_YOUR_BACKEND,
        .tensor_create = your_backend_tensor_create,
        .add = your_backend_add,
        .tensor_free = your_backend_tensor_free,
        // ... other operations
    };
    
    vsla_register_backend(&backend);
}
#endif
```

### Step 4: Implement Kernels

Create `src/backends/your_backend/vsla_your_backend_kernels.ext`:

#### For CUDA-like backends (ROCm/HIP):
```cpp
// vsla_hip_kernels.hip
#include <hip/hip_runtime.h>

__global__ void hip_add_kernel(double* out, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

vsla_error_t your_backend_add_kernel_launch(vsla_tensor_t* out, 
                                           const vsla_tensor_t* a, 
                                           const vsla_tensor_t* b) {
    size_t n = out->total_elements;
    
    // Calculate launch parameters
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Launch kernel
    hipLaunchKernelGGL(hip_add_kernel, grid, block, 0, 0,
                       (double*)out->gpu_data, 
                       (const double*)a->gpu_data, 
                       (const double*)b->gpu_data, 
                       n);
    
    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? VSLA_SUCCESS : VSLA_ERROR_COMPUTE;
}
```

#### For SYCL backends:
```cpp
// vsla_sycl_kernels.cpp
#include <sycl/sycl.hpp>
using namespace sycl;

vsla_error_t your_backend_add_kernel_launch(vsla_tensor_t* out, 
                                           const vsla_tensor_t* a, 
                                           const vsla_tensor_t* b) {
    queue q;  // SYCL queue
    
    size_t n = out->total_elements;
    
    q.parallel_for(range<1>(n), [=](id<1> idx) {
        ((double*)out->gpu_data)[idx] = ((double*)a->gpu_data)[idx] + ((double*)b->gpu_data)[idx];
    }).wait();
    
    return VSLA_SUCCESS;
}
```

#### For Metal backends:
```objc
// vsla_metal_kernels.metal
#include <metal_stdlib>
using namespace metal;

kernel void metal_add_kernel(device float* out [[buffer(0)]],
                           const device float* a [[buffer(1)]],
                           const device float* b [[buffer(2)]],
                           uint index [[thread_position_in_grid]],
                           uint total [[threads_per_grid]]) {
    if (index < total) {
        out[index] = a[index] + b[index];
    }
}
```

### Step 5: Update Core Headers

Add your backend to `include/vsla/vsla_core.h`:

```c
typedef enum {
    VSLA_BACKEND_CPU = 0,
    VSLA_BACKEND_CUDA,
    VSLA_BACKEND_ROCM,        // Add your backend here
    VSLA_BACKEND_SYCL,
    VSLA_BACKEND_METAL,
    VSLA_BACKEND_OPENCL,
    VSLA_BACKEND_YOUR_BACKEND,  // Your new backend
    VSLA_BACKEND_COUNT
} vsla_backend_type_t;
```

### Step 6: Add Backend Registration

Update `src/vsla_core.c` to register your backend:

```c
#ifdef VSLA_BUILD_YOUR_BACKEND
extern void register_your_backend_backend();
#endif

vsla_context_t* vsla_init(const vsla_config_t* config) {
    // ... existing code ...
    
    #ifdef VSLA_BUILD_YOUR_BACKEND
    register_your_backend_backend();
    #endif
    
    // ... rest of initialization ...
}
```

### Step 7: Add Tests

Create `tests/test_your_backend.c`:

```c
#include "vsla/vsla_unified.h"
#include <stdio.h>

void test_your_backend_basic_operations() {
    vsla_context_t* ctx = vsla_init(NULL);
    
    // Force your backend
    vsla_context_set_backend(ctx, VSLA_BACKEND_YOUR_BACKEND);
    
    // Test tensor creation
    uint64_t shape[] = {1000};
    vsla_tensor_t* a = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* c = vsla_tensor_create(ctx, 1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    
    // Fill with test data
    for (uint64_t i = 0; i < 1000; i++) {
        uint64_t idx[] = {i};
        vsla_set_f64(ctx, a, idx, (double)i);
        vsla_set_f64(ctx, b, idx, (double)(i * 2));
    }
    
    // Test addition
    vsla_error_t err = vsla_add(ctx, c, a, b);
    assert(err == VSLA_SUCCESS);
    
    // Validate results
    for (uint64_t i = 0; i < 1000; i++) {
        uint64_t idx[] = {i};
        double val;
        vsla_get_f64(ctx, c, idx, &val);
        double expected = (double)i + (double)(i * 2);
        assert(fabs(val - expected) < 1e-10);
    }
    
    printf("✅ YourBackend basic operations test passed\n");
    
    vsla_tensor_free(a);
    vsla_tensor_free(b);
    vsla_tensor_free(c);
    vsla_cleanup(ctx);
}

int main() {
    #ifdef VSLA_BUILD_YOUR_BACKEND
    test_your_backend_basic_operations();
    #else
    printf("⚠️  YourBackend not built, skipping tests\n");
    #endif
    return 0;
}
```

Update `tests/CMakeLists.txt`:
```cmake
if(VSLA_HAS_YOUR_BACKEND)
    add_executable(test_your_backend test_your_backend.c)
    target_link_libraries(test_your_backend vsla_static)
    add_test(NAME YourBackendTests COMMAND test_your_backend)
endif()
```

## Backend-Specific Implementation Notes

### AMD ROCm/HIP
```cmake
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
enable_language(HIP)
target_link_libraries(vsla_static PRIVATE hip::host roc::rocblas)
```

### Intel SYCL  
```cmake
find_package(IntelSYCL REQUIRED)
# No need to enable_language, uses C++
target_link_libraries(vsla_static PRIVATE IntelSYCL::SYCL_CXX)
```

### Apple Metal
```cmake
if(APPLE)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(METALKIT_LIBRARY MetalKit REQUIRED) 
    target_link_libraries(vsla_static PRIVATE ${METAL_LIBRARY} ${METALKIT_LIBRARY})
endif()
```

### OpenCL
```cmake
find_package(OpenCL REQUIRED)
target_link_libraries(vsla_static PRIVATE OpenCL::OpenCL)
```

## Testing Your Backend

### Build Test
```bash
cd vsla
rm -rf build && mkdir build && cd build
cmake -DVSLA_BUILD_YOUR_BACKEND=ON ..
make -j$(nproc)
```

### Runtime Test  
```bash
./tests/test_your_backend
```

### Integration Test
```bash
# Should show your backend in the list
./examples/backend_info
```

## Common Issues and Solutions

### Issue: Backend not found during CMake
**Solution**: Ensure the backend's development packages are installed and findable by CMake.

### Issue: Linking errors
**Solution**: Check that all required libraries are linked in the CMakeLists.txt.

### Issue: Runtime crashes
**Solution**: Verify memory management and check that tensors are properly allocated on the GPU.

### Issue: Incorrect results
**Solution**: Compare against CPU reference implementation, check kernel logic and data types.

## Performance Optimization

1. **Use vendor-optimized libraries** (cuBLAS, rocBLAS, oneMKL, Metal Performance Shaders)
2. **Optimize memory access patterns** (coalesced access, shared memory)
3. **Tune launch parameters** (block/grid sizes, occupancy)
4. **Implement asynchronous execution** where supported
5. **Add mixed-precision support** (FP16, BF16, INT8)

## Contributing Your Backend

1. Follow the established patterns from CUDA implementation
2. Add comprehensive tests matching existing test coverage
3. Update documentation with backend-specific notes
4. Ensure CI/CD compatibility (graceful fallback when unavailable)
5. Add performance benchmarks comparing to other backends

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-22