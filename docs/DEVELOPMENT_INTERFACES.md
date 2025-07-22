# VSLA Multi-Agent Development Interfaces

## 🎯 Architecture Status: READY FOR PARALLEL DEVELOPMENT

The legacy conflicts have been resolved. All agents can now work concurrently on their assigned backends.

### ✅ RESOLVED ISSUES
- **Removed conflicting `vsla_backend_registry.c`** - This was causing Gemini's compilation errors
- **Clean unified interface established** - All operations go through `vsla_unified.c`
- **Updated backend interface** - Context-aware function signatures implemented
- **Working build system** - Core library compiles successfully

## 🔧 Agent Assignments

### Claude: CPU Backend Implementation
**Responsibility:** Optimize CPU operations in `/src/backends/cpu/` files

**Critical Functions to Implement (Priority Order):**

#### Tier 1 - Foundation:
```c
// src/backends/cpu/vsla_cpu_memory.c
vsla_error_t cpu_allocate(vsla_tensor_t* tensor);
vsla_error_t cpu_deallocate(vsla_tensor_t* tensor);

// src/backends/cpu/vsla_cpu_arithmetic.c  
vsla_error_t cpu_add(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t cpu_sub(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t cpu_scale(vsla_tensor_t* out, const vsla_tensor_t* in, double scalar);
vsla_error_t cpu_hadamard(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t cpu_fill(vsla_tensor_t* tensor, double value);
```

#### Tier 2 - Mathematical Core:
```c
// src/backends/cpu/vsla_cpu_advanced.c
vsla_error_t cpu_conv(vsla_tensor_t* out, const vsla_tensor_t* signal, const vsla_tensor_t* kernel);
vsla_error_t cpu_kron(vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);

// src/backends/cpu/vsla_cpu_reduction.c  
vsla_error_t cpu_sum(const vsla_tensor_t* tensor, double* sum);
vsla_error_t cpu_norm(const vsla_tensor_t* tensor, double* norm);
vsla_error_t cpu_mean(const vsla_tensor_t* tensor, double* mean);
```

**Performance Requirements:**
- **SIMD optimization** - Use 64-byte aligned memory, vectorized loops
- **FFT acceleration** - Implement O(mn d_max log d_max) convolution using FFTW
- **Sparse-aware** - Only operate on materialized elements (no padding)
- **Modern C17** - Use restrict pointers, pragma omp for parallelization

### Gemini: CUDA Backend Implementation  
**Responsibility:** Implement GPU operations in `/src/backends/vsla_backend_cuda.c`

**Clean Interface to Implement:**
```c
// This is the ONLY interface Gemini needs to implement:
vsla_backend_interface_t* vsla_backend_cuda_create(void);
```

**Critical Functions (Same signatures as CPU, different implementation):**

#### Memory Management:
```c
vsla_error_t cuda_allocate(vsla_context_t* ctx, vsla_tensor_t* tensor);
vsla_error_t cuda_copy_to_device(vsla_context_t* ctx, vsla_tensor_t* tensor);
vsla_error_t cuda_copy_to_host(vsla_context_t* ctx, vsla_tensor_t* tensor);
vsla_error_t cuda_synchronize(vsla_context_t* ctx);
```

#### Arithmetic Operations:
```c
vsla_error_t cuda_add(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
vsla_error_t cuda_conv(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* signal, const vsla_tensor_t* kernel);
// ... same pattern for all operations
```

**Performance Requirements:**
- **cuFFT acceleration** - Use cuFFT for O(mn d_max log d_max) convolution  
- **Unified memory** - Support zero-copy operations where possible
- **Kernel fusion** - Combine operations to minimize memory bandwidth
- **Stream management** - Implement async operations with CUDA streams

**Testing:** Use `/tests/test_clean_architecture.c` to verify your implementation works.

### Codex: Code Review & Quality Assurance
**Responsibility:** Ensure consistency and performance across implementations

**Review Checklist:**
- **API Consistency** - All backends implement exact same interface
- **Error Handling** - Proper VSLA error codes returned
- **Memory Safety** - No leaks, proper alignment, bounds checking  
- **Performance** - SIMD usage (CPU), kernel efficiency (CUDA)
- **Mathematical Correctness** - Results match paper specifications

## 📋 Shared Interface Specification

### Backend Interface Structure
Every backend must implement `vsla_backend_interface_t` defined in `/include/vsla/vsla_backend.h`:

```c
struct vsla_backend_interface_s {
    vsla_backend_caps_t caps;  // Backend metadata
    
    // Memory management (5 functions)
    vsla_error_t (*allocate)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    vsla_error_t (*deallocate)(vsla_context_t* ctx, vsla_tensor_t* tensor);
    // ... etc
    
    // Arithmetic operations (5 functions)  
    vsla_error_t (*add)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    // ... etc
    
    // Advanced operations (2 functions)
    vsla_error_t (*conv)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    vsla_error_t (*kron)(vsla_context_t* ctx, vsla_tensor_t* out, const vsla_tensor_t* a, const vsla_tensor_t* b);
    
    // Reduction operations (5 functions)
    vsla_error_t (*sum)(vsla_context_t* ctx, const vsla_tensor_t* tensor, double* sum);
    // ... etc
};
```

### Mathematical Requirements (From Paper)

#### Model A: Convolution Semiring
- **Operation:** `(v ∗ w)_k = Σ_{i+j=k+1} v_i w_j`  
- **Complexity:** O(mn d_max log d_max) via FFT
- **Implementation:** Zero-pad to next power of 2, use FFT/IFFT

#### Model B: Kronecker Semiring  
- **Operation:** Kronecker product `v ⊗ w`
- **Complexity:** O(mn d_max²) 
- **Implementation:** Direct computation, sparse-aware

#### Memory Model
- **Alignment:** 64-byte boundaries for SIMD
- **Growth:** Power-of-2 capacity expansion
- **Sparsity:** Store only minimal representatives (no trailing zeros)

## 🧪 Testing & Validation

### Shared Test Harness
```bash
# Test CPU backend
cd build && ./test_clean_architecture

# Test CUDA backend (when Gemini completes implementation)  
VSLA_BACKEND=CUDA ./test_clean_architecture

# Run benchmarks (when available)
cd benchmarks && ./run_comparison_suite.sh
```

### Performance Targets (From Paper)
- **Speed:** 3-5× faster than zero-padding approaches
- **Memory:** 62-68% reduction vs traditional padding
- **Comparison:** Outperform TensorFlow Ragged, PyTorch Nested

## 🔄 Development Workflow

1. **Each agent works in their assigned directories**
2. **All changes go through clean interface in `vsla_backend.h`**  
3. **Test frequently with `test_clean_architecture.c`**
4. **Coordinate through this specification document**
5. **No cross-backend dependencies** - each backend is self-contained

## 🎯 Success Criteria

- **✅ CPU Backend:** All 22 functions implemented and tested
- **✅ CUDA Backend:** All 22 functions implemented with GPU acceleration  
- **✅ Benchmarks:** Performance targets achieved vs competition
- **✅ Integration:** Python bindings work with both backends
- **✅ Documentation:** Complete API coverage and examples

---

**🚀 Ready to proceed! The architecture is clean and each agent can work independently on their assigned backend without conflicts.**