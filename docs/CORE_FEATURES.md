# VSLA Core Features Specification

## Hardware-Agnostic Interface Design

This document defines the comprehensive set of core features that VSLA exposes through a unified, hardware-agnostic interface, enabling maximum performance while hiding hardware complexity from users.

## 1. Core Tensor Operations

### 1.1 Tensor Creation and Management
```c
// Automatic memory placement (CPU/GPU) based on size and availability
vsla_unified_tensor_t* vsla_tensor_create(vsla_unified_context_t* ctx, ...);
vsla_unified_tensor_t* vsla_tensor_from_array(vsla_unified_context_t* ctx, ...);
vsla_unified_tensor_t* vsla_tensor_zeros(vsla_unified_context_t* ctx, ...);
vsla_unified_tensor_t* vsla_tensor_ones(vsla_unified_context_t* ctx, ...);
vsla_unified_tensor_t* vsla_tensor_eye(vsla_unified_context_t* ctx, size_t n);
```

### 1.2 Element-wise Operations
```c
// All operations automatically use best available hardware
vsla_error_t vsla_add(vsla_unified_context_t* ctx, ...);           // Addition with variable-shape zero-padding
vsla_error_t vsla_sub(vsla_unified_context_t* ctx, ...);           // Subtraction with broadcasting
vsla_error_t vsla_mul(vsla_unified_context_t* ctx, ...);           // Element-wise multiplication
vsla_error_t vsla_div(vsla_unified_context_t* ctx, ...);           // Element-wise division
vsla_error_t vsla_scale(vsla_unified_context_t* ctx, ...);         // Scalar multiplication
vsla_error_t vsla_negate(vsla_unified_context_t* ctx, ...);        // Negation
vsla_error_t vsla_abs(vsla_unified_context_t* ctx, ...);           // Absolute value
```

### 1.3 Reduction Operations
```c
vsla_error_t vsla_sum(vsla_unified_context_t* ctx, ...);           // Sum along axes
vsla_error_t vsla_mean(vsla_unified_context_t* ctx, ...);          // Mean along axes
vsla_error_t vsla_max(vsla_unified_context_t* ctx, ...);           // Maximum along axes
vsla_error_t vsla_min(vsla_unified_context_t* ctx, ...);           // Minimum along axes
vsla_error_t vsla_norm(vsla_unified_context_t* ctx, ...);          // L1/L2/Frobenius norms
```

## 2. Convolution Operations (Model A)

### 2.1 1D Convolution
```c
// Automatic algorithm selection: direct, FFT, or vendor FFT (cuFFT/rocFFT/MKL)
vsla_error_t vsla_conv1d(vsla_unified_context_t* ctx,
                         vsla_unified_tensor_t* output,
                         const vsla_unified_tensor_t* signal,
                         const vsla_unified_tensor_t* kernel,
                         vsla_conv_mode_t mode);

// Specialized convolutions
vsla_error_t vsla_conv1d_fft(vsla_unified_context_t* ctx, ...);    // Force FFT algorithm
vsla_error_t vsla_conv1d_direct(vsla_unified_context_t* ctx, ...); // Force direct algorithm
vsla_error_t vsla_conv1d_separable(vsla_unified_context_t* ctx, ...); // Separable convolution
```

### 2.2 Multi-dimensional Convolution
```c
vsla_error_t vsla_conv2d(vsla_unified_context_t* ctx, ...);        // 2D convolution (images)
vsla_error_t vsla_conv3d(vsla_unified_context_t* ctx, ...);        // 3D convolution (volumes)
vsla_error_t vsla_convnd(vsla_unified_context_t* ctx, ...);        // N-dimensional convolution

// Neural network optimized convolutions
vsla_error_t vsla_conv2d_nn(vsla_unified_context_t* ctx,           // Stride, padding, dilation
                            vsla_unified_tensor_t* output,
                            const vsla_unified_tensor_t* input,
                            const vsla_unified_tensor_t* weights,
                            const vsla_conv_params_t* params);
```

### 2.3 Correlation and Cross-correlation
```c
vsla_error_t vsla_correlate(vsla_unified_context_t* ctx, ...);     // Cross-correlation
vsla_error_t vsla_autocorr(vsla_unified_context_t* ctx, ...);      // Auto-correlation
vsla_error_t vsla_xcorr_normalized(vsla_unified_context_t* ctx, ...); // Normalized cross-correlation
```

## 3. Kronecker Product Operations (Model B)

### 3.1 Basic Kronecker Operations
```c
vsla_error_t vsla_kron(vsla_unified_context_t* ctx,                // Standard Kronecker product
                       vsla_unified_tensor_t* output,
                       const vsla_unified_tensor_t* a,
                       const vsla_unified_tensor_t* b);

vsla_error_t vsla_kron_sum(vsla_unified_context_t* ctx, ...);      // Kronecker sum A ⊕ B
vsla_error_t vsla_kron_power(vsla_unified_context_t* ctx, ...);    // Kronecker power A^⊗n
```

### 3.2 Sparse Kronecker Operations
```c
vsla_error_t vsla_kron_sparse(vsla_unified_context_t* ctx, ...);   // Sparse Kronecker product
vsla_error_t vsla_kron_block(vsla_unified_context_t* ctx, ...);    // Block Kronecker product
```

### 3.3 Advanced Kronecker Applications
```c
vsla_error_t vsla_vec_kron(vsla_unified_context_t* ctx, ...);      // Vectorized Kronecker
vsla_error_t vsla_kron_solve(vsla_unified_context_t* ctx, ...);    // Solve (A ⊗ B)x = c
vsla_error_t vsla_kron_eigen(vsla_unified_context_t* ctx, ...);    // Eigendecomposition
```

## 4. Linear Algebra Operations

### 4.1 Matrix Operations
```c
vsla_error_t vsla_matmul(vsla_unified_context_t* ctx, ...);        // Matrix multiplication (vendor BLAS)
vsla_error_t vsla_transpose(vsla_unified_context_t* ctx, ...);     // Matrix transpose
vsla_error_t vsla_inverse(vsla_unified_context_t* ctx, ...);       // Matrix inverse
vsla_error_t vsla_determinant(vsla_unified_context_t* ctx, ...);   // Determinant
```

### 4.2 Decompositions
```c
vsla_error_t vsla_lu_decomp(vsla_unified_context_t* ctx, ...);     // LU decomposition
vsla_error_t vsla_qr_decomp(vsla_unified_context_t* ctx, ...);     // QR decomposition
vsla_error_t vsla_svd(vsla_unified_context_t* ctx, ...);           // SVD decomposition
vsla_error_t vsla_cholesky(vsla_unified_context_t* ctx, ...);      // Cholesky decomposition
```

### 4.3 Eigenvalue Problems
```c
vsla_error_t vsla_eigenvalues(vsla_unified_context_t* ctx, ...);   // Eigenvalues
vsla_error_t vsla_eigenvectors(vsla_unified_context_t* ctx, ...);  // Eigenvectors
vsla_error_t vsla_generalized_eigen(vsla_unified_context_t* ctx, ...); // Generalized eigenvalue
```

## 5. Polynomial Algebra

### 5.1 Polynomial Operations
```c
vsla_error_t vsla_poly_add(vsla_unified_context_t* ctx, ...);      // Polynomial addition
vsla_error_t vsla_poly_mul(vsla_unified_context_t* ctx, ...);      // Polynomial multiplication
vsla_error_t vsla_poly_div(vsla_unified_context_t* ctx, ...);      // Polynomial division
vsla_error_t vsla_poly_eval(vsla_unified_context_t* ctx, ...);     // Polynomial evaluation
```

### 5.2 Polynomial Transforms
```c
vsla_error_t vsla_poly_roots(vsla_unified_context_t* ctx, ...);    // Find polynomial roots
vsla_error_t vsla_poly_fit(vsla_unified_context_t* ctx, ...);      // Polynomial fitting
vsla_error_t vsla_poly_derivative(vsla_unified_context_t* ctx, ...); // Polynomial derivative
vsla_error_t vsla_poly_integral(vsla_unified_context_t* ctx, ...); // Polynomial integral
```

## 6. Signal Processing Operations

### 6.1 Transforms
```c
vsla_error_t vsla_fft(vsla_unified_context_t* ctx, ...);           // FFT (vendor optimized)
vsla_error_t vsla_ifft(vsla_unified_context_t* ctx, ...);          // Inverse FFT
vsla_error_t vsla_dct(vsla_unified_context_t* ctx, ...);           // Discrete Cosine Transform
vsla_error_t vsla_dwt(vsla_unified_context_t* ctx, ...);           // Discrete Wavelet Transform
```

### 6.2 Filtering
```c
vsla_error_t vsla_filter_iir(vsla_unified_context_t* ctx, ...);    // IIR filtering
vsla_error_t vsla_filter_fir(vsla_unified_context_t* ctx, ...);    // FIR filtering
vsla_error_t vsla_filter_median(vsla_unified_context_t* ctx, ...); // Median filtering
vsla_error_t vsla_filter_gaussian(vsla_unified_context_t* ctx, ...); // Gaussian filtering
```

### 6.3 Spectral Analysis
```c
vsla_error_t vsla_spectrogram(vsla_unified_context_t* ctx, ...);   // Short-time FFT
vsla_error_t vsla_periodogram(vsla_unified_context_t* ctx, ...);   // Power spectral density
vsla_error_t vsla_coherence(vsla_unified_context_t* ctx, ...);     // Coherence analysis
```

## 7. Variable-Shape Capabilities

### 7.1 Automatic Shape Handling
```c
// All operations automatically handle variable shapes with zero-padding
vsla_error_t vsla_broadcast_op(vsla_unified_context_t* ctx, ...);  // Explicit broadcasting
vsla_error_t vsla_resize(vsla_unified_context_t* ctx, ...);        // Tensor resizing
vsla_error_t vsla_pad(vsla_unified_context_t* ctx, ...);           // Explicit padding
vsla_error_t vsla_trim(vsla_unified_context_t* ctx, ...);          // Remove padding
```

### 7.2 Shape Utilities
```c
vsla_error_t vsla_shape_compatible(const vsla_unified_tensor_t* a,
                                   const vsla_unified_tensor_t* b);
vsla_error_t vsla_optimal_shape(vsla_unified_context_t* ctx, ...); // Compute optimal padded shape
```

## 8. Hardware Abstraction Features

### 8.1 Automatic Hardware Selection
```c
// Context automatically detects and uses best hardware
vsla_unified_context_t* vsla_init(const vsla_config_t* config);
vsla_error_t vsla_set_device(vsla_unified_context_t* ctx, int device_id);
vsla_error_t vsla_get_device_info(vsla_unified_context_t* ctx, ...);
```

### 8.2 Memory Management
```c
// Transparent CPU/GPU memory management
vsla_error_t vsla_prefetch_gpu(vsla_unified_context_t* ctx, vsla_unified_tensor_t* tensor);
vsla_error_t vsla_prefetch_cpu(vsla_unified_context_t* ctx, vsla_unified_tensor_t* tensor);
vsla_error_t vsla_synchronize(vsla_unified_context_t* ctx);
```

### 8.3 Performance Monitoring
```c
vsla_error_t vsla_get_stats(const vsla_unified_context_t* ctx, vsla_stats_t* stats);
vsla_error_t vsla_profile_operation(vsla_unified_context_t* ctx, const char* op_name);
vsla_error_t vsla_benchmark_backends(vsla_unified_context_t* ctx, ...);
```

## 9. Batch Operations

### 9.1 Vectorized Operations
```c
vsla_error_t vsla_batch_conv(vsla_unified_context_t* ctx, ...);    // Batch convolution
vsla_error_t vsla_batch_matmul(vsla_unified_context_t* ctx, ...);  // Batch matrix multiply
vsla_error_t vsla_batch_fft(vsla_unified_context_t* ctx, ...);     // Batch FFT
```

### 9.2 Graph Execution
```c
vsla_error_t vsla_execute_graph(vsla_unified_context_t* ctx,       // Execute operation graph
                                const vsla_operation_t* ops,
                                size_t count);
```

## 10. Use Case Examples

### 10.1 Real-world Applications
1. **Signal Processing**: Audio/radar processing with variable-length signals
2. **Neural Networks**: CNN layers with dynamic input sizes
3. **Scientific Computing**: PDE solvers with adaptive meshes
4. **Image Processing**: Multi-resolution image analysis
5. **Time Series**: Financial modeling with irregular sampling
6. **Quantum Computing**: Tensor network contractions
7. **Graph Processing**: Adjacency matrix operations

### 10.2 Performance Targets
- **CPU Operations**: Competitive with NumPy/SciPy
- **GPU Operations**: 10-100× speedup for large tensors
- **Vendor Libraries**: Near-optimal performance using cuFFT/rocFFT/MKL
- **Memory Efficiency**: <20% overhead for variable-shape handling
- **Latency**: <1ms overhead for hardware abstraction

## 11. Error Handling and Robustness

### 11.1 Comprehensive Error Reporting
```c
const char* vsla_error_string(vsla_error_t error);
vsla_error_t vsla_get_last_error(vsla_unified_context_t* ctx);
vsla_error_t vsla_set_error_callback(vsla_unified_context_t* ctx, vsla_error_callback_t cb);
```

### 11.2 Memory Safety
- Automatic bounds checking
- Memory leak detection
- GPU memory management
- Safe tensor aliasing

This comprehensive feature set provides a complete, hardware-agnostic interface that maximizes performance while maintaining the mathematical rigor and variable-shape advantages that make VSLA unique.