# VSLA Comprehensive Benchmarking Plan

## Current Status

### ✅ Completed Benchmarks
1. **Vector Addition (1D)** - `bench_vector_add_large.c`
   - Tests large 1D vectors with microsecond precision
   - Shows VSLA is 13-46% faster for equal sizes
   - Replaced misleading small-scale benchmark

2. **FFT Convolution** - `bench_fft_convolution.c`
   - Custom FFT implementation
   - Shows throughput of 258k-326k ops/ms

3. **Kronecker Product** - `bench_kronecker.c`
   - Limited to small sizes (10x10)
   - Needs expansion to larger tensors

4. **Window Stacking** - `bench_window_stacking.c`
   - Compares against circular buffers
   - Shows competitive performance

5. **Pyramid Stacking** - `bench_pyramid_stacking.c`
   - Compares against hierarchical buffers
   - Good for multi-resolution analysis

### 🚧 Critical Missing Benchmarks

1. **Multi-Dimensional Shape Mismatch Benchmark**
   - Test cases:
     - 3D CNN: [32,128,128,3] + [32,128,1,3] (spatial broadcasting)
     - 4D Attention: [8,512,512,64] + [8,1,512,64] (batch broadcasting)
     - 5D Video: [10,30,1920,1080,3] + [1,30,1,1,3] (temporal broadcasting)
   - Compare vs:
     - Manual zero-padding to largest shape
     - NumPy-style broadcasting with copies
     - Custom sparse implementations

2. **Sensor Fusion Benchmark**
   - 8 heterogeneous sensors with different rates/shapes
   - VSLA pyramid vs manual padding vs ragged tensors
   - Measure memory, latency, throughput
   - Real-world autonomous vehicle scenario

3. **Memory Efficiency Benchmark**
   - Track actual memory usage (RSS, heap)
   - Compare storage of sparse vs dense
   - Measure cache misses and bandwidth
   - Show 60-80% memory savings claim

4. **Sparse Tensor Operations**
   - Various sparsity levels (10%, 50%, 90%, 99%)
   - Operations: add, multiply, convolve
   - Show computational savings
   - Compare vs CSR, COO formats

5. **Broadcasting Semantics**
   - NumPy-compatible broadcasting rules
   - Performance vs explicit loops
   - Memory usage vs temporary copies

## Benchmark Quality Criteria

Each benchmark must:
- Use realistic problem sizes (minimum 100k elements)
- Run sufficient iterations for statistical validity
- Measure with microsecond precision or better
- Include memory usage metrics
- Test actual multi-dimensional scenarios
- Compare against relevant alternatives

## Paper Benchmark Suite

The `run_paper_benchmarks.sh` should execute:
1. ✅ Large-scale vector addition
2. ✅ FFT convolution performance  
3. ❌ Multi-dimensional shape mismatches (TODO)
4. ❌ Sensor fusion scenario (TODO)
5. ❌ Memory efficiency analysis (TODO)
6. ✅ Window/pyramid stacking
7. ❌ Sparse tensor operations (TODO)

## Expected Results

VSLA should demonstrate:
- **2-5x faster** than zero-padding for shape mismatches
- **60-80% less memory** for sparse/mismatched tensors
- **10-100x faster** for high sparsity operations
- **Superior scaling** with tensor dimensions
- **Practical advantages** in real-world scenarios

## Implementation Priority

1. **High Priority**:
   - Multi-dimensional shape mismatch benchmark
   - Sensor fusion benchmark
   - Update run_paper_benchmarks.sh

2. **Medium Priority**:
   - Memory efficiency tracking
   - Sparse tensor benchmarks
   - Broadcasting semantics

3. **Low Priority**:
   - Expand Kronecker to larger sizes
   - Add more stacking scenarios
   - GPU comparison benchmarks