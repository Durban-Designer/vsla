# VSLA Benchmark Suite

This directory contains comprehensive benchmarks for the Variable-Shape Linear Algebra (VSLA) library, designed to validate theoretical complexity claims and populate Table 2 in the research paper.

## Overview

The benchmark suite measures VSLA performance against traditional approaches across key operations:

- **Vector Addition**: Variable-shape vs manual padding + BLAS
- **Matrix-Vector Multiplication**: Model A convolution vs standard approaches  
- **Kronecker Products**: Model B vs direct implementations
- **FFT Convolution**: VSLA's FFT vs NumPy/SciPy implementations

## Benchmark Strategy

### Core Metrics
- **Wall-clock time**: Primary performance measure (microseconds)
- **Memory usage**: Peak RSS and allocation patterns
- **Scalability**: Performance vs dimension size (64, 256, 1024, 4096, 16384)
- **Statistical analysis**: Mean, std deviation, confidence intervals

### Competitors (Top 3 GPU Libraries)
- **CuPy**: GPU-accelerated NumPy equivalent with CUDA
- **cuBLAS**: NVIDIA's GPU-optimized BLAS library
- **cuFFT**: NVIDIA's GPU-accelerated FFT library
- **Legacy baselines**: OpenBLAS, NumPy/SciPy, FFTW for reference

### Benchmarked Operations
- **Vector Addition**: Element-wise addition with variable shapes
- **Matrix Multiplication**: Dense matrix operations
- **Convolution**: FFT-based convolution for signal processing

### Test Matrix

| Operation | VSLA Method | Baseline | Dimensions Tested |
|-----------|-------------|----------|-------------------|
| Vector Add | Automatic padding | Manual pad + BLAS | 64-16K |
| Matrix-Vec | Model A (FFT conv) | BLAS gemv | 64x64 - 1Kx1K |
| Kronecker | Model B (tiled) | Direct product | 32x32 - 512x512 |
| Convolution | Custom FFT | NumPy convolve | 64-8K |

## Directory Structure

```
bench/
├── README.md              # This file
├── CMakeLists.txt          # Benchmark build system
├── run_benchmarks.py       # Master benchmark runner
├── results/                # Benchmark outputs
│   ├── 2025-07-v1/        # Timestamped results for paper
│   └── latest/            # Most recent results
├── src/                   # Benchmark implementations
│   ├── bench_vector_add.c  # Variable-shape addition
│   ├── bench_matvec.c      # Matrix-vector operations
│   ├── bench_kronecker.c   # Kronecker products
│   ├── bench_convolution.c # FFT convolution tests
│   └── benchmark_utils.h   # Common timing/memory utilities
├── baselines/             # Reference implementations
│   ├── numpy_baselines.py  # NumPy/SciPy comparisons
│   ├── blas_baselines.c    # OpenBLAS comparisons
│   └── naive_baselines.c   # Direct algorithm implementations
├── scripts/               # Analysis and plotting
│   ├── analyze_results.py  # Statistical analysis
│   ├── plot_performance.py # Generate performance graphs
│   └── generate_table2.py  # Create Table 2 for paper
└── data/                  # Test datasets
    ├── synthetic/         # Generated test data
    └── real_world/        # Application-specific datasets
```

## Building and Running

### Prerequisites
```bash
# Install dependencies
sudo apt-get install libblas-dev libopenblas-dev libfftw3-dev
pip install numpy scipy matplotlib pandas

# Build VSLA library first
cd /home/kenth56/vsla
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Build Benchmarks
```bash
cd /home/kenth56/vsla/bench
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Run All Benchmarks

#### Comprehensive Competitor Benchmark (Recommended)
```bash
# Single command to run full competitive analysis
python run_full_benchmark.py --reproducible

# Custom configuration
python run_full_benchmark.py \
    --sizes 64,256,1024,4096 \
    --iterations 100 \
    --competitors cupy,cublas,cufft \
    --output-dir results/$(date +%Y-%m-%d) \
    --enable-gpu
```

#### Individual Benchmark Components
```bash
# Original benchmark suite (legacy)
python run_benchmarks.py --output results/$(date +%Y-%m-%d)

# Specific operation benchmarks
./build/bench_vector_add --sizes 64,256,1024 --iterations 1000
./build/bench_matvec --matrices small,medium,large --methods vsla,blas
./build/bench_kronecker --dimensions 32,64,128 --compare-all
./build/bench_convolution --signals 256,512,1024 --fft-comparison
```

#### Competitor Setup
```bash
# Build GPU competitor benchmarks (requires CUDA)
cd competitors/
make all

# Test competitor availability
make check-cuda
make test
```

### Generate Results
```bash
# Comprehensive benchmark generates automatic report
python run_full_benchmark.py --reproducible
# Output: benchmark_results.json + comprehensive_report.md

# Legacy analysis tools
python scripts/analyze_results.py results/latest/
python scripts/generate_table2.py --output table2.tex

# Create performance plots
python scripts/plot_performance.py --input results/latest/ --output plots/
```

### Benchmark Output

The comprehensive benchmark generates:
- **benchmark_results.json**: Raw performance data with system info
- **comprehensive_report.md**: Executive summary and detailed analysis
- **Competitive analysis**: Direct comparison with CuPy, cuBLAS, cuFFT
- **Reproducibility info**: Complete environment specification

## Benchmark Implementations

### 1. Vector Addition (`bench_vector_add.c`)
Tests variable-shape addition vs manual padding approaches.

```c
// VSLA approach: automatic padding
vsla_add(result, a, b);  // Different shapes handled automatically

// Baseline: manual padding + BLAS
pad_vectors(a_padded, b_padded, max_size);
cblas_daxpy(max_size, 1.0, a_padded, 1, b_padded, 1);
```

**Measured**: Time vs dimension difference, memory overhead, cache performance

### 2. Matrix-Vector Multiplication (`bench_matvec.c`)
Compares Model A convolution-based matrix multiplication vs BLAS.

```c
// VSLA Model A: O(mn d_max log d_max)
vsla_conv(result, matrix_row, vector);  // Per row, then sum

// Baseline: O(mn d_max^2)  
cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, x, 1, 0.0, y, 1);
```

**Measured**: Time vs matrix size, crossover point for FFT advantage

### 3. Kronecker Products (`bench_kronecker.c`)
Tests Model B tiled implementation vs direct Kronecker products.

```c
// VSLA Model B: Cache-friendly tiled approach
vsla_kron_tiled(result, a, b, tile_size);

// Baseline: Direct Kronecker product
kron_naive(result, a, b);  // Standard textbook algorithm
```

**Measured**: Time vs tensor size, memory bandwidth utilization

### 4. FFT Convolution (`bench_convolution.c`)
Compares VSLA's custom FFT vs established libraries.

```c
// VSLA: Integrated FFT with variable shapes
vsla_conv_fft(result, signal, kernel);

// Baseline: NumPy/FFTW
numpy.convolve(signal, kernel, mode='full')  // Via Python C API
```

**Measured**: Time vs signal length, accuracy comparison, memory usage

## Timing Infrastructure

### High-Resolution Timing
```c
typedef struct {
    double wall_time;
    double cpu_time; 
    size_t peak_memory;
    uint64_t cache_misses;  // Optional
} benchmark_result_t;

static inline double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
```

### Memory Measurement
```c
static size_t get_peak_memory(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024;  // Convert to bytes
}
```

### Statistical Analysis
- **Warm-up iterations**: 5 runs to stabilize caches
- **Measurement iterations**: 50+ runs for statistical significance
- **Outlier detection**: Remove top/bottom 5% of measurements
- **Confidence intervals**: 95% confidence bounds
- **Effect size**: Cohen's d for practical significance

## Result Format

Benchmarks output JSON for easy analysis:

```json
{
  "benchmark": "vector_add",
  "method": "vsla_auto_pad",
  "dimensions": [64, 128],
  "iterations": 1000,
  "results": {
    "mean_time_us": 12.34,
    "std_time_us": 0.89,
    "min_time_us": 11.12,
    "max_time_us": 15.67,
    "peak_memory_mb": 0.125,
    "cache_miss_rate": 0.023
  },
  "system_info": {
    "cpu": "Intel Xeon E5-2680 v4",
    "memory": "64GB DDR4-2400",
    "compiler": "GCC 11.2.0",
    "blas": "OpenBLAS 0.3.21"
  }
}
```

## Expected Results

Based on theoretical analysis, we expect:

### Vector Addition
- **VSLA advantage**: ~2x faster for mixed dimensions due to cache-friendly access
- **Memory**: 20-30% lower peak usage (no duplicate padded storage)

### Matrix-Vector (Model A)
- **Crossover point**: d_max ≈ 64 where FFT becomes advantageous
- **Large matrices**: 5-10x speedup for d_max > 256
- **Small matrices**: 10-20% overhead due to FFT setup costs

### Kronecker Products (Model B)  
- **Tiled advantage**: 3-5x speedup for large tensors (>1024 elements)
- **Cache efficiency**: 50-80% reduction in cache misses
- **Memory bandwidth**: Better utilization of available bandwidth

### FFT Convolution
- **Accuracy**: Within 1e-12 of NumPy (double precision)
- **Performance**: Competitive with FFTW for medium signals (512-2048)
- **Memory**: Lower allocation overhead due to variable-shape design

## Reproducibility

### Environment Specification
- **OS**: Ubuntu 20.04+ or equivalent
- **Compiler**: GCC 9+ or Clang 10+
- **Dependencies**: Locked versions in `requirements.txt`
- **Hardware**: x86_64 with AVX2 support recommended

### Result Validation
```bash
# Generate reproducible results
export OMP_NUM_THREADS=1          # Disable threading
export OPENBLAS_NUM_THREADS=1     # Single-threaded BLAS
export MKL_NUM_THREADS=1          # Single-threaded MKL

# Set CPU governor for consistent performance  
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Run benchmarks
python run_benchmarks.py --reproducible --seed 42
```

### Data Archival
- Results timestamped and stored in `results/YYYY-MM-DD/`
- Git commit hash recorded with each benchmark run
- System configuration captured automatically
- Raw data preserved alongside processed results

## Usage in Paper

Results from this benchmark suite directly populate:

- **Table 2**: Complexity comparison with real measurements
- **Figure 3**: Performance scaling graphs  
- **Section 8**: Empirical evaluation discussion
- **Appendix B**: Detailed benchmark methodology

To regenerate Table 2 for the paper:
```bash
python scripts/generate_table2.py --input results/2025-07-v1/ --format latex
```

This produces LaTeX-formatted table ready for inclusion in `vsla_paper.tex`.

## Contributing

When adding new benchmarks:

1. **Follow naming convention**: `bench_<operation>.c`
2. **Use common utilities**: Include `benchmark_utils.h`
3. **Output JSON format**: For automated analysis
4. **Add baseline comparison**: Always compare vs established method
5. **Document methodology**: Update this README with new benchmark details
6. **Validate results**: Ensure correctness before measuring performance

## Troubleshooting

### Common Issues

**Benchmark crashes with large dimensions**:
- Check system memory limits (`ulimit -v`)
- Reduce maximum test dimension
- Enable memory debugging (`valgrind --tool=massif`)

**Inconsistent timing results**:
- Disable CPU frequency scaling
- Check for background processes
- Increase iteration count
- Run on dedicated machine

**Missing baseline libraries**:
```bash
# Install missing dependencies
sudo apt-get install libblas-dev liblapack-dev libfftw3-dev
pip install numpy scipy
```

**Build failures**:
- Ensure VSLA library built successfully first
- Check CMake cache for stale configuration
- Verify compiler supports C99

For additional help, see [VSLA documentation](../docs/) or [open an issue](https://github.com/username/vsla/issues).