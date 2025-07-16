# VSLA Benchmark Suite - Complete Usage Guide

## Quick Start

Run the comprehensive benchmark with a single command:

```bash
./benchmark.sh
```

## Options

- `--quick` - Fast benchmark with smaller sizes (64, 128, 256) and fewer iterations
- `--sizes SIZE1,SIZE2,...` - Custom test sizes (default: 128,256,512,1024)
- `--iterations N` - Number of iterations per test (default: 50, quick: 10)
- `--no-gpu` - Disable GPU benchmarks
- `--no-cpu` - Disable CPU benchmarks  
- `--no-competitors` - Disable competitor benchmarks

## Examples

```bash
# Quick benchmark (recommended for testing)
./benchmark.sh --quick

# Full benchmark with default settings
./benchmark.sh

# CPU-only benchmark
./benchmark.sh --no-gpu --no-competitors

# Custom sizes with high iteration count
./benchmark.sh --sizes 512,1024,2048 --iterations 100

# GPU vs competitors only
./benchmark.sh --no-cpu
```

## What Gets Benchmarked

### CPU Operations (VSLA Native)
- **Vector Addition**: `vsla_add()` - Element-wise vector addition
- **Convolution**: `vsla_conv()` - Model A convolution operation 
- **Kronecker Product**: `vsla_kron()` - Model B Kronecker operation

### GPU Operations (VSLA CUDA)
- **Vector Addition**: `vsla_gpu_add()` - GPU-accelerated vector addition
- **Matrix Multiplication**: `vsla_gpu_matmul()` - GPU matrix operations

### Competitor Libraries
- **cuBLAS**: Industry-standard CUDA linear algebra (vector ops + GEMM)
- **CuPy**: NumPy-compatible GPU arrays (when available)

## Output

Each benchmark run generates:

1. **JSON Results**: Raw benchmark data with statistical analysis
   - Filename: `vsla_benchmark_{SYSTEM}_{TIMESTAMP}.json`
   - Location: `bench/reports/`

2. **Markdown Report**: Human-readable performance analysis
   - Filename: `vsla_benchmark_{SYSTEM}_{TIMESTAMP}.md`
   - Location: `bench/reports/`

## Report Contents

- **Performance Tables**: Timing results for all operations
- **GFLOPS Calculations**: Computational throughput for matrix operations
- **Speedup Analysis**: VSLA performance vs competitors
- **System Configuration**: Hardware/software environment
- **Reproducibility Info**: Exact commands to reproduce results

## System Info in Filenames

Report filenames automatically include system information:
- CPU model (simplified)
- GPU model (simplified) 
- Total RAM
- Timestamp

Example: `vsla_benchmark_13thGeni9-13900HX_RTX4060GPU_15GB_20250716_140525.md`

## Benchmark Structure

```
bench/
├── run_benchmark.py          # Main benchmark orchestrator
├── src/
│   ├── cpu_benchmark.c       # CPU benchmark implementation
│   ├── gpu_head_to_head.c    # GPU benchmark implementation
│   └── ...                   # Other benchmark sources
├── build/
│   ├── cpu_benchmark         # CPU benchmark binary
│   ├── gpu_head_to_head      # GPU benchmark binary
│   ├── cublas_benchmark      # cuBLAS competitor binary
│   └── ...                   # Other built benchmarks
├── competitors/
│   ├── cublas_benchmark.c    # cuBLAS benchmark source
│   ├── cupy_benchmark.py     # CuPy benchmark (if available)
│   └── ...
└── reports/                  # Generated benchmark reports
    ├── *.json               # Raw results
    └── *.md                 # Formatted reports
```

## Key Performance Results

From our latest benchmarks on 13th Gen i9-13900HX + RTX 4060:

### GPU Performance Highlights
- **Vector Addition**: 1.14× to 1.40× faster than cuBLAS
- **Matrix Multiplication**: 3.56× to 5.75× faster than cuBLAS
- **Peak Performance**: 791 GFLOPS (256×256 matrices)

### CPU Performance
- **Vector Addition**: 1.5-6.4μs (depending on size)
- **Convolution**: Efficient FFT-based implementation
- **Kronecker**: Consistent ~26μs across sizes

## Dependencies

### Required
- GCC compiler
- VSLA library (built)
- CUDA 12.6+ (for GPU benchmarks)

### Optional
- cuBLAS (for competitor benchmarks) - ✅ Available
- CuPy (for Python GPU benchmarks) - ⚠️ Install needed

## Installing Missing Dependencies

### CuPy (Optional)
```bash
pip install cupy-cuda12x --break-system-packages
# OR use system packages if available
```

## Troubleshooting

### GPU Benchmarks Fail
- Check CUDA installation: `nvidia-smi`
- Verify CUDA path: `/usr/local/cuda-12.6/bin/nvcc --version`
- Ensure GPU is available: `nvidia-smi`

### Competitor Benchmarks Fail
- cuBLAS: Check if `bench/build/cublas_benchmark` exists
- CuPy: Install with pip or disable with `--no-competitors`

### Build Errors
- Ensure VSLA is built: `cd .. && make`
- Check include paths in build commands
- Verify CUDA library paths

## Contributing

To add new benchmarks:

1. Add source to `bench/src/`
2. Update build process in `bench/CMakeLists.txt`
3. Integrate into `run_benchmark.py`
4. Test with `./benchmark.sh --quick`

## Performance Analysis

The benchmark suite provides comprehensive analysis:

- **Statistical reliability**: Multiple iterations with std dev
- **Comparative analysis**: Direct speedup calculations
- **System reproducibility**: Complete environment documentation
- **GFLOPS calculations**: Computational throughput metrics

Perfect for:
- Performance regression testing
- Hardware comparison
- Algorithm optimization validation
- Academic/research publications