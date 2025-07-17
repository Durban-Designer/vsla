# VSLA Repository Structure

This document describes the organization of the VSLA repository after cleanup.

## Root Directory Structure

```
vsla/
├── bench/                    # Benchmarking framework and results
├── docs/                     # All project documentation  
├── examples/                 # Usage examples and demos
├── include/                  # Public C headers
├── python/                   # Python bindings
├── src/                      # Core C implementation
├── tests/                    # Unit and integration tests
├── CMakeLists.txt           # Build configuration
├── README.md                # Main project documentation
├── STATUS.md                # Current development status
├── LICENSE                  # MIT license
└── setup.py                 # Python package setup
```

## Documentation Organization (docs/)

All documentation is now centralized in the `docs/` folder:

- `API_REFERENCE.md` - Complete API documentation
- `ARCHITECTURE.md` - System architecture overview  
- `BENCHMARK_REPORT.md` - Performance analysis
- `BENCHMARK_USAGE.md` - How to run benchmarks
- `CORE_FEATURES.md` - Feature documentation
- `CUDA_C23_MIGRATION.md` - GPU implementation notes
- `FINAL_GPU_PERFORMANCE_REPORT.md` - GPU benchmarks
- `GPU_IMPLEMENTATION.md` - GPU-specific documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `VALIDATION.md` - Testing and validation procedures
- `papers/` - Academic papers (LaTeX source and PDFs)

## Build Artifacts

The following are excluded via .gitignore:

- `build*/` - CMake build directories
- `bin/` - Python virtual environment binaries
- `lib/`, `lib64` - Python virtual environment libraries  
- `test_stack_*`, `debug_stack*` - Temporary test files
- All standard build artifacts (*.o, *.so, executables, etc.)

## Development Workflow

1. **Building**: Use CMake from project root: `mkdir build && cd build && cmake .. && make`
2. **Testing**: Run `ctest` from build directory or individual test executables
3. **Benchmarking**: Use scripts in `bench/` directory
4. **Documentation**: Papers in `docs/papers/`, general docs in `docs/`
5. **Examples**: C examples in `examples/`, Python examples in `python/`

This organization follows standard C/C++ project conventions and keeps the repository clean and navigable.