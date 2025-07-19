# VSLA Repository Structure

This document describes the organization of the VSLA repository.

## Root Directory Structure

```
vsla/
├── bench/                    # Benchmarking framework and results
├── docs/                     # Project documentation and papers
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

All project documentation is centralized in the `docs/` folder:

- `papers/` - Academic papers (LaTeX source and PDFs)

## Development Workflow

1. **Building**: Use CMake from project root: `mkdir build && cd build && cmake .. && make`
2. **Testing**: Run `ctest` from build directory or individual test executables
3. **Benchmarking**: Use scripts in `bench/` directory
4. **Documentation**: The main source of documentation is the paper in `docs/papers/`
5. **Examples**: C examples in `examples/`, Python examples in `python/`

This organization follows standard C/C++ project conventions and keeps the repository clean and navigable.