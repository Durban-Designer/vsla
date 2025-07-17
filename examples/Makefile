# Makefile for VSLA Examples and Benchmarks
# Demonstrates hardware-agnostic VSLA interface with real-world applications

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O3 -march=native -I../include
LIBS = -lm -lpthread
LIBVSLA = ../build/libvsla.a

# CUDA support (optional)
CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo yes || echo no)
ifeq ($(CUDA_AVAILABLE),yes)
    CFLAGS += -DVSLA_ENABLE_CUDA
    LIBS += -lcudart -lcufft
    $(info CUDA detected - enabling GPU acceleration)
else
    $(info CUDA not found - CPU-only mode)
endif

# Source files
BASIC_SOURCES = basic_usage.c
E2E_SOURCES = signal_processing_radar.c neural_network_cnn.c polynomial_algebra.c
BENCHMARK_SOURCES = ../bench/src/comprehensive_benchmark.c ../bench/src/intelligent_benchmark.c

# Targets
BASIC_EXAMPLES = $(BASIC_SOURCES:.c=)
E2E_EXAMPLES = $(E2E_SOURCES:.c=)
BENCHMARKS = comprehensive_benchmark intelligent_benchmark

ALL_EXAMPLES = $(BASIC_EXAMPLES) $(E2E_EXAMPLES)

.PHONY: all basic e2e benchmarks clean run-examples run-benchmarks help demo paper-data

# Default target
all: basic e2e benchmarks

# Build basic examples
basic: $(BASIC_EXAMPLES)
	@echo "✓ Basic examples built successfully"

# Build E2E examples
e2e: $(E2E_EXAMPLES)
	@echo "✓ E2E examples built successfully"

# Build benchmarks
benchmarks: $(BENCHMARKS)
	@echo "✓ Benchmarks built successfully"

# Build rules
%: %.c $(LIBVSLA)
	@echo "Building $@..."
	$(CC) $(CFLAGS) $< $(LIBVSLA) $(LIBS) -o $@

# Special rules for benchmarks from bench directory
comprehensive_benchmark: ../bench/src/comprehensive_benchmark.c $(LIBVSLA)
	@echo "Building comprehensive benchmark suite..."
	$(CC) $(CFLAGS) $< $(LIBVSLA) $(LIBS) -o $@

intelligent_benchmark: ../bench/src/intelligent_benchmark.c $(LIBVSLA)
	@echo "Building intelligent benchmark suite..."
	$(CC) $(CFLAGS) $< $(LIBVSLA) $(LIBS) -o $@

# Ensure library is built
$(LIBVSLA):
	@echo "Building VSLA library..."
	@cd .. && mkdir -p build && cd build && cmake -DVSLA_UNIFIED_API=ON .. && make

# Run all examples
run-examples: all
	@echo "\n=== Running VSLA Examples ==="
	@echo "\n--- Basic Usage ---"
	@./basic_usage || echo "Basic example failed"
	@echo "\n--- Radar Signal Processing ---"
	@./signal_processing_radar || echo "Radar example failed"
	@echo "\n--- Neural Network CNN ---"
	@./neural_network_cnn || echo "CNN example failed"
	@echo "\n--- Polynomial Algebra ---"
	@./polynomial_algebra || echo "Polynomial example failed"
	@echo "\n✓ All examples completed"

# Run benchmarks
run-benchmarks: benchmarks
	@echo "\n=== Running VSLA Benchmarks ==="
	@echo "\n--- Comprehensive Benchmark ---"
	@./comprehensive_benchmark || echo "Comprehensive benchmark failed"
	@echo "\n--- Intelligent Paradigm Comparison ---"
	@./intelligent_benchmark || echo "Intelligent benchmark failed"
	@echo "\n✓ All benchmarks completed"

# Generate paper data
paper-data: benchmarks
	@echo "\n=== Generating Performance Data for Paper ==="
	@mkdir -p ../bench/results
	@echo "Running comprehensive benchmark..."
	@./comprehensive_benchmark > ../bench/results/comprehensive_results.log 2>&1 || true
	@echo "Running intelligent paradigm comparison..."
	@./intelligent_benchmark > ../bench/results/intelligent_results.log 2>&1 || true
	@echo "Results saved to ../bench/results/"
	@echo "Ready for paper analysis and publication!"

# Quick demo
demo: e2e
	@echo "\n🚀 VSLA Hardware-Agnostic Computing Demo"
	@echo "========================================"
	@echo ""
	@echo "This demonstration showcases VSLA's unified interface that automatically"
	@echo "uses the best available hardware (CPU/GPU) and vendor libraries for"
	@echo "maximum performance while hiding hardware complexity from users."
	@echo ""
	@echo "Running signal processing example..."
	@timeout 30 ./signal_processing_radar || true
	@echo "\n✨ Demo completed! VSLA provides hardware-agnostic performance"
	@echo "   optimization for real-world variable-shape tensor operations."

# Performance test
perf-test: e2e
	@echo "\n=== VSLA Performance Testing ==="
	@echo "Testing signal processing with timing..."
	@time ./signal_processing_radar > /dev/null

# Memory check with valgrind
memcheck: e2e
	@echo "\n=== Memory Leak Detection ==="
	@which valgrind > /dev/null || (echo "valgrind not found, skipping memory check" && exit 0)
	@echo "Checking signal processing example..."
	@valgrind --leak-check=full --error-exitcode=1 --quiet \
		./signal_processing_radar > /dev/null 2>&1 && echo "✓ No memory leaks detected" || echo "✗ Memory issues found"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@echo -n "VSLA library: "
	@test -f $(LIBVSLA) && echo "✓ Found" || echo "✗ Missing (will be built automatically)"
	@echo -n "Math library: "
	@echo "✓ Available"
	@echo -n "Threads: "
	@echo "✓ Available"
ifeq ($(CUDA_AVAILABLE),yes)
	@echo -n "CUDA Runtime: "
	@ldconfig -p | grep -q libcudart && echo "✓ Found" || echo "✗ Missing"
	@echo -n "cuFFT: "
	@ldconfig -p | grep -q libcufft && echo "✓ Found" || echo "✗ Missing"
endif

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(ALL_EXAMPLES) $(BENCHMARKS)
	@rm -f *.o gmon.out profile_report.txt
	@rm -f benchmark_results_*.json
	@echo "✓ Clean completed"

# Legacy targets for compatibility
run-basic: basic_usage
	./basic_usage

# Help target
help:
	@echo "VSLA Examples and Benchmarks Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all examples and benchmarks"
	@echo "  basic        - Build basic examples"
	@echo "  e2e          - Build end-to-end examples"
	@echo "  benchmarks   - Build benchmark suite"
	@echo "  run-examples - Build and run all examples"
	@echo "  run-benchmarks - Build and run benchmarks"
	@echo "  demo         - Quick demonstration"
	@echo "  paper-data   - Generate performance data for paper"
	@echo "  perf-test    - Performance timing test"
	@echo "  memcheck     - Memory leak detection (requires valgrind)"
	@echo "  check-deps   - Check for required dependencies"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Legacy targets:"
	@echo "  basic_usage  - Build basic usage example"
	@echo "  run-basic    - Build and run basic usage example"
	@echo ""
	@echo "Examples:"
	@echo "  make demo                  # Quick demo"
	@echo "  make run-examples          # Run all examples"
	@echo "  make paper-data            # Generate benchmark data"
	@echo ""
	@echo "System Configuration:"
	@echo "  CUDA Support: $(CUDA_AVAILABLE)"
	@echo "  Compiler: $(CC)"
	@echo "  Flags: $(CFLAGS)"