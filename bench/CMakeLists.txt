cmake_minimum_required(VERSION 3.12)
project(vsla_benchmarks C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required libraries
find_package(PkgConfig REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)

# Try to find FFTW3
pkg_check_modules(FFTW3 fftw3)

# Include VSLA headers
include_directories(../include)

# Link to VSLA library
link_directories(../build)

# Common benchmark utilities
add_library(benchmark_utils STATIC src/benchmark_utils.c)
target_include_directories(benchmark_utils PUBLIC src)

# Compiler flags for benchmarks
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Link libraries for all benchmarks
set(BENCHMARK_LIBS benchmark_utils vsla ${BLAS_LIBRARIES} m)

if(FFTW3_FOUND)
    list(APPEND BENCHMARK_LIBS ${FFTW3_LIBRARIES})
    add_definitions(-DHAVE_FFTW3)
endif()

# Vector addition benchmark (TODO: implement)
# add_executable(bench_vector_add src/bench_vector_add.c)
# target_link_libraries(bench_vector_add ${BENCHMARK_LIBS})

# Matrix-vector multiplication benchmark (TODO: implement)
# add_executable(bench_matvec src/bench_matvec.c)
# target_link_libraries(bench_matvec ${BENCHMARK_LIBS})

# Kronecker product benchmark (TODO: implement)
# add_executable(bench_kronecker src/bench_kronecker.c)
# target_link_libraries(bench_kronecker ${BENCHMARK_LIBS})

# Convolution benchmark
add_executable(bench_convolution src/bench_convolution.c src/benchmark_utils.c)
target_link_libraries(bench_convolution ${BENCHMARK_LIBS})

# Fair comparison benchmark (VSLA vs manual padding)
add_executable(bench_comparison src/bench_comparison.c src/benchmark_utils.c)
target_link_libraries(bench_comparison ${BENCHMARK_LIBS})

# Fair comparison against established C libraries
add_executable(bench_fair_comparison src/bench_fair_comparison.c src/benchmark_utils.c)
target_link_libraries(bench_fair_comparison ${BENCHMARK_LIBS})

# Master benchmark runner (TODO: implement)
# add_executable(run_all_benchmarks src/run_all_benchmarks.c)
# target_link_libraries(run_all_benchmarks ${BENCHMARK_LIBS})

# Create directories for results
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/results)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/results/latest)

# Custom targets
add_custom_target(bench_all
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_all_benchmarks
    DEPENDS run_all_benchmarks bench_vector_add bench_matvec bench_kronecker bench_convolution
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all VSLA benchmarks"
)

# Install benchmarks
install(TARGETS bench_convolution
    RUNTIME DESTINATION bin
)

# Copy Python scripts to build directory
configure_file(run_benchmarks.py ${CMAKE_CURRENT_BINARY_DIR}/run_benchmarks.py COPYONLY)
configure_file(scripts/analyze_results.py ${CMAKE_CURRENT_BINARY_DIR}/analyze_results.py COPYONLY)
configure_file(scripts/generate_table2.py ${CMAKE_CURRENT_BINARY_DIR}/generate_table2.py COPYONLY)