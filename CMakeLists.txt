cmake_minimum_required(VERSION 3.12)
project(vsla C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Backend Build Options ---
option(VSLA_BUILD_CPU "Build the CPU backend" ON)
option(VSLA_BUILD_CUDA "Build the CUDA backend" ON)

# --- Core Library Sources ---
set(VSLA_CORE_SOURCES
    src/vsla_core.c
    src/vsla_tensor.c
    src/vsla_unified.c
)

# --- CPU Backend Sources ---
if(VSLA_BUILD_CPU)
    list(APPEND VSLA_CORE_SOURCES src/backends/vsla_backend_cpu.c)
    add_compile_definitions(VSLA_BUILD_CPU=1)
endif()

# --- CUDA Backend Sources ---
if(VSLA_BUILD_CUDA)
    # Check if CUDA language is available before enabling it
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        
        # Set CUDA architecture to auto-detect from native GPU
        set(CMAKE_CUDA_ARCHITECTURES native)
        
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA found, enabling CUDA backend with native GPU architecture.")
            list(APPEND VSLA_CORE_SOURCES 
                src/backends/vsla_backend_cuda.c 
                src/backends/cuda/vsla_gpu.cu
                src/backends/cuda/vsla_tensor_utils.c
            )
        else()
            message(WARNING "CUDA compiler found but CUDAToolkit not found. CUDA backend will not be built.")
        endif()
    else()
        message(WARNING "VSLA_BUILD_CUDA is ON but CUDA compiler was not found. CUDA backend will not be built.")
    endif()
endif()

# --- VSLA Library Target ---
add_library(vsla_static STATIC ${VSLA_CORE_SOURCES})
target_include_directories(vsla_static PUBLIC include)

# Link CUDA libraries if the backend was built
if(VSLA_BUILD_CUDA AND CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    target_link_libraries(vsla_static PRIVATE CUDA::cudart)
endif()

# Benchmarks
add_subdirectory(benchmarks)

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS vsla_static DESTINATION lib)
install(DIRECTORY include/vsla DESTINATION include)