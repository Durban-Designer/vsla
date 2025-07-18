cmake_minimum_required(VERSION 3.14)
project(libvsla VERSION 1.0.0 LANGUAGES C CXX)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(VSLA_BUILD_PYTHON "Build Python bindings" OFF)
option(VSLA_ENABLE_TESTS "Enable testing" ON)
option(VSLA_ENABLE_BENCHMARKS "Enable benchmarks" OFF)
option(USE_FFTW "Use FFTW library for FFT operations" OFF)
option(VSLA_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(VSLA_ENABLE_CUFFT "Enable cuFFT for GPU FFT operations" OFF)
option(VSLA_ENABLE_ROCFFT "Enable rocFFT for AMD GPU FFT operations" OFF)
option(VSLA_ENABLE_MKL "Enable Intel MKL for CPU FFT operations" OFF)
option(VSLA_UNIFIED_API "Build unified hardware-agnostic API" ON)

# C99 standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wstrict-prototypes)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)
if(USE_FFTW)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW fftw3)
    endif()
endif()

# Vendor FFT libraries
if(VSLA_ENABLE_CUFFT)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CUDAToolkit_cufft_FOUND)
        message(FATAL_ERROR "cuFFT not found but VSLA_ENABLE_CUFFT is ON")
    endif()
endif()

if(VSLA_ENABLE_ROCFFT)
    find_package(rocfft QUIET)
    if(NOT rocfft_FOUND)
        message(WARNING "rocFFT not found, disabling rocFFT support")
        set(VSLA_ENABLE_ROCFFT OFF)
    endif()
endif()

if(VSLA_ENABLE_MKL)
    find_package(MKL QUIET)
    if(NOT MKL_FOUND)
        find_package(PkgConfig)
        if(PkgConfig_FOUND)
            pkg_check_modules(MKL mkl-dynamic-lp64)
        endif()
        if(NOT MKL_FOUND)
            message(WARNING "Intel MKL not found, disabling MKL support")
            set(VSLA_ENABLE_MKL OFF)
        endif()
    endif()
endif()

# CUDA support
if(VSLA_ENABLE_CUDA)
    # Set environment variables before enabling CUDA language
    set(ENV{CUDAHOSTCXX} "/usr/bin/g++")
    set(ENV{CUDACXX} "/usr/local/cuda-12.6/bin/nvcc")
    
    # Set flags before enabling the language
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler -std=c++17")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_TYPES_EXT__=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_FUNCS_EXT__=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_ATTRIBS_EXT__=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_BFP_EXT__=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_DFP_EXT__=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STDC_WANT_IEC_60559_EXT__=0")
    
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Use older CUDA architecture that's more compatible
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(VSLA_SOURCES
    src/vsla_core.c
    src/vsla_tensor.c
    src/vsla_io.c
    src/vsla_conv.c
    src/vsla_kron.c
    src/vsla_autograd.c
    src/vsla_utils.c
    src/backends/vsla_backend_cpu_new.c
    src/backends/vsla_backend_cuda_new.c
)

# Add unified API sources if enabled
# Temporarily disabled until architecture is fully aligned
#if(VSLA_UNIFIED_API)
#    list(APPEND VSLA_SOURCES 
#        src/vsla_unified.c
#    )
#endif()

# Add vendor FFT sources
if(VSLA_ENABLE_CUFFT)
    list(APPEND VSLA_SOURCES src/vsla_fft_cufft.c)
endif()

if(VSLA_ENABLE_ROCFFT)
    list(APPEND VSLA_SOURCES src/vsla_fft_rocfft.c)
endif()

if(VSLA_ENABLE_MKL)
    list(APPEND VSLA_SOURCES src/vsla_fft_mkl.c)
endif()

# Add GPU sources conditionally
if(VSLA_ENABLE_CUDA)
    list(APPEND VSLA_SOURCES src/backends/vsla_backend_cuda_kernels.cu)
    list(APPEND VSLA_SOURCES src/vsla_gpu.cu)
else()
    list(APPEND VSLA_SOURCES src/vsla_gpu.c)
endif()

# Create library targets
add_library(vsla_static STATIC ${VSLA_SOURCES})
set_target_properties(vsla_static PROPERTIES OUTPUT_NAME vsla)
target_link_libraries(vsla_static PRIVATE Threads::Threads m)

if(BUILD_SHARED_LIBS)
    add_library(vsla_shared SHARED ${VSLA_SOURCES})
    set_target_properties(vsla_shared PROPERTIES OUTPUT_NAME vsla)
    target_link_libraries(vsla_shared PRIVATE Threads::Threads m)
    if(USE_FFTW AND FFTW_FOUND)
        target_link_libraries(vsla_shared PRIVATE ${FFTW_LIBRARIES})
        target_include_directories(vsla_shared PRIVATE ${FFTW_INCLUDE_DIRS})
        target_compile_definitions(vsla_shared PRIVATE USE_FFTW)
    endif()
endif()

if(USE_FFTW AND FFTW_FOUND)
    target_link_libraries(vsla_static PRIVATE ${FFTW_LIBRARIES})
    target_include_directories(vsla_static PRIVATE ${FFTW_INCLUDE_DIRS})
    target_compile_definitions(vsla_static PRIVATE USE_FFTW)
endif()

# CUDA configuration
if(VSLA_ENABLE_CUDA)
    target_compile_definitions(vsla_static PRIVATE VSLA_ENABLE_CUDA)
    target_link_libraries(vsla_static PRIVATE CUDA::cudart)
    
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(vsla_shared PRIVATE VSLA_ENABLE_CUDA)
        target_link_libraries(vsla_shared PRIVATE CUDA::cudart)
    endif()
endif()

# Vendor FFT library configuration
if(VSLA_ENABLE_CUFFT)
    target_compile_definitions(vsla_static PRIVATE VSLA_ENABLE_CUFFT)
    target_link_libraries(vsla_static PRIVATE CUDA::cufft)
    
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(vsla_shared PRIVATE VSLA_ENABLE_CUFFT)
        target_link_libraries(vsla_shared PRIVATE CUDA::cufft)
    endif()
endif()

if(VSLA_ENABLE_ROCFFT)
    target_compile_definitions(vsla_static PRIVATE VSLA_ENABLE_ROCFFT)
    target_link_libraries(vsla_static PRIVATE roc::rocfft)
    
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(vsla_shared PRIVATE VSLA_ENABLE_ROCFFT)
        target_link_libraries(vsla_shared PRIVATE roc::rocfft)
    endif()
endif()

if(VSLA_ENABLE_MKL)
    target_compile_definitions(vsla_static PRIVATE VSLA_ENABLE_MKL)
    if(TARGET MKL::MKL)
        target_link_libraries(vsla_static PRIVATE MKL::MKL)
    else()
        target_link_libraries(vsla_static PRIVATE ${MKL_LIBRARIES})
        target_include_directories(vsla_static PRIVATE ${MKL_INCLUDE_DIRS})
    endif()
    
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(vsla_shared PRIVATE VSLA_ENABLE_MKL)
        if(TARGET MKL::MKL)
            target_link_libraries(vsla_shared PRIVATE MKL::MKL)
        else()
            target_link_libraries(vsla_shared PRIVATE ${MKL_LIBRARIES})
            target_include_directories(vsla_shared PRIVATE ${MKL_INCLUDE_DIRS})
        endif()
    endif()
endif()

# Unified API configuration
if(VSLA_UNIFIED_API)
    target_compile_definitions(vsla_static PRIVATE VSLA_UNIFIED_API)
    
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(vsla_shared PRIVATE VSLA_UNIFIED_API)
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
#if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
#    add_subdirectory(examples)
#endif()

# Python bindings
if(VSLA_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    
    # Enable C++ for bindings
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    pybind11_add_module(_core python/src/bindings.cpp)
    target_link_libraries(_core PRIVATE vsla_static)
    target_compile_definitions(_core PRIVATE VERSION_INFO=\"${PROJECT_VERSION}\")
    
    # Include directories for bindings
    target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    # Set properties for Python module
    set_target_properties(_core PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

# Install targets
install(TARGETS vsla_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(BUILD_SHARED_LIBS)
    install(TARGETS vsla_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

install(DIRECTORY include/vsla DESTINATION include)

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()