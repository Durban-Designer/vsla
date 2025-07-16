cmake_minimum_required(VERSION 3.14)
project(libvsla VERSION 1.0.0 LANGUAGES C)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_FFTW "Use FFTW library for FFT operations" OFF)

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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(VSLA_SOURCES
    src/vsla_core.c
    src/vsla_tensor.c
    src/vsla_ops.c
    src/vsla_io.c
    src/vsla_conv.c
    src/vsla_kron.c
    src/vsla_autograd.c
    src/vsla_utils.c
)

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

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
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