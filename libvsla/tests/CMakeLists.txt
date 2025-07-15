# Test configuration for VSLA library

find_package(Threads REQUIRED)

# Common test settings
set(TEST_INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Test framework - using a simple doctest-style approach
set(TEST_SOURCES
    test_framework.c
    test_main.c
    test_core.c
    test_tensor.c
    test_ops.c
    test_io.c
    test_conv.c
    test_kron.c
    test_autograd.c
)

# Create test executable
add_executable(vsla_tests ${TEST_SOURCES})

target_include_directories(vsla_tests PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(vsla_tests PRIVATE vsla_static Threads::Threads m)

# Compiler flags for tests
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vsla_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add tests to CTest
add_test(NAME core_tests COMMAND vsla_tests --suite=core)
add_test(NAME tensor_tests COMMAND vsla_tests --suite=tensor)
add_test(NAME ops_tests COMMAND vsla_tests --suite=ops)
add_test(NAME io_tests COMMAND vsla_tests --suite=io)
add_test(NAME conv_tests COMMAND vsla_tests --suite=conv)
add_test(NAME kron_tests COMMAND vsla_tests --suite=kron)
add_test(NAME autograd_tests COMMAND vsla_tests --suite=autograd)
add_test(NAME all_tests COMMAND vsla_tests)

# Memory leak testing with valgrind (if available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_test(NAME memory_tests 
             COMMAND ${VALGRIND_EXECUTABLE} 
                     --leak-check=full 
                     --show-leak-kinds=all 
                     --track-origins=yes
                     --error-exitcode=1
                     $<TARGET_FILE:vsla_tests>)
    set_tests_properties(memory_tests PROPERTIES TIMEOUT 300)
endif()

# Coverage target (if supported)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    option(ENABLE_COVERAGE "Enable test coverage" OFF)
    if(ENABLE_COVERAGE)
        target_compile_options(vsla_tests PRIVATE --coverage)
        target_link_options(vsla_tests PRIVATE --coverage)
        
        find_program(GCOV_EXECUTABLE gcov)
        find_program(LCOV_EXECUTABLE lcov)
        find_program(GENHTML_EXECUTABLE genhtml)
        
        if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
            add_custom_target(coverage
                COMMAND ${LCOV_EXECUTABLE} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --list coverage.info
                COMMAND ${GENHTML_EXECUTABLE} -o coverage_html coverage.info
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating test coverage report"
            )
        endif()
    endif()
endif()