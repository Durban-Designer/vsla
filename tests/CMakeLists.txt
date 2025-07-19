# Original backend tests
add_executable(backend_tests test_backend.c test_framework.c)
target_link_libraries(backend_tests PRIVATE vsla_static)
add_test(NAME BackendTests COMMAND backend_tests)

# Unified interface tests
add_executable(unified_tests 
    test_main_unified.c 
    test_unified_framework.c 
    test_arithmetic_unified.c
)
target_link_libraries(unified_tests PRIVATE vsla_static)

# Add math library for test framework
if(UNIX)
    target_link_libraries(unified_tests PRIVATE m)
endif()

# Register unified tests with CTest
add_test(NAME UnifiedTests_CPU COMMAND unified_tests --backend cpu)
add_test(NAME UnifiedTests_Auto COMMAND unified_tests --backend auto)

# Verbose test variants for debugging
add_test(NAME UnifiedTests_CPU_Verbose COMMAND unified_tests --backend cpu --verbose)
add_test(NAME UnifiedTests_Auto_Verbose COMMAND unified_tests --backend auto --verbose)
