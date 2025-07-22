# Modern unified backend test
add_executable(unified_backend_tests test_unified_backend.c)
target_link_libraries(unified_backend_tests PRIVATE vsla_static m)
add_test(NAME UnifiedBackendTests COMMAND unified_backend_tests)

# Original backend tests (disabled due to API compatibility issues)
# add_executable(backend_tests test_backend.c test_framework.c)
# target_link_libraries(backend_tests PRIVATE vsla_static m)

# Unified interface tests
add_executable(unified_tests
    test_main_unified.c
    test_unified_framework.c
    test_arithmetic_unified.c
)
target_link_libraries(unified_tests PRIVATE vsla_static m)

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

add_test(NAME UnifiedTests_CUDA COMMAND unified_tests --backend cuda)
add_test(NAME UnifiedTests_CUDA_Verbose COMMAND unified_tests --backend cuda --verbose)

# VSLA specification validation tests
add_executable(vsla_spec_tests test_vsla_spec_validation.c)
target_link_libraries(vsla_spec_tests PRIVATE vsla_static m)
add_test(NAME VSLASpecTests COMMAND vsla_spec_tests)

# VSLA stacking operations tests
add_executable(vsla_stacking_tests test_stacking_comprehensive.c)
target_link_libraries(vsla_stacking_tests PRIVATE vsla_static m)
add_test(NAME VSLAStackingTests COMMAND vsla_stacking_tests)
