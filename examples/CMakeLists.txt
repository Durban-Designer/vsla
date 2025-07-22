cmake_minimum_required(VERSION 3.10)

# Add an executable for each example
add_executable(basic_usage basic_usage.c)
add_executable(gpu_demo gpu_demo.c)
add_executable(neural_network_cnn neural_network_cnn.c)
add_executable(polynomial_algebra polynomial_algebra.c)
add_executable(signal_processing_radar signal_processing_radar.c)
add_executable(tensor_stacking tensor_stacking.c)

# Link each example against the VSLA library target and the math library
target_link_libraries(basic_usage PRIVATE vsla_static m)
target_link_libraries(gpu_demo PRIVATE vsla_static m)
target_link_libraries(neural_network_cnn PRIVATE vsla_static m)
target_link_libraries(polynomial_algebra PRIVATE vsla_static m)
target_link_libraries(signal_processing_radar PRIVATE vsla_static m)
target_link_libraries(tensor_stacking PRIVATE vsla_static m)
