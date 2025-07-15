# CMake generated Testfile for 
# Source directory: /home/kenth56/Documents/vsla/libvsla/tests
# Build directory: /home/kenth56/Documents/vsla/libvsla/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(core_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=core")
set_tests_properties(core_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;36;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(tensor_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=tensor")
set_tests_properties(tensor_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;37;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(ops_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=ops")
set_tests_properties(ops_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;38;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(io_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=io")
set_tests_properties(io_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;39;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(conv_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=conv")
set_tests_properties(conv_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;40;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(kron_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=kron")
set_tests_properties(kron_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;41;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(autograd_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests" "--suite=autograd")
set_tests_properties(autograd_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;42;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
add_test(all_tests "/home/kenth56/Documents/vsla/libvsla/build/tests/vsla_tests")
set_tests_properties(all_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;43;add_test;/home/kenth56/Documents/vsla/libvsla/tests/CMakeLists.txt;0;")
