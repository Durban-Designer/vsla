# CMake generated Testfile for 
# Source directory: /home/kenth56/vsla/tests
# Build directory: /home/kenth56/vsla/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(core_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=core")
set_tests_properties(core_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;37;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(tensor_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=tensor")
set_tests_properties(tensor_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;38;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(ops_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=ops")
set_tests_properties(ops_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;39;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(io_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=io")
set_tests_properties(io_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;40;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(conv_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=conv")
set_tests_properties(conv_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;41;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(kron_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=kron")
set_tests_properties(kron_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;42;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(autograd_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=autograd")
set_tests_properties(autograd_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;43;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(utils_tests "/home/kenth56/vsla/build/tests/vsla_tests" "--suite=utils")
set_tests_properties(utils_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;44;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(all_tests "/home/kenth56/vsla/build/tests/vsla_tests")
set_tests_properties(all_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;45;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
add_test(memory_tests "/usr/bin/valgrind" "--leak-check=full" "--show-leak-kinds=all" "--track-origins=yes" "--error-exitcode=1" "/home/kenth56/vsla/build/tests/vsla_tests")
set_tests_properties(memory_tests PROPERTIES  TIMEOUT "300" _BACKTRACE_TRIPLES "/home/kenth56/vsla/tests/CMakeLists.txt;50;add_test;/home/kenth56/vsla/tests/CMakeLists.txt;0;")
