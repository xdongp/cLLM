# CMakeLists.txt for tests

cmake_minimum_required(VERSION 3.14)
project(cllm_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include)

# 添加测试可执行文件
add_executable(test_gguf_loader test_gguf_loader.cpp)

# 链接库
target_link_libraries(test_gguf_loader 
    GTest::gtest 
    GTest::gtest_main
    cllm_core
)

# 启用测试
enable_testing()
add_test(NAME test_gguf_loader COMMAND test_gguf_loader)


