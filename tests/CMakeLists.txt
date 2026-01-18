# CMakeLists.txt for tests

cmake_minimum_required(VERSION 3.14)
project(cllm_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include)

# 注意：旧的测试文件已被删除，因为它们缺少必要的依赖
# 新的 llama.cpp 后端测试在主 CMakeLists.txt 中定义


