cmake_minimum_required(VERSION 3.15)
project(test_gguf_loading)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含项目目录
include_directories(include)

# 查找依赖
find_package(Threads REQUIRED)

# 创建可执行文件
add_executable(test_gguf_full test_gguf_loading.cpp)

# 链接库
target_link_libraries(test_gguf_full cllm_core Threads::Threads)