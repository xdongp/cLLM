# CMakeLists.txt for tests

cmake_minimum_required(VERSION 3.14)
project(cllm_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # 包含utils目录

# 查找或添加 googletest
if(NOT TARGET gtest)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
endif()

# ========== 测试工具库验证 ==========
add_executable(minimal_test
    unit/minimal_test.cpp
)

target_include_directories(minimal_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(minimal_test PRIVATE
    gtest
    gtest_main
    pthread
)

set_target_properties(minimal_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# ========== 单元测试 ==========

# 超时检测测试
add_executable(unit_timeout_detection_test
    unit/common/timeout_detection_test.cpp
)

target_include_directories(unit_timeout_detection_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(unit_timeout_detection_test PRIVATE
    cllm_core
    gtest
    gtest_main
    pthread
    spdlog::spdlog
)

set_target_properties(unit_timeout_detection_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# KV缓存管理器测试
add_executable(unit_kv_cache_manager_test
    unit/cache/kv_cache_manager_test.cpp
)

target_include_directories(unit_kv_cache_manager_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(unit_kv_cache_manager_test PRIVATE
    cllm_core
    gtest
    gtest_main
    pthread
    spdlog::spdlog
)

set_target_properties(unit_kv_cache_manager_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# ========== Kylin Backend Test Suite ==========
# 独立的测试框架，不依赖 GTest
# 注意：test_*.cpp 被 kylin_test_main.cpp 包含，不需要单独编译
add_executable(kylin_test_suite
    kylin_test_suite/kylin_test_main.cpp
)

target_include_directories(kylin_test_suite PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json/include
)

target_link_libraries(kylin_test_suite PRIVATE
    cllm_core
)

set_target_properties(kylin_test_suite PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========== Layer Debug Test ==========
add_executable(test_layer_debug
    test_layer_debug.cpp
)

target_include_directories(test_layer_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json/include
)

target_link_libraries(test_layer_debug PRIVATE
    cllm_core
    spdlog::spdlog
)

set_target_properties(test_layer_debug PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加测试到 CTest
enable_testing()
add_test(NAME minimal_test COMMAND minimal_test)
add_test(NAME unit_timeout_detection_test COMMAND unit_timeout_detection_test)
add_test(NAME unit_kv_cache_manager_test COMMAND unit_kv_cache_manager_test)
add_test(NAME kylin_test_suite COMMAND kylin_test_suite --all)


