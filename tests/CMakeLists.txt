cmake_minimum_required(VERSION 3.15)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_memory_monitor
    test_memory_monitor.cpp
)
target_link_libraries(test_memory_monitor
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_memory_monitor COMMAND test_memory_monitor)

add_executable(test_memory_manager
    test_memory_manager.cpp
)
target_link_libraries(test_memory_manager
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_memory_manager COMMAND test_memory_manager)

add_executable(test_threadpool
    test_threadpool.cpp
)
target_link_libraries(test_threadpool
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_threadpool COMMAND test_threadpool)

add_executable(test_request_queue
    test_queue.cpp
)
target_link_libraries(test_request_queue
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_request_queue COMMAND test_request_queue)

add_executable(test_batch_manager
    test_batch_manager.cpp
)
target_link_libraries(test_batch_manager
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_batch_manager COMMAND test_batch_manager)

add_executable(test_kv_cache
    test_kv_cache.cpp
)
target_link_libraries(test_kv_cache
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

add_executable(test_sampler
    test_sampler.cpp
)
target_link_libraries(test_sampler
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_scheduler
    test_scheduler.cpp
)
target_link_libraries(test_scheduler
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_tokenizer
    test_tokenizer.cpp
)
target_link_libraries(test_tokenizer
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

# CTokenizer module tests
add_executable(test_ctokenizer
    test_ctokenizer.cpp
)
target_link_libraries(test_ctokenizer
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_ctokenizer COMMAND test_ctokenizer)

# P0 Features tests for CTokenizer (LlamaTokenizer, BatchTokenizer, PerformanceMonitor)
add_executable(test_tokenizer_p0_features
    tokenizer_p0_features_test.cpp
)
target_link_libraries(test_tokenizer_p0_features
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer_p0_features COMMAND test_tokenizer_p0_features)
set_tests_properties(test_tokenizer_p0_features PROPERTIES LABELS "p0_features")

# Unicode normalization tests (P1 feature)
add_executable(test_tokenizer_unicode
    tokenizer_unicode_test.cpp
)
target_link_libraries(test_tokenizer_unicode
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer_unicode COMMAND test_tokenizer_unicode)
set_tests_properties(test_tokenizer_unicode PROPERTIES LABELS "p1_unicode")

# P1 Features tests (TokenCache, PerformanceConfig)
add_executable(test_tokenizer_p1_features
    tokenizer_p1_features_test.cpp
)
target_link_libraries(test_tokenizer_p1_features
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer_p1_features COMMAND test_tokenizer_p1_features)
set_tests_properties(test_tokenizer_p1_features PROPERTIES LABELS "p1_features")

# DeepSeek preprocessing unit tests (不需要模型文件)
add_executable(test_deepseek_preprocessing_unit
    test_deepseek_preprocessing_unit.cpp
)
target_link_libraries(test_deepseek_preprocessing_unit
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_deepseek_preprocessing_unit COMMAND test_deepseek_preprocessing_unit)

# ModelExecutor inference tests
add_executable(test_modelexecutor_inference
    test_modelexecutor_inference.cpp
)
target_link_libraries(test_modelexecutor_inference
    cllm_core
    gtest
    gtest_main
    Threads::Threads
)
add_test(NAME test_modelexecutor_inference COMMAND test_modelexecutor_inference)
set_tests_properties(test_deepseek_preprocessing_unit PROPERTIES LABELS "unit_test")

# Qwen preprocessing unit tests (不需要模型文件)
add_executable(test_qwen_preprocessing_unit
    test_qwen_preprocessing_unit.cpp
)
target_link_libraries(test_qwen_preprocessing_unit
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_qwen_preprocessing_unit COMMAND test_qwen_preprocessing_unit)
set_tests_properties(test_qwen_preprocessing_unit PROPERTIES LABELS "unit_test")

# test_http_server.cpp does not exist yet
# add_executable(test_http_server
#     test_http_server.cpp
# )
# target_link_libraries(test_http_server
#     cllm_core
#     gtest
#     gtest_main
# )
# add_test(NAME test_http_server COMMAND test_http_server)

add_executable(test_inference_engine_qwen
    test_inference_engine_qwen.cpp
)
target_link_libraries(test_inference_engine_qwen
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_inference_engine_qwen COMMAND test_inference_engine_qwen)

add_executable(test_libtorch_backend
    test_libtorch_backend.cpp
)
target_link_libraries(test_libtorch_backend
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_libtorch_backend COMMAND test_libtorch_backend)

add_executable(test_sentencepiece_integration
    test_sentencepiece_integration.cpp
)
target_link_libraries(test_sentencepiece_integration
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_sentencepiece_integration COMMAND test_sentencepiece_integration)

# Simple ModelExecutor initialization test (for debugging)
add_executable(test_modelexecutor_init
    test_modelexecutor_init.cpp
)
target_link_libraries(test_modelexecutor_init
    cllm_core
)

# Tokenizer ↔ ModelExecutor 联调集成测试
add_executable(test_tokenizer_executor_integration
    test_tokenizer_executor_integration.cpp
)
target_link_libraries(test_tokenizer_executor_integration
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer_executor_integration COMMAND test_tokenizer_executor_integration)
set_tests_properties(test_tokenizer_executor_integration PROPERTIES LABELS "integration_test;priority_high")

# HTTP Server Direct Integration Test (跳过Scheduler)
add_executable(test_http_server_direct
    test_http_server_direct.cpp
)
target_link_libraries(test_http_server_direct
    cllm_core
    gtest
    gtest_main
    /opt/homebrew/lib/libjsoncpp.dylib
)
add_test(NAME test_http_server_direct COMMAND test_http_server_direct)
set_tests_properties(test_http_server_direct PROPERTIES LABELS "integration_test;http_server;priority_high")

add_executable(test_kylin_backend
    test_kylin_backend.cpp
)
target_link_libraries(test_kylin_backend
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_kylin_backend COMMAND test_kylin_backend)

# HTTP 端点单元测试（不需要启动服务器）
add_executable(test_endpoints
    test_endpoints.cpp
)
target_link_libraries(test_endpoints
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_endpoints COMMAND test_endpoints)
set_tests_properties(test_endpoints PROPERTIES LABELS "unit_test")

# 服务器集成测试（需要启动完整服务器）
# 需要 libcurl 支持
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(test_server_integration
        test_server_integration.cpp
    )
    target_link_libraries(test_server_integration
        cllm_core
        gtest
        gtest_main
        CURL::libcurl
    )
    add_test(NAME test_server_integration COMMAND test_server_integration)
    set_tests_properties(test_server_integration PROPERTIES LABELS "integration_test")
    message(STATUS "Server integration tests enabled (CURL found)")
else()
    message(WARNING "CURL not found, skipping server integration tests")
endif()

# HTTP Server Tokenize Integration Tests
add_executable(test_http_server_tokenize
    test_http_server_tokenize.cpp
)
target_link_libraries(test_http_server_tokenize
    cllm_core
    gtest
    gtest_main
    /opt/homebrew/lib/libjsoncpp.dylib
)
add_test(NAME test_http_server_tokenize COMMAND test_http_server_tokenize)
set_tests_properties(test_http_server_tokenize PROPERTIES LABELS "integration_test;http_server")

# HTTP Server Complete Integration Tests
add_executable(test_http_server_complete
    test_http_server_complete.cpp
)
target_link_libraries(test_http_server_complete
    cllm_core
    gtest
    gtest_main
    /opt/homebrew/lib/libjsoncpp.dylib
)
add_test(NAME test_http_server_complete COMMAND test_http_server_complete)
set_tests_properties(test_http_server_complete PROPERTIES LABELS "integration_test;http_server")

# ==================== Phase 2 模块集成测试 ====================

# P2.1: HTTP + Tokenizer 集成测试
add_executable(test_http_tokenizer_integration
    test_http_tokenizer_integration.cpp
)
target_link_libraries(test_http_tokenizer_integration
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_http_tokenizer_integration COMMAND test_http_tokenizer_integration)
set_tests_properties(test_http_tokenizer_integration PROPERTIES LABELS "phase2;integration_test;priority_high")

# P2.2: Tokenizer + Executor 集成测试
add_executable(test_tokenizer_executor_integration_p2
    test_tokenizer_executor_integration.cpp
)
target_link_libraries(test_tokenizer_executor_integration_p2
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_tokenizer_executor_integration_p2 COMMAND test_tokenizer_executor_integration_p2)
set_tests_properties(test_tokenizer_executor_integration_p2 PROPERTIES LABELS "phase2;integration_test;priority_high")

# P2.3: Executor + Backend 集成测试
add_executable(test_executor_backend_integration
    test_executor_backend_integration.cpp
)
target_link_libraries(test_executor_backend_integration
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_executor_backend_integration COMMAND test_executor_backend_integration)
set_tests_properties(test_executor_backend_integration PROPERTIES LABELS "phase2;integration_test;priority_high")

# P2.4: Backend + Qwen3 集成测试
add_executable(test_backend_qwen3_integration
    test_backend_qwen3_integration.cpp
)
target_link_libraries(test_backend_qwen3_integration
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_backend_qwen3_integration COMMAND test_backend_qwen3_integration)
set_tests_properties(test_backend_qwen3_integration PROPERTIES LABELS "phase2;integration_test;priority_high")

# ==================== Phase 3 子系统测试 ====================

# P3.1: 前端子系统测试
add_executable(test_frontend_subsystem
    test_frontend_subsystem.cpp
)
target_link_libraries(test_frontend_subsystem
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_frontend_subsystem COMMAND test_frontend_subsystem)
set_tests_properties(test_frontend_subsystem PROPERTIES LABELS "phase3;integration_test;priority_high")

# P3.2: 推理子系统测试
add_executable(test_inference_subsystem
    test_inference_subsystem.cpp
)
target_link_libraries(test_inference_subsystem
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_inference_subsystem COMMAND test_inference_subsystem)
set_tests_properties(test_inference_subsystem PROPERTIES LABELS "phase3;integration_test;priority_high")

# P3.3: E2E子系统测试
add_executable(test_e2e_subsystem
    test_e2e_subsystem.cpp
)
target_link_libraries(test_e2e_subsystem
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_e2e_subsystem COMMAND test_e2e_subsystem)
set_tests_properties(test_e2e_subsystem PROPERTIES LABELS "phase3;integration_test;priority_high")

# ==================== Phase 4 系统功能测试 ====================

# P4.1: 系统功能测试
add_executable(test_system_functionality
    test_system_functionality.cpp
)
target_link_libraries(test_system_functionality
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_system_functionality COMMAND test_system_functionality)
set_tests_properties(test_system_functionality PROPERTIES LABELS "phase4;system_test;priority_high")

# P4.2: 性能基准测试
add_executable(test_performance_benchmark
    test_performance_benchmark.cpp
)
target_link_libraries(test_performance_benchmark
    cllm_core
    gtest
    gtest_main
)
add_test(NAME test_performance_benchmark COMMAND test_performance_benchmark)
set_tests_properties(test_performance_benchmark PROPERTIES LABELS "phase4;performance_test;priority_high")

# P4.3: 压力和稳定性测试
# 需要 libcurl 和 jsoncpp 支持
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(test_stress_stability
        test_stress_stability.cpp
    )
    target_link_libraries(test_stress_stability
        cllm_core
        gtest
        gtest_main
        CURL::libcurl
        /opt/homebrew/lib/libjsoncpp.dylib
    )
    add_test(NAME test_stress_stability COMMAND test_stress_stability)
    set_tests_properties(test_stress_stability PROPERTIES LABELS "phase4;stress_test;stability_test;priority_high")
    message(STATUS "Stress and stability tests enabled (CURL found)")
else()
    message(WARNING "CURL not found, skipping stress and stability tests")
endif()

# ==================== Phase 5 E2E 场景测试 ====================

# P5: E2E 场景测试
# 需要 libcurl 和 jsoncpp 支持
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(test_e2e_scenarios
        test_e2e_scenarios.cpp
    )
    target_link_libraries(test_e2e_scenarios
        cllm_core
        gtest
        gtest_main
        CURL::libcurl
        /opt/homebrew/lib/libjsoncpp.dylib
    )
    add_test(NAME test_e2e_scenarios COMMAND test_e2e_scenarios)
    set_tests_properties(test_e2e_scenarios PROPERTIES LABELS "phase5;e2e_test;scenario_test;priority_high")
    message(STATUS "E2E scenarios tests enabled (CURL found)")
else()
    message(WARNING "CURL not found, skipping E2E scenarios tests")
endif()