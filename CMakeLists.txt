cmake_minimum_required(VERSION 3.15)
project(cllm VERSION 1.0.0 LANGUAGES C CXX)

# Find Drogon (temporarily commented out for testing)
# find_package(Drogon REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

find_package(Threads REQUIRED)

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 REQUIRED)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
endif()

# Find Asio
find_package(asio CONFIG QUIET)
if(NOT asio_FOUND)
    message(WARNING "Asio not found. You may need to install it manually.")
    # For now, we'll skip Asio integration in this build
else()
    target_link_libraries(cllm_core PRIVATE asio::asio)
endif()

# Find YAML-CPP
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp: ${yaml-cpp_LIBRARIES}")
endif()

# Find spdlog
find_package(spdlog REQUIRED)

# Find LibTorch (PyTorch C++ API)
set(TORCH_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Found LibTorch: ${TORCH_LIBRARIES}")

# Find llama.cpp
set(LLAMA_CPP_BUILD_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build")
if(EXISTS "${LLAMA_CPP_BUILD_DIR}/llama-config.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${LLAMA_CPP_BUILD_DIR}")
    find_package(Llama QUIET HINTS "${LLAMA_CPP_BUILD_DIR}")
    if(Llama_FOUND)
        message(STATUS "✅ Found llama.cpp: ${llama_LIBRARY}")
        set(LLAMA_CPP_FOUND TRUE)
    else()
        message(WARNING "⚠️  llama.cpp not found, LlamaCppBackend will not be available")
        message(STATUS "   Build llama.cpp first: cd third_party/llama.cpp && mkdir -p build && cd build && cmake .. && make -j$(sysctl -n hw.ncpu)")
        set(LLAMA_CPP_FOUND FALSE)
    endif()
else()
    message(WARNING "⚠️  llama.cpp build directory not found, LlamaCppBackend will not be available")
    message(STATUS "   Build llama.cpp first: cd third_party/llama.cpp && mkdir -p build && cd build && cmake .. && make -j$(sysctl -n hw.ncpu)")
    set(LLAMA_CPP_FOUND FALSE)
endif()

# Tokenizers 选项 (✅ 默认启用)
option(USE_TOKENIZERS_CPP "Use tokenizers-cpp for HuggingFace tokenizer" ON)

if(USE_TOKENIZERS_CPP)
    message(STATUS "✅ Enabling HuggingFace tokenizers support (tokenizers-cpp)")
    
    # 查找tokenizers-cpp头文件
    find_path(TOKENIZERS_INCLUDE_DIR 
        NAMES tokenizers_cpp.h tokenizers_c.h
        PATHS 
            /opt/homebrew/include
            /usr/local/include
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/include
    )
    
    # 查找tokenizers-cpp库文件
    find_library(TOKENIZERS_LIBRARY 
        NAMES tokenizers_cpp
        PATHS 
            /opt/homebrew/lib
            /usr/local/lib
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/lib
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/build
    )
    
    # 查找tokenizers_c库 (Rust部分)
    find_library(TOKENIZERS_C_LIBRARY 
        NAMES tokenizers_c
        PATHS 
            /opt/homebrew/lib
            /usr/local/lib
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/lib
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/build
    )

    # 查找sentencepiece库 (tokenizers-cpp依赖)
    find_library(TOKENIZERS_SP_LIBRARY 
        NAMES sentencepiece
        PATHS 
            ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp/build/sentencepiece/src
    )
    
    if(TOKENIZERS_INCLUDE_DIR AND TOKENIZERS_LIBRARY AND TOKENIZERS_C_LIBRARY)
        message(STATUS "✅ Found tokenizers-cpp:")
        message(STATUS "   Include: ${TOKENIZERS_INCLUDE_DIR}")
        message(STATUS "   Library (CPP): ${TOKENIZERS_LIBRARY}")
        message(STATUS "   Library (C): ${TOKENIZERS_C_LIBRARY}")
        
        add_compile_definitions(USE_TOKENIZERS_CPP)
        include_directories(${TOKENIZERS_INCLUDE_DIR})
        
        set(TOKENIZERS_LIBRARIES ${TOKENIZERS_LIBRARY} ${TOKENIZERS_C_LIBRARY})
        if(TOKENIZERS_SP_LIBRARY)
            list(APPEND TOKENIZERS_LIBRARIES ${TOKENIZERS_SP_LIBRARY})
        endif()
    else()
        message(WARNING "⚠️  tokenizers-cpp not found, falling back to NativeTokenizer")
        message(WARNING "   Install guide: https://github.com/mlc-ai/tokenizers-cpp")
        message(STATUS "   Quick install on macOS:")
        message(STATUS "     brew install rust")
        message(STATUS "     git clone https://github.com/mlc-ai/tokenizers-cpp")
        message(STATUS "     cd tokenizers-cpp && mkdir build && cd build")
        message(STATUS "     cmake .. -DCMAKE_INSTALL_PREFIX=/opt/homebrew")
        message(STATUS "     make -j8 && make install")
        set(USE_TOKENIZERS_CPP OFF)
    endif()
else()
    message(STATUS "❌ HuggingFace tokenizers support disabled")
endif()

# Find SentencePiece - try multiple methods
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SentencePiece QUIET sentencepiece)
    if(SentencePiece_FOUND)
        message(STATUS "Found SentencePiece via pkg-config: ${SentencePiece_LIBRARIES}")
        include_directories(${SentencePiece_INCLUDE_DIRS})
    endif()
endif()

# If pkg-config failed, try manual find
if(NOT SentencePiece_FOUND)
    find_path(SentencePiece_INCLUDE_DIRS
        NAMES sentencepiece_processor.h
        PATHS /usr/include /usr/local/include /opt/homebrew/include
        PATH_SUFFIXES sentencepiece
    )
    
    find_library(SentencePiece_LIBRARIES
        NAMES sentencepiece
        PATHS /usr/lib /usr/local/lib /opt/homebrew/lib
    )
    
    if(SentencePiece_INCLUDE_DIRS AND SentencePiece_LIBRARIES)
        message(STATUS "Found SentencePiece manually: ${SentencePiece_LIBRARIES}")
        include_directories(${SentencePiece_INCLUDE_DIRS})
        set(SentencePiece_FOUND TRUE)
    else()
        message(FATAL_ERROR "SentencePiece not found. Please install it and set SentencePiece_ROOT")
    endif()
endif()

find_package(Drogon REQUIRED)

add_subdirectory(third_party)
add_subdirectory(src/CTokenizer)

# CTokenizer sources are now available via CTOKENIZER_SOURCES variable

add_library(cllm_core
    src/common/config.cpp
    src/common/types.cpp
    src/common/utils.cpp
    src/common/json.cpp
    src/common/logger.cpp
    src/common/asio_handler.cpp
    src/common/memory_utils.cpp
    src/common/time_utils.cpp
    src/memory/monitor.cpp
    src/memory/float_array.cpp
    src/memory/cache_manager.cpp
    src/memory/executor_manager.cpp
    src/thread_pool/manager.cpp
    src/thread_pool/monitor.cpp
    src/common/queue.cpp
    src/common/request_state.cpp
    src/batch/manager.cpp
    src/batch/processor.cpp
    src/kv_cache/cache.cpp
    src/kv_cache/cache_manager.cpp
    src/sampler/sampler.cpp
    src/sampler/config.cpp
    src/sampler/stats.cpp
    src/model/config.cpp
    src/model/executor.cpp
    src/model/stats.cpp
    src/model/inference_optimizer.cpp
    src/model/quantization_manager.cpp
    src/model/batch_processor.cpp
    src/model/loader_interface.cpp
    src/model/weight_data.cpp
    src/model/gguf_loader_new.cpp
    src/model/gguf_dequantization.cpp
    # Kylin 模块 - 核心
    src/kylin/core/kernels.cpp
    src/kylin/core/quantization.cpp
    src/kylin/core/tensor_stats.cpp
    # Kylin 模块 - 算子
    src/kylin/ops/rope.cpp
    src/kylin/ops/attention.cpp
    src/kylin/ops/attention_graph.cpp
    src/kylin/ops/feed_forward.cpp
    src/kylin/ops/kv_cache_ops.cpp
    # Kylin 模块 - HuggingFace 后端 (主要)
    src/kylin/hf/config.cpp
    src/kylin/hf/safetensors_loader.cpp
    src/kylin/hf/transformer.cpp
    # Kylin 模块 - GGUF 后端 (保留)
    src/kylin/gguf/loader.cpp
    src/kylin/gguf/context.cpp
    src/kylin/gguf/transformer.cpp
    src/kylin/gguf/operator_interface.cpp
    src/kylin/gguf/native_operator.cpp
    src/kylin/gguf/ggml_operator.cpp
    # Kylin 模块 - 模型抽象
    src/kylin/model/transformer_block.cpp
    src/kylin/model/transformer_model.cpp
    src/kylin/model/model_loader.cpp
    src/inference/inference_engine.cpp
    src/inference/backend_factory.cpp
    src/inference/kylin_backend.cpp
    src/inference/libtorch_backend.cpp
    src/inference/llama_cpp_backend.cpp
    src/inference/kv_cache_manager.cpp
    src/scheduler/scheduler.cpp
    src/scheduler/stats.cpp
    src/scheduler/tracker.cpp
    src/scheduler/batch_processor.cpp
    src/tokenizer/token.cpp
    src/tokenizer/config.cpp
    src/tokenizer/request.cpp
    src/tokenizer/response.cpp
    src/tokenizer/stats.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/generator.cpp
    src/tokenizer/manager.cpp
    src/tokenizer/json_tokenizer.cpp
    src/tokenizer/qwen2_tokenizer.cpp
    src/tokenizer/hf_tokenizer.cpp
    src/tokenizer/native_tokenizer.cpp
    src/tokenizer/unicode_utils.cpp
    src/tokenizer/unified_tokenizer.cpp
    src/tokenizer/gguf_tokenizer.cpp
    src/http/request.cpp
    src/http/response.cpp
    src/http/request_validator.cpp
    src/http/response_builder.cpp
    src/http/handler.cpp
    src/http/api_endpoint.cpp
    src/http/health_endpoint.cpp
    src/http/generate_endpoint.cpp
    src/http/encode_endpoint.cpp
    src/http/benchmark_endpoint.cpp
    src/http/http_server.cpp
    src/http/json_request_parser.cpp
    src/http/request_queue.cpp
    ${CTOKENIZER_SOURCES}
)

target_link_libraries(cllm_core PRIVATE
    Threads::Threads
    bs_thread_pool
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    "${TORCH_LIBRARIES}"
    ${SentencePiece_LIBRARIES}
    ${TOKENIZERS_LIBRARIES}  # ✅ Add tokenizers-cpp
    Drogon::Drogon
)

# Link llama.cpp if available
if(LLAMA_CPP_FOUND)
    target_sources(cllm_core PRIVATE src/inference/llama_cpp_backend.cpp)
    # 添加 llama.cpp include 目录
    include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
    target_link_libraries(cllm_core PRIVATE llama)
    target_compile_definitions(cllm_core PRIVATE CLLM_USE_LLAMA_CPP)
    message(STATUS "✅ Linked llama.cpp library")
else()
    message(STATUS "ℹ️  llama.cpp not available, LlamaCppBackend disabled")
endif()

# Set ASIO_STANDALONE flag to use header-only version
if(NOT asio_FOUND)
    target_compile_definitions(cllm_core PUBLIC ASIO_STANDALONE)
endif()

if(BUILD_TESTS)
    add_subdirectory(third_party/googletest)
    enable_testing()
    add_subdirectory(tests)
    
    # HFTokenizer 专项测试 - 暂时注释掉，文件不存在
    # add_executable(test_hf_tokenizer
    #     tests/test_hf_tokenizer.cpp
    # )
    # target_link_libraries(test_hf_tokenizer
    #     cllm_core
    #     gtest
    #     gtest_main
    #     Threads::Threads
    # )
    # add_test(NAME HFTokenizerTest COMMAND test_hf_tokenizer)
    
    # GGUF Loader 完整测试 - 暂时注释掉，因为需要更新测试以适配新的GGUFLoader接口
    # add_executable(test_gguf_loader_complete
    #     tests/test_gguf_loader_complete.cpp
    # )
    # target_link_libraries(test_gguf_loader_complete
    #     cllm_core
    #     gtest
    #     gtest_main
    #     Threads::Threads
    # )
    # add_test(NAME GGUFLoaderCompleteTest COMMAND test_gguf_loader_complete)
    
    # GGUF Loader 手动测试程序 - 暂时注释掉，文件不存在
    # add_executable(test_gguf_full
    #     tests/test_gguf_loading.cpp
    # )
    # target_link_libraries(test_gguf_full
    #     cllm_core
    #     Threads::Threads
    # )
    
    # Kylin引擎GGUF Q4_K_M推理测试
    add_executable(test_kylin_gguf_q4k
        tests/test_kylin_gguf_q4k.cpp
    )
    target_link_libraries(test_kylin_gguf_q4k
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME KylinGGUFQ4KTest COMMAND test_kylin_gguf_q4k)
    
    # 简单的 hello 推理测试 - 暂时注释掉，文件不存在
    # add_executable(test_hello_inference
    #     tests/test_hello_inference.cpp
    # )
    # target_link_libraries(test_hello_inference
    #     cllm_core
    #     Threads::Threads
    # )
    
    # llama.cpp 后端 GGUF 测试
    add_executable(test_llama_cpp_backend_gguf
        tests/test_llama_cpp_backend_gguf.cpp
    )
    target_link_libraries(test_llama_cpp_backend_gguf
        cllm_core
        Threads::Threads
        spdlog::spdlog
    )
    
    # 条件链接 llama.cpp（如果可用）
    if(LLAMA_CPP_FOUND)
        target_compile_definitions(test_llama_cpp_backend_gguf PRIVATE CLLM_USE_LLAMA_CPP)
    else()
        message(WARNING "⚠️  test_llama_cpp_backend_gguf requires llama.cpp, but it's not available")
    endif()
    
    # Phase 2: 序列ID管理单元测试
    add_executable(test_sequence_id_manager
        tests/test_sequence_id_manager.cpp
    )
    target_link_libraries(test_sequence_id_manager
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    if(LLAMA_CPP_FOUND)
        target_compile_definitions(test_sequence_id_manager PRIVATE CLLM_USE_LLAMA_CPP)
        add_test(NAME SequenceIdManagerTest COMMAND test_sequence_id_manager)
    else()
        message(WARNING "⚠️  test_sequence_id_manager requires llama.cpp, but it's not available")
    endif()
    
    # Phase 2: 序列ID管理单元测试（简化版，不依赖后端初始化）
    add_executable(test_sequence_id_manager_simple
        tests/test_sequence_id_manager_simple.cpp
    )
    target_link_libraries(test_sequence_id_manager_simple
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME SequenceIdManagerSimpleTest COMMAND test_sequence_id_manager_simple)
    
    # Phase 1: 状态转换专项测试
    add_executable(test_state_transition
        tests/test_state_transition.cpp
    )
    target_link_libraries(test_state_transition
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME StateTransitionTest COMMAND test_state_transition)
    
    # Phase 1: 状态转换专项测试（简化版，不依赖 Scheduler 初始化）
    add_executable(test_state_transition_simple
        tests/test_state_transition_simple.cpp
    )
    target_link_libraries(test_state_transition_simple
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME StateTransitionSimpleTest COMMAND test_state_transition_simple)
    
    # Phase 3: 超时检测机制
    add_executable(test_timeout_detection
        tests/test_timeout_detection.cpp
    )
    target_link_libraries(test_timeout_detection
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME TimeoutDetectionTest COMMAND test_timeout_detection)
    
    # Phase 4: KV缓存统计管理
    add_executable(test_kv_cache_manager
        tests/test_kv_cache_manager.cpp
    )
    target_link_libraries(test_kv_cache_manager
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME KVCacheManagerTest COMMAND test_kv_cache_manager)
    
    # Phase 5: KV缓存LRU淘汰
    add_executable(test_kv_cache_lru_eviction
        tests/test_kv_cache_lru_eviction.cpp
    )
    target_link_libraries(test_kv_cache_lru_eviction
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME KVCacheLRUEvictionTest COMMAND test_kv_cache_lru_eviction)
    
    # Phase 6: HTTP层并发检查
    add_executable(test_http_concurrency
        tests/test_http_concurrency.cpp
    )
    target_link_libraries(test_http_concurrency
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
        nlohmann_json::nlohmann_json
    )
    add_test(NAME HttpConcurrencyTest COMMAND test_http_concurrency)
    
    # Phase 7: 响应回调机制
    add_executable(test_response_callback
        tests/test_response_callback.cpp
    )
    target_link_libraries(test_response_callback
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME ResponseCallbackTest COMMAND test_response_callback)
    
    # Phase 7: 响应回调机制（简化测试）
    add_executable(test_response_callback_simple
        tests/test_response_callback_simple.cpp
    )
    target_link_libraries(test_response_callback_simple
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
    )
    add_test(NAME ResponseCallbackSimpleTest COMMAND test_response_callback_simple)
    
    # Phase 7: 集成测试 - /generate接口（简化版）
    add_executable(test_generate_simple
        tests/test_generate_simple.cpp
    )
    target_link_libraries(test_generate_simple
        cllm_core
        gtest
        gtest_main
        Threads::Threads
        spdlog::spdlog
        nlohmann_json::nlohmann_json
    )
    add_test(NAME GenerateSimpleTest COMMAND test_generate_simple)
    
    # 加载Qwen模型的测试程序
    add_executable(load_qwen_model
        examples/load_qwen_model.cpp
    )
    target_link_libraries(load_qwen_model
        cllm_core
        Threads::Threads
    )
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    
    # HFTokenizer 示例
    add_executable(hf_tokenizer_example
        examples/hf_tokenizer_example.cpp
    )
    target_link_libraries(hf_tokenizer_example
        cllm_core
        Threads::Threads
    )
endif()

# Add executable for the main program
add_executable(cllm_server src/main.cpp)

# Add test server executable
target_link_libraries(cllm_server
    cllm_core
    Threads::Threads
    Drogon::Drogon
)

# 添加直接性能测试程序（参考 llama-bench）
add_executable(direct_benchmark ${CMAKE_SOURCE_DIR}/tools/direct_benchmark.cpp)
target_link_libraries(direct_benchmark
    cllm_core
    Threads::Threads
    spdlog::spdlog
)
if(LLAMA_CPP_FOUND)
    target_compile_definitions(direct_benchmark PRIVATE CLLM_USE_LLAMA_CPP)
endif()

# 添加渐进式性能测试程序（逐步验证各阶段性能衰减）
add_executable(incremental_benchmark ${CMAKE_SOURCE_DIR}/tools/incremental_benchmark.cpp)
target_link_libraries(incremental_benchmark
    cllm_core
    Threads::Threads
    spdlog::spdlog
)
if(LLAMA_CPP_FOUND)
    target_compile_definitions(incremental_benchmark PRIVATE CLLM_USE_LLAMA_CPP)
endif()

# 添加 Kylin 分阶段测试程序
add_executable(kylin_stage_test ${CMAKE_SOURCE_DIR}/tools/kylin_stage_test.cpp)
target_link_libraries(kylin_stage_test
    cllm_core
    Threads::Threads
    spdlog::spdlog
)
if(LLAMA_CPP_FOUND)
    target_compile_definitions(kylin_stage_test PRIVATE CLLM_USE_LLAMA_CPP)
endif()

# 添加批量vs增量调试程序
add_executable(debug_batch_incr ${CMAKE_SOURCE_DIR}/tools/debug_batch_incr.cpp)
target_link_libraries(debug_batch_incr
    cllm_core
    Threads::Threads
    spdlog::spdlog
)

# 添加 tokenizer 调试程序
add_executable(debug_tokenizer ${CMAKE_SOURCE_DIR}/tools/debug_tokenizer.cpp)
target_link_libraries(debug_tokenizer
    cllm_core
    Threads::Threads
    spdlog::spdlog
)

# 添加 embedding 调试程序
add_executable(debug_embedding ${CMAKE_SOURCE_DIR}/tools/debug_embedding.cpp)
target_link_libraries(debug_embedding
    cllm_core
    Threads::Threads
    spdlog::spdlog
)

# 添加 llama.cpp 直接调试程序
add_executable(debug_llamacpp ${CMAKE_SOURCE_DIR}/tools/debug_llamacpp.cpp)
target_include_directories(debug_llamacpp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
target_link_directories(debug_llamacpp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin)
target_link_libraries(debug_llamacpp
    llama
    ggml
    ggml-cpu
    Threads::Threads
)

# 添加 Kylin 逐层调试工具
add_executable(debug_kylin_layers ${CMAKE_SOURCE_DIR}/tools/debug_kylin_layers.cpp)
target_include_directories(debug_kylin_layers PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
target_link_directories(debug_kylin_layers PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin)
target_link_libraries(debug_kylin_layers
    cllm_core
    llama
    ggml
    ggml-cpu
    Threads::Threads
    spdlog::spdlog
)

# 添加 Kylin vs llama.cpp 比较工具
add_executable(debug_kylin_vs_llamacpp ${CMAKE_SOURCE_DIR}/tools/debug_kylin_vs_llamacpp.cpp)
target_include_directories(debug_kylin_vs_llamacpp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
target_link_directories(debug_kylin_vs_llamacpp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin)
target_link_libraries(debug_kylin_vs_llamacpp
    cllm_core
    llama
    ggml
    ggml-cpu
    Threads::Threads
    spdlog::spdlog
)

# 添加 embedding lookup 调试工具
add_executable(debug_embedding_lookup ${CMAKE_SOURCE_DIR}/tools/debug_embedding_lookup.cpp)
target_link_libraries(debug_embedding_lookup
    cllm_core
    Threads::Threads
    spdlog::spdlog
)

# 添加 embedding 比较工具
add_executable(compare_embedding ${CMAKE_SOURCE_DIR}/tools/compare_embedding.cpp)
target_include_directories(compare_embedding PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
target_link_directories(compare_embedding PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin)
target_link_libraries(compare_embedding
    cllm_core
    llama
    ggml
    ggml-cpu
    Threads::Threads
    spdlog::spdlog
)

# 添加 HuggingFace Transformer 测试工具
add_executable(test_hf_transformer ${CMAKE_SOURCE_DIR}/tools/test_hf_transformer.cpp)
target_link_libraries(test_hf_transformer
    cllm_core
    Threads::Threads
    spdlog::spdlog
)

