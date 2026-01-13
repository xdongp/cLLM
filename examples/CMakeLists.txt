# Examples CMakeLists.txt

# Basic usage example
add_executable(basic_usage
    basic_usage.cpp
)
target_link_libraries(basic_usage
    cllm_core
)

# HTTP server example
add_executable(http_server_example
    http_server_example.cpp
)
target_link_libraries(http_server_example
    cllm_core
)

# Batch processing example
add_executable(batch_processing_example
    batch_processing_example.cpp
)
target_link_libraries(batch_processing_example
    cllm_core
)

# LibTorch inference example
add_executable(libtorch_inference_example
    libtorch_inference_example.cpp
)
target_link_libraries(libtorch_inference_example
    cllm_core
)

# LibTorch text generation example
add_executable(libtorch_text_generation
    libtorch_text_generation.cpp
)
target_link_libraries(libtorch_text_generation
    cllm_core
)

# LibTorch API integration test
add_executable(libtorch_api_test
    libtorch_api_test.cpp
)
target_link_libraries(libtorch_api_test
    cllm_core
)

# Minimal inference test
add_executable(minimal_inference_test
    minimal_inference_test.cpp
)
target_link_libraries(minimal_inference_test
    cllm_core
)

# CTokenizer example
add_executable(ctokenizer_example
    ctokenizer_example.cpp
)
target_link_libraries(ctokenizer_example
    cllm_core
)

# LibTorch Qwen test
add_executable(libtorch_qwen_test
    libtorch_qwen_test.cpp
)
target_link_libraries(libtorch_qwen_test
    cllm_core
)

# Integrated test
add_executable(integrated_test
    integrated_test.cpp
)
target_link_libraries(integrated_test
    cllm_core
)

# GGUF loader test example
add_executable(gguf_loader_test
    test_gguf_loader.cpp
)
target_link_libraries(gguf_loader_test
    cllm_core
)
# Dequantization test example
add_executable(dequantization_test
    test_dequantize.cpp
)
target_link_libraries(dequantization_test
    cllm_core
)

# Performance test example
add_executable(performance_test
    performance_test.cpp
)
target_link_libraries(performance_test
    cllm_core
)
