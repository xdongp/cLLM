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
