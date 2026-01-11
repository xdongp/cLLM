# CTokenizer module CMakeLists.txt

set(CTOKENIZER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/tokenizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/sentencepiece_tokenizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/llama_tokenizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/qwen_tokenizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/deepseek_tokenizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/model_detector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cllm/CTokenizer/manager.h
)

set(CTOKENIZER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/manager.cpp
)

# Export variables for parent CMakeLists.txt
set(CTOKENIZER_HEADERS ${CTOKENIZER_HEADERS} PARENT_SCOPE)
set(CTOKENIZER_SOURCES ${CTOKENIZER_SOURCES} PARENT_SCOPE)