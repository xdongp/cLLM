#include "cllm/CTokenizer/manager.h"
#include "cllm/CTokenizer/tokenizer.h"
#include <iostream>
#include <string>

int main() {
    // 创建分词器管理器
    cllm::TokenizerManager manager;
    
    // 获取Qwen分词器
    cllm::CTokenizer* qwenTokenizer = manager.getTokenizer("qwen");
    if (qwenTokenizer) {
        std::cout << "Successfully loaded Qwen tokenizer" << std::endl;
        
        std::string text = "Hello, world!";
        auto tokens = qwenTokenizer->encode(text);
        std::cout << "Encoded tokens count: " << tokens.size() << std::endl;
        
        std::string decoded = qwenTokenizer->decode(tokens);
        std::cout << "Decoded text: " << decoded << std::endl;
        
        std::cout << "Vocabulary size: " << qwenTokenizer->getVocabSize() << std::endl;
    } else {
        std::cout << "Failed to load Qwen tokenizer" << std::endl;
    }
    
    // 获取DeepSeek分词器
    cllm::CTokenizer* deepseekTokenizer = manager.getTokenizer("deepseek-coder");
    if (deepseekTokenizer) {
        std::cout << "Successfully loaded DeepSeek tokenizer" << std::endl;
        
        std::string code = "def hello():\n    return 'world'";
        auto tokens = deepseekTokenizer->encode(code);
        std::cout << "Encoded tokens count: " << tokens.size() << std::endl;
        
        std::string decoded = deepseekTokenizer->decode(tokens);
        std::cout << "Decoded text: " << decoded << std::endl;
    } else {
        std::cout << "Failed to load DeepSeek tokenizer" << std::endl;
    }
    
    return 0;
}