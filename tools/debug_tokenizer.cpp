/**
 * @file debug_tokenizer.cpp
 * @brief 调试 tokenizer 的编码和解码
 */

#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/model/gguf_loader_new.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "Tokenizer Debug Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载 GGUF 模型
    cllm::GGUFLoader loader(modelPath);
    if (!loader.load()) {
        std::cerr << "Failed to load GGUF model" << std::endl;
        return 1;
    }
    
    // 初始化 tokenizer
    cllm::GGUFTokenizer tokenizer;
    if (!tokenizer.loadFromGGUFLoader(loader)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test 1: Encode 'Hello' ===" << std::endl;
    auto helloIds = tokenizer.encode("Hello", false);
    std::cout << "  Token IDs: [";
    for (size_t i = 0; i < helloIds.size(); ++i) {
        std::cout << helloIds[i];
        if (i < helloIds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 逐个解码
    std::cout << "  Individual tokens:" << std::endl;
    for (int id : helloIds) {
        std::string token = tokenizer.idToToken(id);
        std::string decoded = tokenizer.decode({id}, false);
        std::cout << "    ID " << id << " -> token='" << token << "' -> decoded='" << decoded << "'" << std::endl;
    }
    
    std::cout << "\n=== Test 2: Decode generated tokens ===" << std::endl;
    // Kylin backend 生成的 tokens
    std::vector<int> generatedIds = {38413, 2645, 22705, 26683, 29937};
    
    // llama.cpp 生成的 top tokens
    std::cout << "\n  llama.cpp top tokens:" << std::endl;
    std::vector<int> llamacppTokens = {105660, 322, 105534, 108652, 725};
    for (int id : llamacppTokens) {
        std::string token = tokenizer.idToToken(id);
        std::string decoded = tokenizer.decode({id}, false);
        std::cout << "    ID " << id << " -> token='" << token << "' -> decoded='" << decoded << "'" << std::endl;
    }
    
    std::cout << "\n  Kylin backend tokens:" << std::endl;
    std::cout << "  Generated IDs: [";
    for (size_t i = 0; i < generatedIds.size(); ++i) {
        std::cout << generatedIds[i];
        if (i < generatedIds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 逐个解码
    std::cout << "  Individual tokens:" << std::endl;
    for (int id : generatedIds) {
        std::string token = tokenizer.idToToken(id);
        std::string decoded = tokenizer.decode({id}, false);
        std::cout << "    ID " << id << " -> token='" << token << "' -> decoded='" << decoded << "'" << std::endl;
    }
    
    // 完整解码
    std::string fullDecoded = tokenizer.decode(generatedIds, false);
    std::cout << "  Full decoded: '" << fullDecoded << "'" << std::endl;
    
    std::cout << "\n=== Test 3: Check common tokens ===" << std::endl;
    std::vector<std::pair<std::string, int>> testCases = {
        {"Hello", 9707},
        {" world", -1},  // 需要查找
        {"The", -1},
        {" capital", -1},
        {"Paris", -1}
    };
    
    for (auto& [text, expectedId] : testCases) {
        auto ids = tokenizer.encode(text, false);
        std::cout << "  '" << text << "' -> IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i < ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        
        std::string decoded = tokenizer.decode(ids, false);
        std::cout << " -> decoded: '" << decoded << "'";
        
        if (!ids.empty() && ids.size() == 1) {
            std::string token = tokenizer.idToToken(ids[0]);
            std::cout << " (token='" << token << "')";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Test 4: Check special tokens ===" << std::endl;
    std::cout << "  BOS ID: " << tokenizer.getBosId() << std::endl;
    std::cout << "  EOS ID: " << tokenizer.getEosId() << std::endl;
    std::cout << "  Vocab size: " << tokenizer.getVocabSize() << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
