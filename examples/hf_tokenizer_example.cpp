/**
 * @file hf_tokenizer_example.cpp
 * @brief HFTokenizer 使用示例
 * 
 * 演示如何使用 HFTokenizer 加载和使用 HuggingFace 格式的 tokenizer
 */

#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/common/logger.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace cllm;

void printSeparator() {
    std::cout << std::string(60, '=') << std::endl;
}

void example1_basic_usage(const std::string& modelPath) {
    printSeparator();
    std::cout << "Example 1: 基本使用" << std::endl;
    printSeparator();
    
#ifdef USE_TOKENIZERS_CPP
    // 创建 HFTokenizer
    HFTokenizer tokenizer;
    
    // 加载模型
    std::cout << "加载模型: " << modelPath << std::endl;
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ 加载失败" << std::endl;
        return;
    }
    std::cout << "✅ 加载成功" << std::endl;
    
    // 显示基本信息
    std::cout << "\n模型信息:" << std::endl;
    std::cout << "  Vocab size: " << tokenizer.getVocabSize() << std::endl;
    std::cout << "  BOS ID: " << tokenizer.getBosId() << std::endl;
    std::cout << "  EOS ID: " << tokenizer.getEosId() << std::endl;
    std::cout << "  PAD ID: " << tokenizer.getPadId() << std::endl;
    std::cout << "  UNK ID: " << tokenizer.getUnkId() << std::endl;
    
    // 编码测试
    std::string text = "Hello, world!";
    std::cout << "\n编码文本: \"" << text << "\"" << std::endl;
    
    auto tokens = tokenizer.encode(text);
    std::cout << "Token IDs (" << tokens.size() << "): ";
    for (size_t i = 0; i < tokens.size() && i < 20; ++i) {
        std::cout << tokens[i] << " ";
    }
    if (tokens.size() > 20) std::cout << "...";
    std::cout << std::endl;
    
    // 解码测试
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "解码文本: \"" << decoded << "\"" << std::endl;
    
#else
    std::cerr << "❌ HFTokenizer 需要 USE_TOKENIZERS_CPP 编译选项" << std::endl;
    std::cerr << "   重新编译: cmake .. -DUSE_TOKENIZERS_CPP=ON" << std::endl;
#endif
}

void example2_chinese_text(const std::string& modelPath) {
    printSeparator();
    std::cout << "Example 2: 中文文本处理" << std::endl;
    printSeparator();
    
#ifdef USE_TOKENIZERS_CPP
    HFTokenizer tokenizer;
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ 加载失败" << std::endl;
        return;
    }
    
    // 中文文本测试
    std::vector<std::string> testTexts = {
        "你好，世界！",
        "这是一个测试。",
        "我喜欢编程。",
        "Hello, 世界！混合语言测试。"
    };
    
    for (const auto& text : testTexts) {
        std::cout << "\n文本: \"" << text << "\"" << std::endl;
        
        auto tokens = tokenizer.encode(text);
        std::cout << "  Token 数量: " << tokens.size() << std::endl;
        std::cout << "  Token IDs: ";
        for (size_t i = 0; i < tokens.size() && i < 10; ++i) {
            std::cout << tokens[i] << " ";
        }
        if (tokens.size() > 10) std::cout << "...";
        std::cout << std::endl;
        
        auto decoded = tokenizer.decode(tokens);
        std::cout << "  解码结果: \"" << decoded << "\"" << std::endl;
    }
#else
    std::cerr << "❌ 需要 USE_TOKENIZERS_CPP" << std::endl;
#endif
}

void example3_tokenize_method(const std::string& modelPath) {
    printSeparator();
    std::cout << "Example 3: Tokenize 方法" << std::endl;
    printSeparator();
    
#ifdef USE_TOKENIZERS_CPP
    HFTokenizer tokenizer;
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ 加载失败" << std::endl;
        return;
    }
    
    std::string text = "The quick brown fox jumps over the lazy dog";
    std::cout << "文本: \"" << text << "\"" << std::endl;
    
    auto tokens = tokenizer.tokenize(text);
    std::cout << "\nTokens (" << tokens.size() << "):" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "  [" << std::setw(2) << i << "] \"" << tokens[i] << "\"" << std::endl;
    }
#else
    std::cerr << "❌ 需要 USE_TOKENIZERS_CPP" << std::endl;
#endif
}

void example4_performance_test(const std::string& modelPath) {
    printSeparator();
    std::cout << "Example 4: 性能测试" << std::endl;
    printSeparator();
    
#ifdef USE_TOKENIZERS_CPP
    HFTokenizer tokenizer;
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ 加载失败" << std::endl;
        return;
    }
    
    // 准备测试文本
    std::string testText = "The quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 5; ++i) {
        testText += testText;  // 重复扩展
    }
    
    std::cout << "测试文本长度: " << testText.size() << " bytes" << std::endl;
    
    // 编码性能测试
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto tokens = tokenizer.encode(testText);
        (void)tokens;  // 避免优化掉
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n编码性能:" << std::endl;
    std::cout << "  迭代次数: " << iterations << std::endl;
    std::cout << "  总时间: " << duration.count() << " ms" << std::endl;
    std::cout << "  平均时间: " << (duration.count() / (float)iterations) << " ms/次" << std::endl;
    
    // 单次编码获取 token 数量
    auto tokens = tokenizer.encode(testText);
    std::cout << "  Token 数量: " << tokens.size() << std::endl;
    std::cout << "  吞吐量: " << (tokens.size() * iterations * 1000.0 / duration.count()) 
              << " tokens/s" << std::endl;
#else
    std::cerr << "❌ 需要 USE_TOKENIZERS_CPP" << std::endl;
#endif
}

void example5_tokenizer_manager(const std::string& modelPath) {
    printSeparator();
    std::cout << "Example 5: TokenizerManager 自动检测" << std::endl;
    printSeparator();
    
    try {
        // 使用 TokenizerManager 自动检测
        std::cout << "使用 TokenizerManager (AUTO 模式)" << std::endl;
        TokenizerManager manager(modelPath, nullptr, TokenizerManager::TokenizerImpl::AUTO);
        
        // 测试编码
        std::string text = "Hello from TokenizerManager!";
        auto tokens = manager.encode(text);
        
        std::cout << "  文本: \"" << text << "\"" << std::endl;
        std::cout << "  Token 数量: " << tokens.size() << std::endl;
        
        // 测试解码
        auto decoded = manager.decode(tokens);
        std::cout << "  解码: \"" << decoded << "\"" << std::endl;
        
        // 获取 tokenizer 类型
        auto* tokenizer = manager.getTokenizer();
        std::cout << "  Tokenizer 类型: " 
                  << (dynamic_cast<HFTokenizer*>(tokenizer) ? "HFTokenizer" : "其他")
                  << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "\n";
    printSeparator();
    std::cout << "       HFTokenizer 使用示例" << std::endl;
    printSeparator();
    std::cout << std::endl;
    
    // 检查参数
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <model_path>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "示例:" << std::endl;
        std::cerr << "  " << argv[0] << " /path/to/qwen2-model" << std::endl;
        std::cerr << "  " << argv[0] << " /path/to/deepseek-model" << std::endl;
        std::cerr << std::endl;
        std::cerr << "要求:" << std::endl;
        std::cerr << "  - 模型目录必须包含 tokenizer.json" << std::endl;
        std::cerr << "  - 编译时需要启用 USE_TOKENIZERS_CPP" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    // 检查编译选项
#ifndef USE_TOKENIZERS_CPP
    std::cerr << "❌ 错误: HFTokenizer 需要 USE_TOKENIZERS_CPP 编译选项" << std::endl;
    std::cerr << "   重新编译:" << std::endl;
    std::cerr << "     cd build" << std::endl;
    std::cerr << "     cmake .. -DUSE_TOKENIZERS_CPP=ON" << std::endl;
    std::cerr << "     make" << std::endl;
    return 1;
#endif
    
    // 运行示例
    try {
        example1_basic_usage(modelPath);
        std::cout << std::endl;
        
        example2_chinese_text(modelPath);
        std::cout << std::endl;
        
        example3_tokenize_method(modelPath);
        std::cout << std::endl;
        
        example4_performance_test(modelPath);
        std::cout << std::endl;
        
        example5_tokenizer_manager(modelPath);
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ 异常: " << e.what() << std::endl;
        return 1;
    }
    
    printSeparator();
    std::cout << "✅ 所有示例完成！" << std::endl;
    printSeparator();
    std::cout << std::endl;
    
    return 0;
}
