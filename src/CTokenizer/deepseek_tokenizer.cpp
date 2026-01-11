#include "cllm/CTokenizer/deepseek_tokenizer.h"
#include <regex>

namespace cllm {

std::string DeepSeekTokenizer::applyDeepSeekPreprocessing(const std::string& text) {
    // DeepSeek特定的预处理逻辑
    // 根据模型类型应用不同的正则表达式
    switch(getModelType()) {
        case ModelType::DEEPSEEK_LLM:
            return applyDeepSeekLLMPreprocessing(text);
        case ModelType::DEEPSEEK_CODER:
            return applyDeepSeekCoderPreprocessing(text);
        case ModelType::DEEPSEEK3_LLM:
            return applyDeepSeek3Preprocessing(text);
        default:
            return text;
    }
}

std::string DeepSeekTokenizer::applyDeepSeekLLMPreprocessing(const std::string& text) {
    // DeepSeek LLM使用的正则表达式模式：
    // 1. 换行符保持不变
    // 2. 字母序列（可选的前导空格）
    // 3. 标点符号（可选的前导空格）
    // 4. 中文字符（包括CJK统一表意文字）
    // 5. 数字序列
    
    if (text.empty()) {
        return text;
    }
    
    std::string result;
    result.reserve(text.size());
    
    // 使用正则表达式进行分段
    // 注意：这里简化实现，实际的DeepSeek LLM可能有更复杂的处理逻辑
    std::regex pattern(
        R"([\r\n]|)"                          // 换行符
        R"(\s?[A-Za-zÀ-ÿ]+|)"                 // 字母（包括带变音符号的）
        R"(\s?[!-/:-@\[-`{-~]+|)"            // 标点符号
        R"([\u4E00-\u9FFF\u3400-\u4DBF]+|)"  // 中文字符
        R"(\d+|)"                             // 数字
        R"(.)"                                // 其他单个字符
    );
    
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        result += iter->str();
    }
    
    // 如果正则表达式处理失败，返回原始文本
    return result.empty() ? text : result;
}

std::string DeepSeekTokenizer::applyDeepSeekCoderPreprocessing(const std::string& text) {
    // DeepSeek Coder使用的正则表达式模式：
    // 专门为代码分词优化
    // 1. 保持换行符
    // 2. 字母序列（可选的前导空格）
    // 3. 标点符号（可选的前导空格）
    // 4. 中文字符
    // 5. 数字（单个）
    
    if (text.empty()) {
        return text;
    }
    
    std::string result;
    result.reserve(text.size());
    
    // 针对代码的分词模式
    std::regex pattern(
        R"([\r\n]|)"                          // 换行符
        R"(\s?[A-Za-z_]\w*|)"                 // 标识符（字母、数字、下划线）
        R"(\s?[^\w\s\u4E00-\u9FFF]+|)"       // 标点和操作符
        R"([\u4E00-\u9FFF\u3400-\u4DBF]+|)"  // 中文字符
        R"(\d|)"                              // 单个数字
        R"(\s+|)"                             // 空白字符
        R"(.)"                                // 其他单个字符
    );
    
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        result += iter->str();
    }
    
    return result.empty() ? text : result;
}

std::string DeepSeekTokenizer::applyDeepSeek3Preprocessing(const std::string& text) {
    // DeepSeek3使用的正则表达式模式：
    // 更先进的分词策略
    // 1. 1-3位数字序列
    // 2. 中文字符（单个）
    // 3. 字母序列（可选的前导空格）
    // 4. 复杂的混合模式
    
    if (text.empty()) {
        return text;
    }
    
    std::string result;
    result.reserve(text.size());
    
    // DeepSeek3的分词模式（更精细的数字处理）
    std::regex pattern(
        R"(\d{1,3}|)"                         // 1-3位数字
        R"([\u4E00-\u9FFF\u3400-\u4DBF]|)"   // 单个中文字符
        R"(\s?[A-Za-zÀ-ÿ]+|)"                 // 字母序列
        R"([\r\n]|)"                          // 换行符
        R"([^\w\s\u4E00-\u9FFF]+|)"          // 标点符号
        R"(\s+|)"                             // 空白字符
        R"(.)"                                // 其他单个字符
    );
    
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        result += iter->str();
    }
    
    return result.empty() ? text : result;
}

} // namespace cllm