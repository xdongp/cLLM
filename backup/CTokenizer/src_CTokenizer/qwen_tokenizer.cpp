#include "cllm/CTokenizer/qwen_tokenizer.h"
#include <algorithm>
#include <regex>

namespace cllm {

bool QwenTokenizer::needsFimProcessing(const std::string& text) {
    // 检查是否需要FIM处理
    // Qwen模型特有的FIM tokens: <|fim_begin|>, <|fim_end|>, 
    return text.find("<|fim_begin|>") != std::string::npos || 
           text.find("<|fim_end|>") != std::string::npos ||
           text.find("``") != std::string::npos ||
           text.find("<|fim_suf|>") != std::string::npos ||
           text.find("<|fim_pre|>") != std::string::npos;
}

std::vector<llama_token> QwenTokenizer::encodeWithFim(const std::string& text, bool addSpecialTokens) {
    // 实现Qwen的FIM（Fill-in-the-Middle）处理逻辑
    // 这里需要识别FIM相关的特殊标记并进行相应处理
    
    // 查找FIM标记
    std::string fim_begin = "<|fim_begin|>";
    std::string fim_suffix = "<|fim_suf|>";
    std::string fim_end = "<|fim_end|>";
    
    // 在Qwen模型中，FIM格式通常是：``...```
    std::string fim_prefix = "<|fim_pre|>";
    std::string fim_middle = "``";
    
    // 检查文本中是否包含FIM标记
    size_t prefix_pos = text.find(fim_prefix);
    size_t suffix_pos = text.find(fim_suffix);
    size_t end_pos = text.find(fim_end);
    
    if (prefix_pos != std::string::npos && suffix_pos != std::string::npos && end_pos != std::string::npos) {
        // 如果找到了FIM标记，则按FIM方式进行分词
        std::string prefix = text.substr(0, prefix_pos);
        std::string suffix = text.substr(prefix_pos + fim_prefix.length(), suffix_pos - (prefix_pos + fim_prefix.length()));
        std::string middle = text.substr(suffix_pos + fim_suffix.length(), end_pos - (suffix_pos + fim_suffix.length()));
        
        // 分别对各部分进行编码
        std::vector<llama_token> result;
        auto prefix_tokens = SentencePieceTokenizer::encode(prefix, addSpecialTokens);
        auto suffix_tokens = SentencePieceTokenizer::encode(suffix, false); // FIM中间部分通常不加特殊token
        auto middle_tokens = SentencePieceTokenizer::encode(middle, false);
        
        // 按FIM格式组合
        result.insert(result.end(), prefix_tokens.begin(), prefix_tokens.end());
        
        // 添加FIM特殊标记
        result.push_back(tokenToId(fim_prefix));
        result.insert(result.end(), suffix_tokens.begin(), suffix_tokens.end());
        result.push_back(tokenToId(fim_suffix));
        result.insert(result.end(), middle_tokens.begin(), middle_tokens.end());
        result.push_back(tokenToId(fim_end));
        
        return result;
    } else {
        // 没有FIM标记，使用普通编码
        return SentencePieceTokenizer::encode(text, addSpecialTokens);
    }
}

std::string QwenTokenizer::applyQwenPreprocessing(const std::string& text) {
    // Qwen2使用的预处理逻辑
    // 正则表达式模式：
    // - "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])": 匹配英语缩写
    // - "[^\r\n\p{L}\p{N}]?\p{L}+": 匹配字母序列
    // - "\p{N}": 匹配数字
    // - 复杂的空白和标点处理模式
    
    // 这里只做简单的预处理示例，实际实现会更复杂
    return text;
}

} // namespace cllm