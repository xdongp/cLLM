#include "cllm/CTokenizer/qwen_tokenizer.h"
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
    // FIM 用于代码补全场景，格式：<|fim_pre|>middle<|fim_suf|>suffix<|fim_end|>
    
    // 定义FIM标记
    std::string fim_pre = "<|fim_pre|>";
    std::string fim_suf = "<|fim_suf|>";
    std::string fim_end = "<|fim_end|>";
    
    // 查找FIM标记位置
    size_t pre_pos = text.find(fim_pre);
    size_t suf_pos = text.find(fim_suf);
    size_t end_pos = text.find(fim_end);
    
    // 检查是否包含完整的FIM标记
    if (pre_pos == std::string::npos || suf_pos == std::string::npos || end_pos == std::string::npos) {
        // FIM标记不完整，检查是否是简化的 `` 格式
        size_t simple_fim_pos = text.find("``");
        if (simple_fim_pos != std::string::npos) {
            // 简化的FIM格式：prefix `` suffix
            std::string prefix = text.substr(0, simple_fim_pos);
            std::string suffix = text.substr(simple_fim_pos + 2);
            
            std::vector<llama_token> result;
            
            // 编码前缀
            if (!prefix.empty()) {
                auto prefix_tokens = SentencePieceTokenizer::encode(prefix, addSpecialTokens);
                result.insert(result.end(), prefix_tokens.begin(), prefix_tokens.end());
            }
            
            // 尝试添加FIM标记（如果存在于词汇表）
            llama_token fim_marker = tokenToId(fim_pre);
            llama_token unk_token = tokenToId("<unk>");
            if (fim_marker != unk_token && fim_marker != 0) {
                result.push_back(fim_marker);
            }
            
            // 编码后缀
            if (!suffix.empty()) {
                auto suffix_tokens = SentencePieceTokenizer::encode(suffix, false);
                result.insert(result.end(), suffix_tokens.begin(), suffix_tokens.end());
            }
            
            return result;
        }
        
        // 没有任何FIM标记，降级到普通编码
        return SentencePieceTokenizer::encode(text, addSpecialTokens);
    }
    
    // 验证FIM标记的顺序正确性
    if (!(pre_pos < suf_pos && suf_pos < end_pos)) {
        // FIM标记顺序错误，降级到普通编码
        return SentencePieceTokenizer::encode(text, addSpecialTokens);
    }
    
    // 提取各部分文本
    std::string prefix_text = text.substr(0, pre_pos);
    std::string middle_text = text.substr(pre_pos + fim_pre.length(), 
                                          suf_pos - (pre_pos + fim_pre.length()));
    std::string suffix_text = text.substr(suf_pos + fim_suf.length(), 
                                          end_pos - (suf_pos + fim_suf.length()));
    
    // 获取FIM特殊tokens
    llama_token fim_pre_token = tokenToId(fim_pre);
    llama_token fim_suf_token = tokenToId(fim_suf);
    llama_token fim_end_token = tokenToId(fim_end);
    llama_token unk_token = tokenToId("<unk>");
    
    // 验证FIM tokens是否有效
    if (fim_pre_token == unk_token || fim_suf_token == unk_token || fim_end_token == unk_token ||
        fim_pre_token == 0 || fim_suf_token == 0 || fim_end_token == 0) {
        // FIM tokens不存在于词汇表，降级到普通编码
        return SentencePieceTokenizer::encode(text, addSpecialTokens);
    }
    
    // 分别编码各部分
    std::vector<llama_token> result;
    
    // 编码前缀部分
    if (!prefix_text.empty()) {
        auto prefix_tokens = SentencePieceTokenizer::encode(prefix_text, addSpecialTokens);
        result.insert(result.end(), prefix_tokens.begin(), prefix_tokens.end());
    }
    
    // FIM格式: [fim_pre] middle [fim_suf] suffix [fim_end]
    result.push_back(fim_pre_token);
    
    // 编码中间部分（要填充的内容）
    if (!middle_text.empty()) {
        auto middle_tokens = SentencePieceTokenizer::encode(middle_text, false);
        result.insert(result.end(), middle_tokens.begin(), middle_tokens.end());
    }
    
    result.push_back(fim_suf_token);
    
    // 编码后缀部分
    if (!suffix_text.empty()) {
        auto suffix_tokens = SentencePieceTokenizer::encode(suffix_text, false);
        result.insert(result.end(), suffix_tokens.begin(), suffix_tokens.end());
    }
    
    result.push_back(fim_end_token);
    
    return result;
}

std::string QwenTokenizer::applyQwenPreprocessing(const std::string& text) {
    // Qwen2使用的预处理逻辑
    // 基于官方设计文档中的正则表达式模式：
    // 1. "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])": 匹配英语缩写
    // 2. "[^\r\n\p{L}\p{N}]?\p{L}+": 匹配字母序列（可选的非字母非数字前缀）
    // 3. "\p{N}": 匹配单个数字
    // 4. " ?[^\s\p{L}\p{N}]+[\r\n]*": 匹配标点符号（可选的前导空格和后缀换行）
    // 5. "\s*[\r\n]+": 匹配换行符（可选的前导空格）
    // 6. "\s+(?!\S)": 匹配尾随空白
    // 7. "\s+": 匹配其他空白字符
    
    if (text.empty()) {
        return text;
    }
    
    std::string result;
    result.reserve(text.size());
    
    // 注意：C++ std::regex 对 Unicode 属性类（\p{L}、\p{N}）支持有限
    // 这里使用字符类近似实现，以确保跨平台兼容性
    // 如果需要完整的 Unicode 支持，建议使用 RE2 或 PCRE2 库
    
    std::regex pattern(
        R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|)"  // 英语缩写
        R"([^\r\n\w\d]?[A-Za-zÀ-ÿ\u4E00-\u9FFF]+|)"                      // 字母序列（包含基本Unicode字符）
        R"(\d|)"                                                          // 单个数字
        R"( ?[^\s\w\d]+[\r\n]*|)"                                         // 标点符号
        R"(\s*[\r\n]+|)"                                                  // 换行符
        R"(\s+(?!\S)|)"                                                   // 尾随空白
        R"(\s+|)"                                                         // 其他空白
        R"(.)"                                                            // 其他单个字符（兜底）
    );
    
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        result += iter->str();
    }
    
    // 如果正则表达式处理失败（结果为空），返回原始文本
    return result.empty() ? text : result;
}

} // namespace cllm