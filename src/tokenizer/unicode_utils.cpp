#include "cllm/tokenizer/unicode_utils.h"
#include <algorithm>
#include <stdexcept>

namespace cllm {

// UTF-8 解码：将字符串转为 Unicode 码点
std::vector<uint32_t> UnicodeUtils::utf8ToCodepoints(const std::string& text) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(text.size()); // 预估大小
    
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        int bytes = 0;
        
        // 判断字符字节长度
        if ((c & 0x80) == 0) {
            // 单字节 ASCII (0xxxxxxx)
            codepoint = c;
            bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 两字节 (110xxxxx 10xxxxxx)
            codepoint = c & 0x1F;
            bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 三字节 (1110xxxx 10xxxxxx 10xxxxxx)
            codepoint = c & 0x0F;
            bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 四字节 (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            codepoint = c & 0x07;
            bytes = 4;
        } else {
            // 非法 UTF-8 序列，跳过
            i++;
            continue;
        }
        
        // 读取后续字节
        bool valid = true;
        for (int j = 1; j < bytes && i + j < text.size(); j++) {
            unsigned char next = static_cast<unsigned char>(text[i + j]);
            if ((next & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (next & 0x3F);
        }
        
        if (valid && i + bytes <= text.size()) {
            codepoints.push_back(codepoint);
            i += bytes;
        } else {
            i++; // 非法序列，跳过首字节
        }
    }
    
    return codepoints;
}

// UTF-8 编码：将 Unicode 码点转为字符串
std::string UnicodeUtils::codepointsToUtf8(const std::vector<uint32_t>& codepoints) {
    std::string result;
    result.reserve(codepoints.size() * 3); // 预估大小
    
    for (uint32_t cp : codepoints) {
        if (cp <= 0x7F) {
            // 单字节 ASCII
            result.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            // 两字节
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            // 三字节
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0x10FFFF) {
            // 四字节
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    
    return result;
}

// 检查是否为有效的 UTF-8
bool UnicodeUtils::isValidUtf8(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int bytes = 0;
        
        if ((c & 0x80) == 0) {
            bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            bytes = 4;
        } else {
            return false;
        }
        
        if (i + bytes > text.size()) {
            return false;
        }
        
        for (int j = 1; j < bytes; j++) {
            if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
                return false;
            }
        }
        
        i += bytes;
    }
    
    return true;
}

// 获取组合类别（简化版，仅处理常见的组合标记）
int UnicodeUtils::getCombiningClass(uint32_t codepoint) {
    // 组合标记区域 (U+0300 - U+036F)
    if (codepoint >= 0x0300 && codepoint <= 0x036F) {
        return 230; // 大多数组合变音符号
    }
    // Hangul 组合 Jamo
    if (codepoint >= 0x1100 && codepoint <= 0x11FF) {
        return 0;
    }
    return 0; // 默认类别为 0
}

// NFC 组合查找表（简化版，仅处理常见情况）
uint32_t UnicodeUtils::getNFCComposed(uint32_t base, uint32_t combining) {
    // 常见的拉丁字母 + 变音符号组合
    struct ComposePair {
        uint32_t base;
        uint32_t combining;
        uint32_t composed;
    };
    
    static const ComposePair composeTable[] = {
        // 重音符号示例
        {0x0065, 0x0301, 0x00E9}, // e + ´ -> é
        {0x0065, 0x0300, 0x00E8}, // e + ` -> è
        {0x0065, 0x0302, 0x00EA}, // e + ^ -> ê
        {0x0061, 0x0301, 0x00E1}, // a + ´ -> á
        {0x0061, 0x0300, 0x00E0}, // a + ` -> à
        {0x006F, 0x0301, 0x00F3}, // o + ´ -> ó
        {0x006F, 0x0300, 0x00F2}, // o + ` -> ò
        {0x0069, 0x0301, 0x00ED}, // i + ´ -> í
        {0x0075, 0x0301, 0x00FA}, // u + ´ -> ú
        {0x006E, 0x0303, 0x00F1}, // n + ~ -> ñ
        // 可以扩展更多...
    };
    
    for (const auto& pair : composeTable) {
        if (pair.base == base && pair.combining == combining) {
            return pair.composed;
        }
    }
    
    return 0; // 未找到组合
}

// NFD 分解查找表（简化版）
std::vector<uint32_t> UnicodeUtils::getNFDDecomposed(uint32_t codepoint) {
    // 常见的预组合字符分解
    struct DecomposePair {
        uint32_t composed;
        uint32_t base;
        uint32_t combining;
    };
    
    static const DecomposePair decomposeTable[] = {
        {0x00E9, 0x0065, 0x0301}, // é -> e + ´
        {0x00E8, 0x0065, 0x0300}, // è -> e + `
        {0x00EA, 0x0065, 0x0302}, // ê -> e + ^
        {0x00E1, 0x0061, 0x0301}, // á -> a + ´
        {0x00E0, 0x0061, 0x0300}, // à -> a + `
        {0x00F3, 0x006F, 0x0301}, // ó -> o + ´
        {0x00F2, 0x006F, 0x0300}, // ò -> o + `
        {0x00ED, 0x0069, 0x0301}, // í -> i + ´
        {0x00FA, 0x0075, 0x0301}, // ú -> u + ´
        {0x00F1, 0x006E, 0x0303}, // ñ -> n + ~
    };
    
    for (const auto& pair : decomposeTable) {
        if (pair.composed == codepoint) {
            return {pair.base, pair.combining};
        }
    }
    
    return {codepoint}; // 无法分解，返回自身
}

// 规范等价排序
void UnicodeUtils::canonicalOrdering(std::vector<uint32_t>& codepoints) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i + 1 < codepoints.size(); i++) {
            int class1 = getCombiningClass(codepoints[i]);
            int class2 = getCombiningClass(codepoints[i + 1]);
            
            // 如果后面的组合类别更小（且不为0），交换
            if (class1 > class2 && class2 != 0) {
                std::swap(codepoints[i], codepoints[i + 1]);
                changed = true;
            }
        }
    }
}

// NFC 规范化
std::string UnicodeUtils::normalizeNFC(const std::string& text) {
    // 1. 先进行 NFD 分解
    std::vector<uint32_t> codepoints = utf8ToCodepoints(text);
    std::vector<uint32_t> decomposed;
    
    for (uint32_t cp : codepoints) {
        auto parts = getNFDDecomposed(cp);
        decomposed.insert(decomposed.end(), parts.begin(), parts.end());
    }
    
    // 2. 规范等价排序
    canonicalOrdering(decomposed);
    
    // 3. 组合
    std::vector<uint32_t> composed;
    size_t i = 0;
    while (i < decomposed.size()) {
        uint32_t base = decomposed[i];
        bool foundCompose = false;
        
        // 尝试与后续的组合标记组合
        if (i + 1 < decomposed.size()) {
            uint32_t combining = decomposed[i + 1];
            uint32_t result = getNFCComposed(base, combining);
            if (result != 0) {
                composed.push_back(result);
                i += 2;
                foundCompose = true;
            }
        }
        
        if (!foundCompose) {
            composed.push_back(base);
            i++;
        }
    }
    
    return codepointsToUtf8(composed);
}

// NFD 规范化
std::string UnicodeUtils::normalizeNFD(const std::string& text) {
    std::vector<uint32_t> codepoints = utf8ToCodepoints(text);
    std::vector<uint32_t> decomposed;
    
    for (uint32_t cp : codepoints) {
        auto parts = getNFDDecomposed(cp);
        decomposed.insert(decomposed.end(), parts.begin(), parts.end());
    }
    
    // 规范等价排序
    canonicalOrdering(decomposed);
    
    return codepointsToUtf8(decomposed);
}

} // namespace cllm
