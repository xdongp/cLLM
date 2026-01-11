#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace cllm {

/**
 * @brief 轻量级 Unicode 工具类（无外部依赖）
 * 
 * 提供基本的 UTF-8 处理和 NFC 规范化功能
 */
class UnicodeUtils {
public:
    /**
     * @brief 对 UTF-8 字符串进行 NFC (Canonical Composition) 规范化
     * 
     * NFC 规范化会将组合字符序列转换为预组合形式，例如：
     * - "café" (U+0063 U+0061 U+0066 U+0065 U+0301) -> "café" (U+0063 U+0061 U+0066 U+00E9)
     * - "한글" 的组合字母 -> 预组合的 Hangul 音节
     * 
     * @param text 输入的 UTF-8 文本
     * @return 规范化后的 UTF-8 文本
     */
    static std::string normalizeNFC(const std::string& text);

    /**
     * @brief 对 UTF-8 字符串进行 NFD (Canonical Decomposition) 规范化
     * 
     * NFD 规范化会将预组合字符分解为基字符 + 组合标记，例如：
     * - "café" (U+00E9) -> "café" (U+0065 U+0301)
     * 
     * @param text 输入的 UTF-8 文本
     * @return 规范化后的 UTF-8 文本
     */
    static std::string normalizeNFD(const std::string& text);

    /**
     * @brief 解码 UTF-8 字符串为 Unicode 码点序列
     * @param text UTF-8 编码的文本
     * @return Unicode 码点（uint32_t）数组
     */
    static std::vector<uint32_t> utf8ToCodepoints(const std::string& text);

    /**
     * @brief 将 Unicode 码点序列编码为 UTF-8 字符串
     * @param codepoints Unicode 码点数组
     * @return UTF-8 编码的文本
     */
    static std::string codepointsToUtf8(const std::vector<uint32_t>& codepoints);

    /**
     * @brief 检查字符串是否为有效的 UTF-8
     * @param text 待检查的字符串
     * @return 如果是有效的 UTF-8 返回 true
     */
    static bool isValidUtf8(const std::string& text);

private:
    // Unicode 组合类别（Combining Class）
    static int getCombiningClass(uint32_t codepoint);
    
    // 获取字符的 NFC 组合形式
    static uint32_t getNFCComposed(uint32_t base, uint32_t combining);
    
    // 获取字符的 NFD 分解形式
    static std::vector<uint32_t> getNFDDecomposed(uint32_t codepoint);
    
    // 规范等价排序（Canonical Ordering）
    static void canonicalOrdering(std::vector<uint32_t>& codepoints);
};

} // namespace cllm
