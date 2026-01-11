#include <gtest/gtest.h>
#include "cllm/tokenizer/unicode_utils.h"

using namespace cllm;

// UTF-8 编解码测试
TEST(UnicodeUtilsTest, Utf8CodepointConversion) {
    // ASCII
    std::string ascii = "Hello";
    auto codepoints = UnicodeUtils::utf8ToCodepoints(ascii);
    EXPECT_EQ(codepoints.size(), 5);
    EXPECT_EQ(codepoints[0], 0x48); // 'H'
    EXPECT_EQ(codepoints[4], 0x6F); // 'o'
    
    std::string reconstructed = UnicodeUtils::codepointsToUtf8(codepoints);
    EXPECT_EQ(reconstructed, ascii);
}

TEST(UnicodeUtilsTest, Utf8MultiByte) {
    // 中文字符
    std::string chinese = "你好";
    auto codepoints = UnicodeUtils::utf8ToCodepoints(chinese);
    EXPECT_EQ(codepoints.size(), 2);
    EXPECT_EQ(codepoints[0], 0x4F60); // '你'
    EXPECT_EQ(codepoints[1], 0x597D); // '好'
    
    std::string reconstructed = UnicodeUtils::codepointsToUtf8(codepoints);
    EXPECT_EQ(reconstructed, chinese);
}

TEST(UnicodeUtilsTest, Utf8Emoji) {
    // Emoji (4字节 UTF-8)
    std::string emoji = "😀";
    auto codepoints = UnicodeUtils::utf8ToCodepoints(emoji);
    EXPECT_EQ(codepoints.size(), 1);
    EXPECT_EQ(codepoints[0], 0x1F600);
    
    std::string reconstructed = UnicodeUtils::codepointsToUtf8(codepoints);
    EXPECT_EQ(reconstructed, emoji);
}

// UTF-8 验证测试
TEST(UnicodeUtilsTest, ValidUtf8) {
    EXPECT_TRUE(UnicodeUtils::isValidUtf8("Hello"));
    EXPECT_TRUE(UnicodeUtils::isValidUtf8("你好"));
    EXPECT_TRUE(UnicodeUtils::isValidUtf8("café"));
    EXPECT_TRUE(UnicodeUtils::isValidUtf8("😀🎉"));
}

TEST(UnicodeUtilsTest, InvalidUtf8) {
    // 非法的 UTF-8 序列
    std::string invalid1 = "\xFF\xFE";  // 非法起始字节
    EXPECT_FALSE(UnicodeUtils::isValidUtf8(invalid1));
    
    std::string invalid2 = "\xC0\x80";  // 过长编码
    EXPECT_FALSE(UnicodeUtils::isValidUtf8(invalid2));
}

// NFC 规范化测试
TEST(UnicodeUtilsTest, NFCNormalization) {
    // 测试 é 的组合形式 (e + 组合标记 ´) -> 预组合形式
    std::vector<uint32_t> decomposed = {0x0065, 0x0301}; // e + combining acute
    std::string decomposedStr = UnicodeUtils::codepointsToUtf8(decomposed);
    
    std::string normalized = UnicodeUtils::normalizeNFC(decomposedStr);
    auto normalizedCp = UnicodeUtils::utf8ToCodepoints(normalized);
    
    // 应该组合为预组合的 é (U+00E9)
    EXPECT_EQ(normalizedCp.size(), 1);
    EXPECT_EQ(normalizedCp[0], 0x00E9);
}

TEST(UnicodeUtilsTest, NFCPrecomposed) {
    // 已经是预组合形式的字符应该保持不变
    std::string precomposed = "café"; // é 是 U+00E9
    std::string normalized = UnicodeUtils::normalizeNFC(precomposed);
    EXPECT_EQ(normalized, precomposed);
}

TEST(UnicodeUtilsTest, NFCMultipleAccents) {
    // 多个重音符号
    std::vector<uint32_t> multiAccents = {
        0x0061, 0x0301, // á
        0x0065, 0x0300, // è
        0x006F, 0x0301  // ó
    };
    std::string multiStr = UnicodeUtils::codepointsToUtf8(multiAccents);
    std::string normalized = UnicodeUtils::normalizeNFC(multiStr);
    
    auto normalizedCp = UnicodeUtils::utf8ToCodepoints(normalized);
    EXPECT_EQ(normalizedCp.size(), 3);
    EXPECT_EQ(normalizedCp[0], 0x00E1); // á
    EXPECT_EQ(normalizedCp[1], 0x00E8); // è
    EXPECT_EQ(normalizedCp[2], 0x00F3); // ó
}

// NFD 规范化测试
TEST(UnicodeUtilsTest, NFDNormalization) {
    // 测试预组合的 é (U+00E9) -> 分解形式 (e + ´)
    std::string precomposed = "é";
    std::string normalized = UnicodeUtils::normalizeNFD(precomposed);
    
    auto normalizedCp = UnicodeUtils::utf8ToCodepoints(normalized);
    EXPECT_EQ(normalizedCp.size(), 2);
    EXPECT_EQ(normalizedCp[0], 0x0065); // 'e'
    EXPECT_EQ(normalizedCp[1], 0x0301); // combining acute
}

TEST(UnicodeUtilsTest, NFDDecomposed) {
    // 已经是分解形式的应该保持不变
    std::vector<uint32_t> decomposed = {0x0065, 0x0301};
    std::string decomposedStr = UnicodeUtils::codepointsToUtf8(decomposed);
    std::string normalized = UnicodeUtils::normalizeNFD(decomposedStr);
    EXPECT_EQ(normalized, decomposedStr);
}

// 实际应用场景测试
TEST(UnicodeUtilsTest, RealWorldCafe) {
    // "café" 可能有两种编码：
    // 1. c + a + f + é(U+00E9) - 预组合
    // 2. c + a + f + e + ´(U+0301) - 分解
    
    std::string precomposed = "café"; // 假设 é 是 U+00E9
    std::vector<uint32_t> decomposedCp = {0x0063, 0x0061, 0x0066, 0x0065, 0x0301};
    std::string decomposed = UnicodeUtils::codepointsToUtf8(decomposedCp);
    
    // 两种形式经过 NFC 规范化后应该相同
    std::string nfc1 = UnicodeUtils::normalizeNFC(precomposed);
    std::string nfc2 = UnicodeUtils::normalizeNFC(decomposed);
    EXPECT_EQ(nfc1, nfc2);
}

TEST(UnicodeUtilsTest, ChineseNoChange) {
    // 中文字符通常没有组合形式，应该保持不变
    std::string chinese = "你好世界";
    std::string nfc = UnicodeUtils::normalizeNFC(chinese);
    std::string nfd = UnicodeUtils::normalizeNFD(chinese);
    
    EXPECT_EQ(nfc, chinese);
    EXPECT_EQ(nfd, chinese);
}

TEST(UnicodeUtilsTest, EmptyString) {
    std::string empty = "";
    EXPECT_EQ(UnicodeUtils::normalizeNFC(empty), empty);
    EXPECT_EQ(UnicodeUtils::normalizeNFD(empty), empty);
    EXPECT_TRUE(UnicodeUtils::isValidUtf8(empty));
}

TEST(UnicodeUtilsTest, MixedContent) {
    // 混合 ASCII、中文、重音字符
    std::vector<uint32_t> mixed = {
        0x0048,        // H
        0x0065, 0x0301, // é (分解)
        0x006C, 0x006C, 0x006F, // llo
        0x4F60, 0x597D  // 你好
    };
    std::string mixedStr = UnicodeUtils::codepointsToUtf8(mixed);
    std::string normalized = UnicodeUtils::normalizeNFC(mixedStr);
    
    auto normalizedCp = UnicodeUtils::utf8ToCodepoints(normalized);
    // é 应该被组合
    EXPECT_LT(normalizedCp.size(), mixed.size());
}
