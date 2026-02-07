#pragma once

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>

namespace cllm {
namespace test {

/**
 * @brief 测试数据生成辅助工具类
 * 提供创建测试数据的便捷方法
 */
class TestDataHelpers {
public:
    /**
     * @brief 生成随机字符串
     * @param length 字符串长度
     * @return 随机字符串
     */
    static std::string generateRandomString(size_t length) {
        static const char charset[] =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789";
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
        
        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            result += charset[dis(gen)];
        }
        return result;
    }
    
    /**
     * @brief 生成随机中文字符串
     * @param length 字符串长度（字符数）
     * @return 随机中文字符串
     */
    static std::string generateRandomChineseString(size_t length) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0x4E00, 0x9FA5);
        
        std::string result;
        for (size_t i = 0; i < length; ++i) {
            int codepoint = dis(gen);
            // 将Unicode编码点转换为UTF-8
            result += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        return result;
    }
    
    /**
     * @brief 生成随机token序列
     * @param length token数量
     * @param vocabSize 词汇表大小
     * @return token序列
     */
    static std::vector<int> generateRandomTokens(size_t length, int vocabSize = 10000) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, vocabSize - 1);
        
        std::vector<int> tokens;
        tokens.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            tokens.push_back(dis(gen));
        }
        return tokens;
    }
    
    /**
     * @brief 生成测试提示词列表
     * @return 测试提示词向量
     */
    static std::vector<std::string> generateTestPrompts() {
        return {
            "Hello, world!",
            "What is the meaning of life?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "Translate 'hello' to Chinese.",
            "Calculate the sum of 1 to 100.",
            "What are the benefits of exercise?",
            "Describe the process of photosynthesis.",
            "Who invented the telephone?",
            "What is artificial intelligence?"
        };
    }
    
    /**
     * @brief 生成中文测试提示词列表
     * @return 中文测试提示词向量
     */
    static std::vector<std::string> generateChineseTestPrompts() {
        return {
            "你好，世界！",
            "生命的意义是什么？",
            "用简单的术语解释量子计算。",
            "写一个关于机器人的短篇故事。",
            "将'hello'翻译成中文。",
            "计算1到100的和。",
            "运动的好处是什么？",
            "描述光合作用的过程。",
            "谁发明了电话？",
            "什么是人工智能？"
        };
    }
    
    /**
     * @brief 创建临时文件
     * @param content 文件内容
     * @param filename 文件名
     * @param tempDir 临时目录
     * @return 文件路径
     */
    static std::filesystem::path createTempFile(
        const std::string& content,
        const std::string& filename,
        const std::filesystem::path& tempDir) {
        
        auto filePath = tempDir / filename;
        std::ofstream file(filePath);
        if (!file) {
            throw std::runtime_error("Failed to create temp file: " + filePath.string());
        }
        file << content;
        file.close();
        return filePath;
    }
    
    /**
     * @brief 创建临时JSON配置文件
     * @param jsonContent JSON内容
     * @param filename 文件名
     * @param tempDir 临时目录
     * @return 文件路径
     */
    static std::filesystem::path createTempJsonFile(
        const std::string& jsonContent,
        const std::string& filename,
        const std::filesystem::path& tempDir) {
        
        return createTempFile(jsonContent, filename, tempDir);
    }
    
    /**
     * @brief 生成压力测试数据
     * @param numRequests 请求数量
     * @param minTokens 最小token数
     * @param maxTokens 最大token数
     * @return 测试数据向量
     */
    struct StressTestData {
        std::string prompt;
        int maxTokens;
        float temperature;
        float topP;
    };
    
    static std::vector<StressTestData> generateStressTestData(
        size_t numRequests,
        int minTokens = 10,
        int maxTokens = 100) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> tokenDis(minTokens, maxTokens);
        std::uniform_real_distribution<float> tempDis(0.1f, 1.5f);
        std::uniform_real_distribution<float> topPDis(0.5f, 1.0f);
        
        std::vector<StressTestData> data;
        data.reserve(numRequests);
        
        auto prompts = generateTestPrompts();
        
        for (size_t i = 0; i < numRequests; ++i) {
            StressTestData item;
            item.prompt = prompts[i % prompts.size()];
            item.maxTokens = tokenDis(gen);
            item.temperature = tempDis(gen);
            item.topP = topPDis(gen);
            data.push_back(item);
        }
        
        return data;
    }
};

} // namespace test
} // namespace cllm
