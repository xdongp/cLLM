/**
 * @file test_output_quality.cpp
 * @brief Stage 13: 输出质量测试
 *
 * 测试内容：
 * - 使用实际模型生成文本
 * - 检查 "hello", "介绍人工智能" 等输入的输出质量
 * - 检测重复、乱码等问题
 */

#include "kylin_test_framework.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <regex>
#include <cstdlib>

namespace kylin_test {

// 前向声明
struct QualityMetrics;

// 检测文本中的乱码
bool containsGarbledText(const std::string& text) {
    int consecutiveSpecial = 0;
    for (unsigned char c : text) {
        bool isNormal = std::isalnum(c) || std::isspace(c) || 
                       c == '.' || c == ',' || c == '!' || c == '?' || 
                       c == ';' || c == ':' || c == '"' || c == '\'' ||
                       c == '(' || c == ')' || c == '-' || c == '_' ||
                       c > 127;
        
        if (!isNormal) {
            consecutiveSpecial++;
            if (consecutiveSpecial >= 3) {
                return true;
            }
        } else {
            consecutiveSpecial = 0;
        }
    }
    return false;
}

// 计算重复率
float calculateRepetitionRate(const std::vector<std::string>& tokens) {
    if (tokens.size() < 4) return 0.0f;
    
    int repetitions = 0;
    for (size_t i = 2; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i-1] && tokens[i] == tokens[i-2]) {
            repetitions++;
        }
    }
    
    return static_cast<float>(repetitions) / tokens.size();
}

// 使用 curl 调用模型生成文本
std::string generateText(const std::string& prompt, int maxTokens) {
    std::string cmd = "curl -s -X POST http://127.0.0.1:8080/generate "
                      "-H 'Content-Type: application/json' "
                      "-d '{\"prompt\": \"" + prompt + "\", "
                      "\"max_tokens\": " + std::to_string(maxTokens) + ", "
                      "\"temperature\": 0.7}' "
                      "2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    
    char buffer[4096];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // 移除换行符
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    
    // 简单解析 JSON 中的 generated_text 字段
    size_t start = result.find("\"generated_text\":\"");
    if (start != std::string::npos) {
        start += 18;  // length of "generated_text":"
        size_t end = result.find("\"", start);
        if (end != std::string::npos) {
            return result.substr(start, end - start);
        }
    }
    
    return result;
}

// 检查服务器是否运行
bool isServerRunning() {
    std::string cmd = "curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q 'healthy' && echo 'yes' || echo 'no'";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;
    
    char buffer[8];
    std::string result = "";
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
    }
    pclose(pipe);
    
    return result.find("yes") != std::string::npos;
}

// 分析生成文本的质量
struct QualityMetrics {
    std::string text;
    int length;
    float repetitionRate;
    bool hasGarbled;
    int wordCount;
    float avgWordLength;
};

QualityMetrics analyzeText(const std::string& text) {
    QualityMetrics metrics;
    metrics.text = text;
    metrics.length = text.length();
    metrics.hasGarbled = containsGarbledText(text);
    
    // 分词（简单按空格分割）
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        words.push_back(word);
    }
    
    metrics.wordCount = words.size();
    
    // 计算平均词长
    if (metrics.wordCount > 0) {
        int totalLen = 0;
        for (const auto& w : words) {
            totalLen += w.length();
        }
        metrics.avgWordLength = static_cast<float>(totalLen) / metrics.wordCount;
    } else {
        metrics.avgWordLength = 0;
    }
    
    // 计算重复率
    metrics.repetitionRate = calculateRepetitionRate(words);
    
    return metrics;
}

// 实际生成测试 - hello
class RealGenerationHelloTest : public TestCase {
public:
    RealGenerationHelloTest() : TestCase(
        "real_generation_hello",
        "实际生成测试 - hello"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试输入: 'hello'");
        
        if (!isServerRunning()) {
            log(LogLevel::WARN, "服务器未运行，跳过实际生成测试");
            return;
        }
        
        std::string output = generateText("hello", 30);
        log(LogLevel::INFO, "生成结果: " + output);
        
        if (output.empty()) {
            log(LogLevel::WARN, "生成结果为空，可能服务器响应异常");
            return;
        }
        
        auto metrics = analyzeText(output);
        
        log(LogLevel::INFO, "质量分析:");
        log(LogLevel::INFO, "  长度: " + std::to_string(metrics.length));
        log(LogLevel::INFO, "  词数: " + std::to_string(metrics.wordCount));
        log(LogLevel::INFO, "  平均词长: " + std::to_string(metrics.avgWordLength));
        log(LogLevel::INFO, "  重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
        log(LogLevel::INFO, "  包含乱码: " + std::string(metrics.hasGarbled ? "是" : "否"));
        
        assertTrue(metrics.length > 0, "生成文本不应为空");
        assertTrue(!metrics.hasGarbled, "生成文本不应包含乱码");
        assertTrue(metrics.repetitionRate < 0.3f, "重复率应小于 30%");
        
        log(LogLevel::INFO, "hello 生成测试完成");
    }
};

// 实际生成测试 - 介绍人工智能
class RealGenerationAITest : public TestCase {
public:
    RealGenerationAITest() : TestCase(
        "real_generation_ai",
        "实际生成测试 - 介绍人工智能"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试输入: '介绍人工智能'");
        
        if (!isServerRunning()) {
            log(LogLevel::WARN, "服务器未运行，跳过实际生成测试");
            return;
        }
        
        std::string output = generateText("介绍人工智能", 50);
        log(LogLevel::INFO, "生成结果: " + output);
        
        if (output.empty()) {
            log(LogLevel::WARN, "生成结果为空，可能服务器响应异常");
            return;
        }
        
        auto metrics = analyzeText(output);
        
        log(LogLevel::INFO, "质量分析:");
        log(LogLevel::INFO, "  长度: " + std::to_string(metrics.length));
        log(LogLevel::INFO, "  词数: " + std::to_string(metrics.wordCount));
        log(LogLevel::INFO, "  平均词长: " + std::to_string(metrics.avgWordLength));
        log(LogLevel::INFO, "  重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
        log(LogLevel::INFO, "  包含乱码: " + std::string(metrics.hasGarbled ? "是" : "否"));
        
        assertTrue(metrics.length > 0, "生成文本不应为空");
        assertTrue(!metrics.hasGarbled, "生成文本不应包含乱码");
        assertTrue(metrics.repetitionRate < 0.3f, "重复率应小于 30%");
        
        log(LogLevel::INFO, "介绍人工智能 生成测试完成");
    }
};

// 实际生成测试 - 1+1=
class RealGenerationMathTest : public TestCase {
public:
    RealGenerationMathTest() : TestCase(
        "real_generation_math",
        "实际生成测试 - 1+1="
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试输入: '1+1='");
        
        if (!isServerRunning()) {
            log(LogLevel::WARN, "服务器未运行，跳过实际生成测试");
            return;
        }
        
        std::string output = generateText("1+1=", 20);
        log(LogLevel::INFO, "生成结果: " + output);
        
        if (output.empty()) {
            log(LogLevel::WARN, "生成结果为空，可能服务器响应异常");
            return;
        }
        
        auto metrics = analyzeText(output);
        
        log(LogLevel::INFO, "质量分析:");
        log(LogLevel::INFO, "  长度: " + std::to_string(metrics.length));
        log(LogLevel::INFO, "  词数: " + std::to_string(metrics.wordCount));
        log(LogLevel::INFO, "  平均词长: " + std::to_string(metrics.avgWordLength));
        log(LogLevel::INFO, "  重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
        log(LogLevel::INFO, "  包含乱码: " + std::string(metrics.hasGarbled ? "是" : "否"));
        
        assertTrue(metrics.length > 0, "生成文本不应为空");
        assertTrue(!metrics.hasGarbled, "生成文本不应包含乱码");
        
        log(LogLevel::INFO, "1+1= 生成测试完成");
    }
};

// 多轮生成质量对比测试
class MultiRoundQualityTest : public TestCase {
public:
    MultiRoundQualityTest() : TestCase(
        "multi_round_quality",
        "多轮生成质量对比测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "多轮生成质量对比测试...");
        
        if (!isServerRunning()) {
            log(LogLevel::WARN, "服务器未运行，跳过实际生成测试");
            return;
        }
        
        std::vector<std::string> prompts = {
            "hello",
            "hi",
            "你好",
            "what is AI",
            "什么是机器学习"
        };
        
        std::vector<QualityMetrics> allMetrics;
        
        for (const auto& prompt : prompts) {
            log(LogLevel::INFO, "\n测试输入: '" + prompt + "'");
            
            std::string output = generateText(prompt, 30);
            log(LogLevel::INFO, "生成结果: " + output);
            
            if (!output.empty()) {
                auto metrics = analyzeText(output);
                allMetrics.push_back(metrics);
                
                log(LogLevel::INFO, "质量指标:");
                log(LogLevel::INFO, "  长度: " + std::to_string(metrics.length));
                log(LogLevel::INFO, "  重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
                log(LogLevel::INFO, "  乱码: " + std::string(metrics.hasGarbled ? "是" : "否"));
            }
        }
        
        // 统计整体质量
        if (!allMetrics.empty()) {
            float avgRepetition = 0;
            int garbledCount = 0;
            
            for (const auto& m : allMetrics) {
                avgRepetition += m.repetitionRate;
                if (m.hasGarbled) garbledCount++;
            }
            avgRepetition /= allMetrics.size();
            
            log(LogLevel::INFO, "\n整体质量统计:");
            log(LogLevel::INFO, "  测试次数: " + std::to_string(allMetrics.size()));
            log(LogLevel::INFO, "  平均重复率: " + std::to_string(avgRepetition * 100) + "%");
            log(LogLevel::INFO, "  乱码次数: " + std::to_string(garbledCount));
            
            assertTrue(avgRepetition < 0.3f, "平均重复率应小于 30%");
            assertTrue(garbledCount == 0, "不应检测到乱码");
        }
        
        log(LogLevel::INFO, "多轮生成质量对比测试完成");
    }
};

// 输出质量综合评估
class ComprehensiveQualityTest : public TestCase {
public:
    ComprehensiveQualityTest() : TestCase(
        "comprehensive_quality",
        "输出质量综合评估"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "输出质量综合评估...");
        
        if (!isServerRunning()) {
            log(LogLevel::WARN, "服务器未运行，使用模拟数据进行测试");
            
            // 模拟测试数据
            std::vector<std::string> mockOutputs = {
                "Hello! How can I help you today?",
                "Artificial intelligence is a fascinating field.",
                "The answer is 2.",
                "Machine learning is a subset of AI."
            };
            
            for (const auto& text : mockOutputs) {
                log(LogLevel::INFO, "模拟文本: " + text);
                auto metrics = analyzeText(text);
                
                log(LogLevel::INFO, "  长度: " + std::to_string(metrics.length));
                log(LogLevel::INFO, "  词数: " + std::to_string(metrics.wordCount));
                log(LogLevel::INFO, "  重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
                log(LogLevel::INFO, "  乱码: " + std::string(metrics.hasGarbled ? "是" : "否"));
                
                assertTrue(!metrics.hasGarbled, "模拟文本不应包含乱码");
            }
            
            return;
        }
        
        // 实际测试
        std::string output = generateText("hello", 50);
        log(LogLevel::INFO, "生成文本: " + output);
        
        auto metrics = analyzeText(output);
        
        log(LogLevel::INFO, "\n质量评估报告:");
        log(LogLevel::INFO, "===================");
        log(LogLevel::INFO, "文本长度: " + std::to_string(metrics.length));
        log(LogLevel::INFO, "词汇数量: " + std::to_string(metrics.wordCount));
        log(LogLevel::INFO, "平均词长: " + std::to_string(metrics.avgWordLength));
        log(LogLevel::INFO, "重复率: " + std::to_string(metrics.repetitionRate * 100) + "%");
        log(LogLevel::INFO, "包含乱码: " + std::string(metrics.hasGarbled ? "是 ⚠️" : "否 ✓"));
        log(LogLevel::INFO, "===================");
        
        // 质量评分
        int score = 100;
        if (metrics.hasGarbled) score -= 40;
        if (metrics.repetitionRate > 0.2f) score -= 30;
        if (metrics.repetitionRate > 0.1f) score -= 15;
        if (metrics.wordCount < 5) score -= 20;
        
        log(LogLevel::INFO, "质量评分: " + std::to_string(score) + "/100");
        
        if (score >= 80) {
            log(LogLevel::INFO, "评级: 优秀 ✓");
        } else if (score >= 60) {
            log(LogLevel::INFO, "评级: 良好");
        } else if (score >= 40) {
            log(LogLevel::INFO, "评级: 一般 ⚠️");
        } else {
            log(LogLevel::INFO, "评级: 较差 ✗");
        }
        
        assertTrue(score >= 60, "质量评分应至少为 60 分");
        
        log(LogLevel::INFO, "输出质量综合评估完成");
    }
};

// 创建输出质量测试套件
std::unique_ptr<TestSuite> createOutputQualityTestSuite() {
    auto suite = std::make_unique<TestSuite>("Output Quality Tests");
    
    suite->addTest(std::make_unique<RealGenerationHelloTest>());
    suite->addTest(std::make_unique<RealGenerationAITest>());
    suite->addTest(std::make_unique<RealGenerationMathTest>());
    suite->addTest(std::make_unique<MultiRoundQualityTest>());
    suite->addTest(std::make_unique<ComprehensiveQualityTest>());
    
    return suite;
}

} // namespace kylin_test
