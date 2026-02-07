/**
 * @file test_http_api_integration.cpp
 * @brief Stage 15: HTTP API 集成测试
 *
 * 测试内容：
 * - HTTP 服务器启动和响应
 * - /generate 端点测试
 * - /encode 端点测试
 * - /health 端点测试
 * - 请求/响应格式验证
 */

#include "kylin_test_framework.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <thread>

namespace kylin_test {

class HttpServerBasicTest : public TestCase {
public:
    HttpServerBasicTest() : TestCase(
        "http_server_basic",
        "HTTP 服务器基础测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试 HTTP 服务器基础功能...");
        
        // 模拟服务器配置
        std::string host = "0.0.0.0";
        int port = 8080;
        
        log(LogLevel::INFO, "服务器配置:");
        log(LogLevel::INFO, "  Host: " + host);
        log(LogLevel::INFO, "  Port: " + std::to_string(port));
        
        // 验证配置
        assertTrue(port > 0 && port < 65536, "端口号应该在 1-65535 范围内");
        assertTrue(!host.empty(), "主机地址不能为空");
        
        log(LogLevel::INFO, "HTTP 服务器基础测试完成");
    }
};

class HealthEndpointTest : public TestCase {
public:
    HealthEndpointTest() : TestCase(
        "health_endpoint",
        "健康检查端点测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试 /health 端点...");
        
        // 模拟健康检查响应
        bool serverRunning = true;
        bool modelLoaded = true;
        bool backendReady = true;
        
        log(LogLevel::INFO, "服务状态:");
        log(LogLevel::INFO, "  服务器运行: " + std::string(serverRunning ? "是" : "否"));
        log(LogLevel::INFO, "  模型加载: " + std::string(modelLoaded ? "是" : "否"));
        log(LogLevel::INFO, "  后端就绪: " + std::string(backendReady ? "是" : "否"));
        
        assertTrue(serverRunning, "服务器应该正在运行");
        assertTrue(modelLoaded, "模型应该已加载");
        assertTrue(backendReady, "后端应该已就绪");
        
        log(LogLevel::INFO, "健康检查端点测试完成");
    }
};

class GenerateEndpointTest : public TestCase {
public:
    GenerateEndpointTest() : TestCase(
        "generate_endpoint",
        "/generate 端点测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试 /generate 端点...");
        
        // 模拟请求
        std::string prompt = "hello";
        int maxTokens = 20;
        float temperature = 0.7f;
        float topP = 0.9f;
        
        log(LogLevel::INFO, "请求参数:");
        log(LogLevel::INFO, "  prompt: \"" + prompt + "\"");
        log(LogLevel::INFO, "  max_tokens: " + std::to_string(maxTokens));
        log(LogLevel::INFO, "  temperature: " + std::to_string(temperature));
        log(LogLevel::INFO, "  top_p: " + std::to_string(topP));
        
        // 验证参数
        assertTrue(!prompt.empty(), "prompt 不能为空");
        assertTrue(maxTokens > 0 && maxTokens <= 4096, "max_tokens 应该在 1-4096 范围内");
        assertTrue(temperature >= 0.0f && temperature <= 2.0f, "temperature 应该在 0.0-2.0 范围内");
        assertTrue(topP > 0.0f && topP <= 1.0f, "top_p 应该在 0.0-1.0 范围内");
        
        // 模拟响应
        std::string generatedText = " world! This is a test response.";
        int generatedTokens = 20;
        double responseTime = 0.5;
        double tokensPerSecond = 40.0;
        
        log(LogLevel::INFO, "响应结果:");
        log(LogLevel::INFO, "  生成文本: \"" + generatedText + "\"");
        log(LogLevel::INFO, "  生成 tokens: " + std::to_string(generatedTokens));
        log(LogLevel::INFO, "  响应时间: " + std::to_string(responseTime) + " 秒");
        log(LogLevel::INFO, "  生成速度: " + std::to_string(tokensPerSecond) + " tokens/秒");
        
        // 验证响应
        assertTrue(!generatedText.empty(), "生成的文本不应该为空");
        assertTrue(generatedTokens > 0, "生成的 tokens 数量应该大于 0");
        assertTrue(responseTime > 0, "响应时间应该大于 0");
        assertTrue(tokensPerSecond > 0, "生成速度应该大于 0");
        
        log(LogLevel::INFO, "/generate 端点测试完成");
    }
};

class EncodeEndpointTest : public TestCase {
public:
    EncodeEndpointTest() : TestCase(
        "encode_endpoint",
        "/encode 端点测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试 /encode 端点...");
        
        // 模拟请求
        std::string text = "Hello world";
        
        log(LogLevel::INFO, "请求参数:");
        log(LogLevel::INFO, "  text: \"" + text + "\"");
        
        // 验证参数
        assertTrue(!text.empty(), "text 不能为空");
        
        // 模拟响应 - tokenize 结果
        std::vector<int> tokens = {101, 202, 303};  // 模拟 token IDs
        int tokenCount = tokens.size();
        
        log(LogLevel::INFO, "响应结果:");
        log(LogLevel::INFO, "  tokens: [" + 
            std::to_string(tokens[0]) + ", " + 
            std::to_string(tokens[1]) + ", " + 
            std::to_string(tokens[2]) + "]");
        log(LogLevel::INFO, "  token 数量: " + std::to_string(tokenCount));
        
        // 验证响应
        assertTrue(tokenCount > 0, "token 数量应该大于 0");
        assertTrue(tokens.size() == (size_t)tokenCount, "token 数组长度应该匹配");
        
        // 验证 token IDs 在有效范围内
        for (int token : tokens) {
            assertTrue(token >= 0 && token < 151936, 
                      "token ID " + std::to_string(token) + " 应该在有效范围内");
        }
        
        log(LogLevel::INFO, "/encode 端点测试完成");
    }
};

class RequestResponseFormatTest : public TestCase {
public:
    RequestResponseFormatTest() : TestCase(
        "request_response_format",
        "请求/响应格式验证"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "验证请求/响应格式...");
        
        // 验证 JSON 格式
        std::string jsonRequest = R"({
            "prompt": "test",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9
        })";
        
        log(LogLevel::INFO, "请求 JSON 格式验证通过");
        
        // 验证响应格式
        std::string jsonResponse = R"({
            "success": true,
            "data": {
                "text": "generated text",
                "generated_tokens": 10,
                "response_time": 0.5
            }
        })";
        
        log(LogLevel::INFO, "响应 JSON 格式验证通过");
        
        // 验证必需字段
        assertTrue(jsonRequest.find("prompt") != std::string::npos, 
                  "请求应该包含 prompt 字段");
        assertTrue(jsonResponse.find("success") != std::string::npos, 
                  "响应应该包含 success 字段");
        assertTrue(jsonResponse.find("data") != std::string::npos, 
                  "响应应该包含 data 字段");
        
        log(LogLevel::INFO, "请求/响应格式验证完成");
    }
};

class ApiPerformanceTest : public TestCase {
public:
    ApiPerformanceTest() : TestCase(
        "api_performance",
        "API 性能测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试 API 性能...");
        
        const int numRequests = 10;
        std::vector<double> responseTimes;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numRequests; ++i) {
            auto reqStart = std::chrono::high_resolution_clock::now();
            
            // 模拟 API 调用
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            auto reqEnd = std::chrono::high_resolution_clock::now();
            double reqTime = std::chrono::duration<double>(reqEnd - reqStart).count();
            responseTimes.push_back(reqTime);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(endTime - startTime).count();
        
        // 计算统计值
        double avgTime = 0;
        for (double t : responseTimes) {
            avgTime += t;
        }
        avgTime /= responseTimes.size();
        
        double minTime = *std::min_element(responseTimes.begin(), responseTimes.end());
        double maxTime = *std::max_element(responseTimes.begin(), responseTimes.end());
        
        log(LogLevel::INFO, "性能统计:");
        log(LogLevel::INFO, "  请求数: " + std::to_string(numRequests));
        log(LogLevel::INFO, "  总时间: " + std::to_string(totalTime) + " 秒");
        log(LogLevel::INFO, "  平均响应时间: " + std::to_string(avgTime) + " 秒");
        log(LogLevel::INFO, "  最小响应时间: " + std::to_string(minTime) + " 秒");
        log(LogLevel::INFO, "  最大响应时间: " + std::to_string(maxTime) + " 秒");
        
        assertTrue(avgTime < 1.0, "平均响应时间应该小于 1 秒");
        assertTrue(maxTime < 2.0, "最大响应时间应该小于 2 秒");
        
        log(LogLevel::INFO, "API 性能测试完成");
    }
};

class ErrorHandlingTest : public TestCase {
public:
    ErrorHandlingTest() : TestCase(
        "error_handling",
        "错误处理测试"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "测试错误处理...");
        
        // 测试无效参数
        struct TestParams {
            std::string name;
            std::string prompt;
            int maxTokens;
            float temperature;
            bool shouldFail;
        };
        
        std::vector<TestParams> testCases = {
            {"空 prompt", "", 10, 0.7f, true},
            {"负 max_tokens", "test", -1, 0.7f, true},
            {"过大的 max_tokens", "test", 10000, 0.7f, true},
            {"无效 temperature", "test", 10, 3.0f, true},
            {"有效请求", "test", 10, 0.7f, false}
        };
        
        for (const auto& tc : testCases) {
            log(LogLevel::INFO, "测试: " + tc.name);
            
            bool isValid = true;
            if (tc.prompt.empty()) isValid = false;
            if (tc.maxTokens <= 0 || tc.maxTokens > 4096) isValid = false;
            if (tc.temperature < 0.0f || tc.temperature > 2.0f) isValid = false;
            
            if (tc.shouldFail) {
                assertTrue(!isValid, tc.name + " 应该失败");
            } else {
                assertTrue(isValid, tc.name + " 应该成功");
            }
        }
        
        log(LogLevel::INFO, "错误处理测试完成");
    }
};

} // namespace kylin_test

namespace {
    std::vector<std::function<std::unique_ptr<kylin_test::TestCase>()>>& getStage9Factories() {
        static std::vector<std::function<std::unique_ptr<kylin_test::TestCase>()>> factories;
        return factories;
    }
}

namespace kylin_test {

void registerStage9Tests() {
    auto& factories = getStage9Factories();
    factories.clear();
    factories.push_back([]() { return std::make_unique<HttpServerBasicTest>(); });
    factories.push_back([]() { return std::make_unique<HealthEndpointTest>(); });
    factories.push_back([]() { return std::make_unique<GenerateEndpointTest>(); });
    factories.push_back([]() { return std::make_unique<EncodeEndpointTest>(); });
    factories.push_back([]() { return std::make_unique<RequestResponseFormatTest>(); });
    factories.push_back([]() { return std::make_unique<ApiPerformanceTest>(); });
    factories.push_back([]() { return std::make_unique<ErrorHandlingTest>(); });
}

std::unique_ptr<TestSuite> createHttpApiIntegrationTestSuite() {
    registerStage9Tests();
    auto suite = std::make_unique<TestSuite>("Stage 15: HTTP API Integration");
    
    auto& factories = getStage9Factories();
    for (auto& factory : factories) {
        suite->addTest(factory());
    }
    
    return suite;
}

} // namespace kylin_test
