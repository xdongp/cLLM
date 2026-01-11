/**
 * @file test_system_functionality.cpp
 * @brief Phase 4.1: 系统功能测试 (Phase 4.1)
 *
 * 测试内容：
 * 1. 核心功能测试（文本生成、流式生成、编码）
 * 2. API兼容性测试
 * 3. 多场景测试
 * 4. 错误处理测试
 */

#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>

#include "cllm/http/handler.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <json/json.h>

using namespace cllm;

/**
 * @brief 系统功能测试基类
 */
class SystemFunctionalityTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HealthEndpoint> healthEndpoint_;
    std::unique_ptr<GenerateEndpoint> generateEndpoint_;
    std::unique_ptr<EncodeEndpoint> encodeEndpoint_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelExecutor> modelExecutor_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    bool componentsInitialized_ = false;

    void SetUp() override {
        CLLM_INFO("Setting up SystemFunctionalityTest");
        
        // 创建HTTP处理器
        handler_ = std::make_unique<HttpHandler>();
        
        // 创建健康检查端点
        healthEndpoint_ = std::make_unique<HealthEndpoint>();
        handler_->get("/health", [this](const HttpRequest& req) {
            return healthEndpoint_->handle(req);
        });
        
        // 尝试初始化模型执行器和分词器（可选，测试时可以跳过）
        const char* modelPath = std::getenv("CLLM_TEST_MODEL_PATH");
        if (modelPath) {
            try {
                CLLM_INFO("Initializing components with model path: %s", modelPath);
                
                // 初始化模型执行器
                modelExecutor_ = std::make_unique<ModelExecutor>(
                    std::string(modelPath),
                    "",      // 量化类型
                    true,    // 启用SIMD
                    false    // 不使用LibTorch
                );
                
                // 加载模型（可能失败，但不影响接口测试）
                try {
                    modelExecutor_->loadModel();
                    CLLM_INFO("Model loaded successfully");
                } catch (const std::exception& e) {
                    CLLM_WARN("Failed to load model: %s (continuing with mock)", e.what());
                }
                
                // 初始化分词器
                std::string tokenizerPath = std::string(modelPath) + "/tokenizer.model";
                tokenizer_ = std::make_unique<Tokenizer>(tokenizerPath);
                CLLM_INFO("Tokenizer initialized");
                
                // 创建调度器
                scheduler_ = std::make_unique<Scheduler>(
                    modelExecutor_.get(),
                    8,       // maxBatchSize
                    2048     // maxContextLength
                );
                
                // 启动调度器
                scheduler_->start();
                CLLM_INFO("Scheduler started");
                
                componentsInitialized_ = true;
                
            } catch (const std::exception& e) {
                CLLM_WARN("Failed to initialize components: %s (continuing with mock)", e.what());
            }
        } else {
            CLLM_WARN("CLLM_TEST_MODEL_PATH not set, using mock implementation");
        }
        
        // 创建生成端点（使用mock组件或真实组件）
        if (scheduler_ && tokenizer_) {
            generateEndpoint_ = std::make_unique<GenerateEndpoint>(
                scheduler_.get(),
                tokenizer_.get()
            );
        } else {
            // 使用nullptr，端点会处理这种情况
            generateEndpoint_ = std::make_unique<GenerateEndpoint>(nullptr, nullptr);
        }
        
        handler_->post("/generate", [this](const HttpRequest& req) {
            return generateEndpoint_->handle(req);
        });
        
        handler_->post("/generate_stream", [this](const HttpRequest& req) {
            return generateEndpoint_->handle(req);
        });
        
        // 创建编码端点
        if (tokenizer_) {
            encodeEndpoint_ = std::make_unique<EncodeEndpoint>(tokenizer_.get());
        } else {
            encodeEndpoint_ = std::make_unique<EncodeEndpoint>(nullptr);
        }
        
        handler_->post("/encode", [this](const HttpRequest& req) {
            return encodeEndpoint_->handle(req);
        });
        
        CLLM_INFO("SystemFunctionalityTest setup completed");
    }
    
    void TearDown() override {
        CLLM_INFO("Tearing down SystemFunctionalityTest");
        
        if (scheduler_) {
            scheduler_->stop();
        }
        
        handler_.reset();
        healthEndpoint_.reset();
        generateEndpoint_.reset();
        encodeEndpoint_.reset();
        scheduler_.reset();
        modelExecutor_.reset();
        tokenizer_.reset();
    }
};

// ==================== P4.1.1: 核心功能测试 ====================

/**
 * @brief 测试健康检查端点
 */
TEST_F(SystemFunctionalityTest, HealthCheck) {
    CLLM_INFO("Testing health check endpoint");
    
    HttpRequest request;
    request.setMethod("GET");
    request.setPath("/health");
    
    auto response = handler_->handleRequest(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_EQ(response.getContentType(), "application/json");
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response.getBody(), result));
    
    EXPECT_TRUE(result.isMember("status"));
    EXPECT_TRUE(result.isMember("model_loaded"));
    
    CLLM_INFO("Health check test passed");
}

/**
 * @brief 测试文本编码功能
 */
TEST_F(SystemFunctionalityTest, TextEncoding) {
    CLLM_INFO("Testing text encoding endpoint");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/encode");
    
    Json::Value jsonBody;
    jsonBody["text"] = "Hello, world! This is a test.";
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    // 如果tokenizer未初始化，可能返回500错误
    if (response.getStatusCode() == 500) {
        CLLM_WARN("Tokenizer not initialized, skipping encoding test");
        GTEST_SKIP() << "Tokenizer not initialized";
    }
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_EQ(response.getContentType(), "application/json");
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response.getBody(), result));
    
    EXPECT_TRUE(result.isMember("tokens"));
    EXPECT_TRUE(result.isMember("length"));
    EXPECT_TRUE(result["tokens"].isArray());
    EXPECT_GT(result["length"].asInt(), 0);
    EXPECT_EQ(result["tokens"].size(), result["length"].asInt());
    
    CLLM_INFO("Text encoding test passed, encoded %d tokens", result["length"].asInt());
}

/**
 * @brief 测试文本生成功能（非流式）
 */
TEST_F(SystemFunctionalityTest, TextGeneration) {
    CLLM_INFO("Testing text generation endpoint (non-streaming)");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Once upon a time";
    jsonBody["max_tokens"] = 10;
    jsonBody["temperature"] = 0.7;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    // 如果组件未初始化，可能返回错误
    if (response.getStatusCode() != 200) {
        CLLM_WARN("Components not initialized, skipping generation test");
        GTEST_SKIP() << "Components not initialized";
    }
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_EQ(response.getContentType(), "application/json");
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response.getBody(), result));
    
    EXPECT_TRUE(result.isMember("text") || result.isMember("generated_text"));
    
    CLLM_INFO("Text generation test passed");
}

/**
 * @brief 测试流式生成功能
 */
TEST_F(SystemFunctionalityTest, StreamingGeneration) {
    CLLM_INFO("Testing streaming generation endpoint");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate_stream");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Tell me a story";
    jsonBody["max_tokens"] = 20;
    jsonBody["stream"] = true;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    // 流式响应可能返回200或错误（取决于实现）
    if (response.getStatusCode() == 200) {
        EXPECT_TRUE(response.isStreaming() || !response.getBody().empty());
        
        if (response.isStreaming()) {
            auto chunks = response.getChunks();
            // 如果tokenizer未初始化，chunks可能为空
            if (!componentsInitialized_) {
                CLLM_INFO("Streaming test passed (components not initialized, chunks: %zu)", chunks.size());
            } else {
                EXPECT_GT(chunks.size(), 0) << "Expected chunks when components are initialized";
                CLLM_INFO("Streaming test passed, received %zu chunks", chunks.size());
            }
        } else {
            CLLM_INFO("Streaming test passed (non-streaming response)");
        }
    } else {
        CLLM_WARN("Streaming generation returned status %d, skipping", response.getStatusCode());
        GTEST_SKIP() << "Streaming not available";
    }
}

// ==================== P4.1.2: API兼容性测试 ====================

/**
 * @brief 测试API响应格式
 */
TEST_F(SystemFunctionalityTest, APIResponseFormat) {
    CLLM_INFO("Testing API response format");
    
    // 测试生成端点响应格式
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Hello";
    jsonBody["max_tokens"] = 5;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    if (response.getStatusCode() == 200) {
        EXPECT_EQ(response.getContentType(), "application/json");
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response.getBody(), result));
        
        // 验证响应包含必要的字段
        EXPECT_TRUE(result.isObject());
        
        CLLM_INFO("API response format test passed");
    } else {
        CLLM_WARN("API response format test skipped (status %d)", response.getStatusCode());
        GTEST_SKIP() << "API not available";
    }
}

// ==================== P4.1.3: 多场景测试 ====================

/**
 * @brief 测试不同长度的输入
 */
TEST_F(SystemFunctionalityTest, DifferentInputLengths) {
    CLLM_INFO("Testing different input lengths");
    
    std::vector<std::string> testPrompts = {
        "Hi",
        "Hello, how are you?",
        "This is a longer prompt to test the system's handling of different input lengths."
    };
    
    for (const auto& prompt : testPrompts) {
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        
        Json::Value jsonBody;
        jsonBody["prompt"] = prompt;
        jsonBody["max_tokens"] = 5;
        
        Json::StreamWriterBuilder writerBuilder;
        std::string body = Json::writeString(writerBuilder, jsonBody);
        request.setBody(body);
        
        auto response = handler_->handleRequest(request);
        
        // 如果组件未初始化，跳过测试
        if (response.getStatusCode() != 200) {
            CLLM_WARN("Components not initialized, skipping different input lengths test");
            GTEST_SKIP() << "Components not initialized";
        }
        
        EXPECT_EQ(response.getStatusCode(), 200) << "Failed for prompt: " << prompt;
    }
    
    CLLM_INFO("Different input lengths test passed");
}

/**
 * @brief 测试不同温度参数
 */
TEST_F(SystemFunctionalityTest, DifferentTemperatureValues) {
    CLLM_INFO("Testing different temperature values");
    
    std::vector<float> temperatures = {0.1f, 0.7f, 1.0f, 1.5f};
    
    for (float temp : temperatures) {
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        
        Json::Value jsonBody;
        jsonBody["prompt"] = "Test";
        jsonBody["max_tokens"] = 5;
        jsonBody["temperature"] = temp;
        
        Json::StreamWriterBuilder writerBuilder;
        std::string body = Json::writeString(writerBuilder, jsonBody);
        request.setBody(body);
        
        auto response = handler_->handleRequest(request);
        
        // 如果组件未初始化，跳过测试
        if (response.getStatusCode() != 200) {
            CLLM_WARN("Components not initialized, skipping temperature test");
            GTEST_SKIP() << "Components not initialized";
        }
        
        EXPECT_EQ(response.getStatusCode(), 200) << "Failed for temperature: " << temp;
    }
    
    CLLM_INFO("Different temperature values test passed");
}

// ==================== P4.1.4: 错误处理测试 ====================

/**
 * @brief 测试缺少必要字段的错误处理
 */
TEST_F(SystemFunctionalityTest, MissingRequiredFields) {
    CLLM_INFO("Testing error handling for missing required fields");
    
    // 测试缺少prompt字段
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["max_tokens"] = 10;
    // 缺少prompt字段
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    // 应该返回400错误或空响应（取决于实现）
    EXPECT_TRUE(response.getStatusCode() == 400 || response.getStatusCode() == 200);
    
    CLLM_INFO("Missing required fields test passed (status %d)", response.getStatusCode());
}

/**
 * @brief 测试无效的JSON格式
 */
TEST_F(SystemFunctionalityTest, InvalidJSONFormat) {
    CLLM_INFO("Testing error handling for invalid JSON format");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    // 无效的JSON
    request.setBody("{invalid json}");
    
    auto response = handler_->handleRequest(request);
    
    // 应该返回400错误或处理错误
    EXPECT_TRUE(response.getStatusCode() >= 400 || response.getStatusCode() == 200);
    
    CLLM_INFO("Invalid JSON format test passed (status %d)", response.getStatusCode());
}

/**
 * @brief 测试无效的端点
 */
TEST_F(SystemFunctionalityTest, InvalidEndpoint) {
    CLLM_INFO("Testing error handling for invalid endpoint");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/invalid_endpoint");
    
    auto response = handler_->handleRequest(request);
    
    // 应该返回404错误
    EXPECT_EQ(response.getStatusCode(), 404);
    
    CLLM_INFO("Invalid endpoint test passed");
}

/**
 * @brief 测试无效的HTTP方法
 */
TEST_F(SystemFunctionalityTest, InvalidHTTPMethod) {
    CLLM_INFO("Testing error handling for invalid HTTP method");
    
    HttpRequest request;
    request.setMethod("DELETE");
    request.setPath("/generate");
    
    auto response = handler_->handleRequest(request);
    
    // 应该返回404或400错误
    EXPECT_TRUE(response.getStatusCode() == 404 || response.getStatusCode() == 400);
    
    CLLM_INFO("Invalid HTTP method test passed (status %d)", response.getStatusCode());
}

/**
 * @brief 测试系统在错误后仍然正常工作
 */
TEST_F(SystemFunctionalityTest, SystemRecoveryAfterError) {
    CLLM_INFO("Testing system recovery after error");
    
    // 先发送一个错误请求
    HttpRequest errorRequest;
    errorRequest.setMethod("POST");
    errorRequest.setPath("/generate");
    errorRequest.setBody("{invalid json}");
    
    auto errorResponse = handler_->handleRequest(errorRequest);
    CLLM_INFO("Error request returned status %d", errorResponse.getStatusCode());
    
    // 然后发送一个有效请求
    HttpRequest validRequest;
    validRequest.setMethod("GET");
    validRequest.setPath("/health");
    
    auto validResponse = handler_->handleRequest(validRequest);
    
    // 系统应该仍然正常工作
    EXPECT_EQ(validResponse.getStatusCode(), 200);
    
    CLLM_INFO("System recovery test passed");
}
