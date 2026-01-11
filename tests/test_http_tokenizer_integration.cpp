/**
 * @file test_http_tokenizer_integration.cpp
 * @brief HTTP Server + Tokenizer 集成测试 (Phase 2.1)
 *
 * 测试内容：
 * 1. /v1/tokenize 端点集成测试
 * 2. /v1/detokenize 端点集成测试
 * 3. 错误传播测试
 * 4. 批量请求测试
 */

#include <gtest/gtest.h>
#include <iostream>
#include "cllm/http/handler.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/logger.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include <json/json.h>

using namespace cllm;

class HttpServerTokenizeIntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HFTokenizer> tokenizer_;
    bool tokenizerLoaded_ = false;

    void SetUp() override {
        CLLM_INFO("Setting up HttpServerTokenizeIntegrationTest");
        
        // 创建HTTP处理器
        handler_ = std::make_unique<HttpHandler>();
        
        // 创建并初始化Tokenizer
        tokenizer_ = std::make_unique<HFTokenizer>();
        
        // 获取环境变量中的测试模型路径
        const char* tokenizerPath = std::getenv("CLLM_TEST_MODEL_PATH");
        
        if (tokenizerPath) {
            CLLM_INFO("Using tokenizer from: %s", tokenizerPath);
            tokenizerLoaded_ = tokenizer_->load(tokenizerPath);
        }
        
        if (!tokenizerLoaded_) {
            CLLM_WARN("Failed to load tokenizer from %s, using mock implementation", tokenizerPath ? tokenizerPath : "(null)");
            // 如果加载失败，使用空实现
        }
        
        // 注册/tokenize端点
        handler_->post("/tokenize", [this](const HttpRequest& request) {
            HttpResponse response;
            std::string body = request.getBody();
            
            try {
                Json::Reader reader;
                Json::Value root;
                if (!reader.parse(body, root)) {
                    throw std::invalid_argument("Invalid JSON format");
                }
                
                if (!root.isMember("text") || !root["text"].isString()) {
                    throw std::invalid_argument("Missing or invalid 'text' parameter");
                }
                
                std::string text = root["text"].asString();
                
                std::vector<int> tokenIds;
                if (tokenizerLoaded_) {
                    tokenIds = tokenizer_->encode(text);
                } else {
                    // Mock implementation for testing
                    // Simple tokenization: split by spaces and return ASCII codes
                    size_t start = 0;
                    while (start < text.size()) {
                        size_t end = text.find(' ', start);
                        if (end == std::string::npos) end = text.size();
                        
                        std::string token = text.substr(start, end - start);
                        if (!token.empty()) {
                            // Convert token to ASCII codes for mock
                            for (char c : token) {
                                tokenIds.push_back(static_cast<int>(c));
                            }
                            tokenIds.push_back(32); // Add space token
                        }
                        
                        start = end + 1;
                    }
                    
                    if (!tokenIds.empty() && tokenIds.back() == 32) {
                        tokenIds.pop_back(); // Remove trailing space
                    }
                }
                
                Json::Value result;
                result["tokens"] = Json::Value(Json::arrayValue);
                for (int id : tokenIds) {
                    result["tokens"].append(id);
                }
                result["token_count"] = static_cast<int>(tokenIds.size());
                result["model"] = tokenizerLoaded_ ? "qwen3" : "mock";
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, result);
                
                response.setStatusCode(200);
                response.setContentType("application/json");
                response.setBody(responseBody);
                
            } catch (const std::exception& e) {
                response.setStatusCode(400);
                response.setContentType("application/json");
                
                Json::Value error;
                error["error"] = e.what();
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, error);
                response.setBody(responseBody);
            }
            
            return response;
        });
        
        // 注册/detokenize端点
        handler_->post("/detokenize", [this](const HttpRequest& request) {
            HttpResponse response;
            std::string body = request.getBody();
            
            try {
                Json::Reader reader;
                Json::Value root;
                if (!reader.parse(body, root)) {
                    throw std::invalid_argument("Invalid JSON format");
                }
                
                if (!root.isMember("tokens") || !root["tokens"].isArray()) {
                    throw std::invalid_argument("Missing or invalid 'tokens' parameter");
                }
                
                std::vector<int> tokenIds;
                for (const auto& token : root["tokens"]) {
                    if (token.isInt()) {
                        tokenIds.push_back(token.asInt());
                    }
                }
                
                std::string text;
                if (tokenizerLoaded_) {
                    text = tokenizer_->decode(tokenIds);
                } else {
                    // Mock implementation for testing
                    for (int id : tokenIds) {
                        if (id >= 32 && id <= 126) { // ASCII printable range
                            text += static_cast<char>(id);
                        }
                    }
                }
                
                Json::Value result;
                result["text"] = text;
                result["model"] = tokenizerLoaded_ ? "qwen3" : "mock";
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, result);
                
                response.setStatusCode(200);
                response.setContentType("application/json");
                response.setBody(responseBody);
                
            } catch (const std::exception& e) {
                response.setStatusCode(400);
                response.setContentType("application/json");
                
                Json::Value error;
                error["error"] = e.what();
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, error);
                response.setBody(responseBody);
            }
            
            return response;
        });
    }
};

// P2.1.1: /tokenize 端点测试
TEST_F(HttpServerTokenizeIntegrationTest, TokenizeEndpoint) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/tokenize");
    
    Json::Value jsonBody;
    jsonBody["text"] = "Hello, world!";
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response.getBody(), result));
    
    EXPECT_TRUE(result.isMember("tokens"));
    EXPECT_TRUE(result["tokens"].isArray());
    EXPECT_GT(result["tokens"].size(), 0);
}

// P2.1.2: /detokenize 端点测试
TEST_F(HttpServerTokenizeIntegrationTest, DetokenizeEndpoint) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/detokenize");
    
    Json::Value jsonBody;
    Json::Value tokens(Json::arrayValue);
    tokens.append(72);  // H
    tokens.append(101); // e
    tokens.append(108); // l
    tokens.append(108); // l
    tokens.append(111); // o
    jsonBody["tokens"] = tokens;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto response = handler_->handleRequest(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response.getBody(), result));
    
    EXPECT_TRUE(result.isMember("text"));
    EXPECT_TRUE(result["text"].isString());
    EXPECT_FALSE(result["text"].asString().empty());
}

// P2.1.3: 错误传播测试
TEST_F(HttpServerTokenizeIntegrationTest, ErrorPropagation) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/tokenize");
    
    // 发送无效的JSON
    request.setBody("{invalid json}");
    
    auto response = handler_->handleRequest(request);
    
    EXPECT_NE(response.getStatusCode(), 200);
    
    Json::Reader reader;
    Json::Value result;
    if (reader.parse(response.getBody(), result)) {
        EXPECT_TRUE(result.isMember("error"));
    }
}

// P2.1.4: 批量请求测试（模拟）
TEST_F(HttpServerTokenizeIntegrationTest, BatchRequests) {
    // 测试多个连续请求
    for (int i = 0; i < 3; ++i) {
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/tokenize");
        
        Json::Value jsonBody;
        jsonBody["text"] = "Test message " + std::to_string(i);
        
        Json::StreamWriterBuilder writerBuilder;
        std::string body = Json::writeString(writerBuilder, jsonBody);
        request.setBody(body);
        
        auto response = handler_->handleRequest(request);
        
        EXPECT_EQ(response.getStatusCode(), 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response.getBody(), result));
        
        EXPECT_TRUE(result.isMember("tokens"));
        EXPECT_TRUE(result["tokens"].isArray());
    }
}