#include <gtest/gtest.h>
#include <iostream>
#include "cllm/http/handler.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/logger.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include <json/json.h>

using namespace cllm;

class HttpServerTokenizeTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HFTokenizer> tokenizer_;
    bool tokenizerLoaded_ = false;

    void SetUp() override {
        CLLM_INFO("Setting up HttpServerTokenizeTest");
        
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
                Json::Value error;
                error["error"] = e.what();
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, error);
                
                response.setStatusCode(400);
                response.setContentType("application/json");
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
                for (const auto& tokenValue : root["tokens"]) {
                    if (!tokenValue.isInt()) {
                        throw std::invalid_argument("Invalid token ID format");
                    }
                    tokenIds.push_back(tokenValue.asInt());
                }
                
                std::string text;
                if (tokenizerLoaded_) {
                    text = tokenizer_->decode(tokenIds);
                } else {
                    // Mock implementation for testing
                    // Convert ASCII codes back to characters
                    for (int id : tokenIds) {
                        if (id >= 32 && id <= 126) {
                            text += static_cast<char>(id);
                        } else if (id == 32) {
                            text += ' ';
                        } else {
                            // Skip invalid ASCII codes
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
                Json::Value error;
                error["error"] = e.what();
                
                Json::StreamWriterBuilder writerBuilder;
                std::string responseBody = Json::writeString(writerBuilder, error);
                
                response.setStatusCode(400);
                response.setContentType("application/json");
                response.setBody(responseBody);
            }
            
            return response;
        });
    }

    void TearDown() override {
        CLLM_INFO("Tearing down HttpServerTokenizeTest");
        handler_.reset();
        tokenizer_.reset();
    }
};

TEST_F(HttpServerTokenizeTest, TokenizeEndpoint) {
    CLLM_INFO("=== Test: Tokenize Endpoint ===");
    
    // 创建测试请求
    Json::Value requestJson;
    requestJson["text"] = "Hello world!";
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/tokenize");
    req.setBody(requestBody);
    
    // 处理请求
    HttpResponse resp = handler_->handleRequest(req);
    
    // 验证响应
    EXPECT_EQ(resp.getStatusCode(), 200);
    EXPECT_EQ(resp.getContentType(), "application/json");
    
    // 解析响应体
    Json::Reader reader;
    Json::Value root;
    ASSERT_TRUE(reader.parse(resp.getBody(), root));
    
    EXPECT_TRUE(root.isMember("tokens"));
    EXPECT_TRUE(root["tokens"].isArray());
    EXPECT_GT(root["tokens"].size(), 0);
    
    CLLM_INFO("Tokenize response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, DetokenizeEndpoint) {
    CLLM_INFO("=== Test: Detokenize Endpoint ===");
    
    // 创建测试请求 - 使用简单的ASCII码作为token IDs
    Json::Value requestJson;
    Json::Value tokensArray(Json::arrayValue);
    tokensArray.append(72);  // H
    tokensArray.append(101); // e
    tokensArray.append(108); // l
    tokensArray.append(108); // l
    tokensArray.append(111); // o
    tokensArray.append(32);  // space
    tokensArray.append(119); // w
    tokensArray.append(111); // o
    tokensArray.append(114); // r
    tokensArray.append(108); // l
    tokensArray.append(100); // d
    tokensArray.append(33);  // !
    
    requestJson["tokens"] = tokensArray;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/detokenize");
    req.setBody(requestBody);
    
    // 处理请求
    HttpResponse resp = handler_->handleRequest(req);
    
    // 验证响应
    EXPECT_EQ(resp.getStatusCode(), 200);
    EXPECT_EQ(resp.getContentType(), "application/json");
    
    // 解析响应体
    Json::Reader reader;
    Json::Value root;
    ASSERT_TRUE(reader.parse(resp.getBody(), root));
    
    EXPECT_TRUE(root.isMember("text"));
    EXPECT_TRUE(root["text"].isString());
    
    // 如果使用mock实现，应该返回"Hello world!"
    if (!tokenizerLoaded_) {
        EXPECT_EQ(root["text"].asString(), "Hello world!");
    }
    
    CLLM_INFO("Detokenize response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, TokenizeInvalidJson) {
    CLLM_INFO("=== Test: Tokenize with Invalid JSON ===");
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/tokenize");
    req.setBody("{invalid json}");
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 400);
    EXPECT_TRUE(resp.getBody().find("error") != std::string::npos);
    
    CLLM_INFO("Tokenize invalid JSON response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, DetokenizeInvalidJson) {
    CLLM_INFO("=== Test: Detokenize with Invalid JSON ===");
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/detokenize");
    req.setBody("{invalid json}");
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 400);
    EXPECT_TRUE(resp.getBody().find("error") != std::string::npos);
    
    CLLM_INFO("Detokenize invalid JSON response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, TokenizeMissingText) {
    CLLM_INFO("=== Test: Tokenize with Missing Text Parameter ===");
    
    Json::Value requestJson;
    // 不包含"text"字段
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/tokenize");
    req.setBody(requestBody);
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 400);
    EXPECT_TRUE(resp.getBody().find("error") != std::string::npos);
    
    CLLM_INFO("Tokenize missing text response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, CompleteDataFlow) {
    CLLM_INFO("=== Test: Complete Data Flow (Tokenize → Detokenize) ===");
    
    // 测试文本
    std::string testText = "Hello world! This is a complete data flow test.";
    
    // Step 1: Tokenize the text
    Json::Value tokenizeReqJson;
    tokenizeReqJson["text"] = testText;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string tokenizeRequestBody = Json::writeString(writerBuilder, tokenizeReqJson);
    
    HttpRequest tokenizeReq;
    tokenizeReq.setMethod("POST");
    tokenizeReq.setPath("/tokenize");
    tokenizeReq.setBody(tokenizeRequestBody);
    
    HttpResponse tokenizeResp = handler_->handleRequest(tokenizeReq);
    EXPECT_EQ(tokenizeResp.getStatusCode(), 200);
    
    // Parse tokenize response to get token IDs
    Json::Reader reader;
    Json::Value tokenizeRoot;
    ASSERT_TRUE(reader.parse(tokenizeResp.getBody(), tokenizeRoot));
    ASSERT_TRUE(tokenizeRoot.isMember("tokens"));
    ASSERT_TRUE(tokenizeRoot["tokens"].isArray());
    
    std::vector<int> tokenIds;
    for (const auto& tokenValue : tokenizeRoot["tokens"]) {
        tokenIds.push_back(tokenValue.asInt());
    }
    
    // Step 2: Detokenize the token IDs back to text
    Json::Value detokenizeReqJson;
    Json::Value tokensArray(Json::arrayValue);
    for (int id : tokenIds) {
        tokensArray.append(id);
    }
    detokenizeReqJson["tokens"] = tokensArray;
    
    std::string detokenizeRequestBody = Json::writeString(writerBuilder, detokenizeReqJson);
    
    HttpRequest detokenizeReq;
    detokenizeReq.setMethod("POST");
    detokenizeReq.setPath("/detokenize");
    detokenizeReq.setBody(detokenizeRequestBody);
    
    HttpResponse detokenizeResp = handler_->handleRequest(detokenizeReq);
    EXPECT_EQ(detokenizeResp.getStatusCode(), 200);
    
    // Parse detokenize response
    Json::Value detokenizeRoot;
    ASSERT_TRUE(reader.parse(detokenizeResp.getBody(), detokenizeRoot));
    ASSERT_TRUE(detokenizeRoot.isMember("text"));
    ASSERT_TRUE(detokenizeRoot["text"].isString());
    
    std::string detokenizedText = detokenizeRoot["text"].asString();
    
    // Step 3: Verify the round-trip (tokenize → detokenize) preserves meaning
    // For mock implementation, we expect exact match
    if (!tokenizerLoaded_) {
        EXPECT_EQ(detokenizedText, testText);
    } else {
        // For real tokenizer, we expect similar content (tokenizer might add/remove whitespace)
        // Just verify the detokenized text is not empty and contains expected words
        EXPECT_FALSE(detokenizedText.empty());
        EXPECT_TRUE(detokenizedText.find("Hello") != std::string::npos || detokenizedText.find("hello") != std::string::npos);
        EXPECT_TRUE(detokenizedText.find("world") != std::string::npos);
    }
    
    CLLM_INFO("Original text: %s", testText.c_str());
    CLLM_INFO("Detokenized text: %s", detokenizedText.c_str());
}

TEST_F(HttpServerTokenizeTest, ErrorPropagation) {
    CLLM_INFO("=== Test: Error Propagation from Tokenizer ===");
    
    // Test with invalid token IDs for detokenization
    Json::Value detokenizeReqJson;
    Json::Value tokensArray(Json::arrayValue);
    // Add some potentially invalid token IDs
    tokensArray.append(-1);  // Invalid token ID
    tokensArray.append(9999999);  // Very large invalid token ID
    
    detokenizeReqJson["tokens"] = tokensArray;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string detokenizeRequestBody = Json::writeString(writerBuilder, detokenizeReqJson);
    
    HttpRequest detokenizeReq;
    detokenizeReq.setMethod("POST");
    detokenizeReq.setPath("/detokenize");
    detokenizeReq.setBody(detokenizeRequestBody);
    
    HttpResponse detokenizeResp = handler_->handleRequest(detokenizeReq);
    
    // For mock implementation, this will succeed since we don't validate token IDs
    // For real tokenizer, this should return an error
    if (tokenizerLoaded_) {
        EXPECT_EQ(detokenizeResp.getStatusCode(), 400);
        EXPECT_TRUE(detokenizeResp.getBody().find("error") != std::string::npos);
    } else {
        // Mock implementation will process any token IDs
        EXPECT_EQ(detokenizeResp.getStatusCode(), 200);
    }
    
    CLLM_INFO("Detokenize error propagation response: %s", detokenizeResp.getBody().c_str());
}

TEST_F(HttpServerTokenizeTest, BatchRequests) {
    CLLM_INFO("=== Test: Batch Tokenize Requests ===");
    
    // Test with multiple requests to simulate batch processing
    std::vector<std::string> testTexts = {
        "Hello world!",
        "This is a batch request test.",
        "HTTP Tokenizer integration is working.",
        "Batch processing is efficient.",
        "Testing multiple requests in sequence."
    };
    
    Json::StreamWriterBuilder writerBuilder;
    
    for (size_t i = 0; i < testTexts.size(); ++i) {
        const std::string& text = testTexts[i];
        
        // Create tokenize request
        Json::Value tokenizeReqJson;
        tokenizeReqJson["text"] = text;
        
        std::string tokenizeRequestBody = Json::writeString(writerBuilder, tokenizeReqJson);
        
        HttpRequest tokenizeReq;
        tokenizeReq.setMethod("POST");
        tokenizeReq.setPath("/tokenize");
        tokenizeReq.setBody(tokenizeRequestBody);
        
        // Process request
        HttpResponse tokenizeResp = handler_->handleRequest(tokenizeReq);
        
        // Verify response
        EXPECT_EQ(tokenizeResp.getStatusCode(), 200) << "Batch request " << i << " failed";
        
        // Parse response
        Json::Reader reader;
        Json::Value root;
        ASSERT_TRUE(reader.parse(tokenizeResp.getBody(), root)) << "Batch request " << i << " invalid JSON response";
        
        EXPECT_TRUE(root.isMember("tokens"));
        EXPECT_TRUE(root["tokens"].isArray());
        EXPECT_GT(root["tokens"].size(), 0);
        
        CLLM_INFO("Batch request %zu completed successfully", i);
    }
    
    CLLM_INFO("All batch requests completed successfully");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
