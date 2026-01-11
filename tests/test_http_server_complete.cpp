#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <future>
#include <atomic>

#include "cllm/http/handler.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/logger.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include <json/json.h>

using namespace cllm;

class HttpServerCompleteTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HFTokenizer> tokenizer_;
    bool tokenizerLoaded_ = false;

    void SetUp() override {
        CLLM_INFO("Setting up HttpServerCompleteTest");
        
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
            CLLM_WARN("Failed to load tokenizer, using mock implementation");
        }
        
        // 注册所有必要的端点
        setupEndpoints();
    }

    void TearDown() override {
        CLLM_INFO("Tearing down HttpServerCompleteTest");
        
        // 清理资源
        handler_.reset();
        tokenizer_.reset();
    }

private:
    void setupEndpoints() {
        // 注册健康检查端点
        handler_->get("/health", [](const HttpRequest& request) {
            HttpResponse response;
            response.setStatusCode(200);
            response.setBody(R"({"status":"healthy","model_loaded":true})");
            return response;
        });
        
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
                    size_t start = 0;
                    while (start < text.size()) {
                        size_t end = text.find(' ', start);
                        if (end == std::string::npos) end = text.size();
                        
                        std::string token = text.substr(start, end - start);
                        if (!token.empty()) {
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
                    for (int id : tokenIds) {
                        if (id >= 32 && id <= 126) {
                            text += static_cast<char>(id);
                        } else if (id == 32) {
                            text += ' ';
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
        
        // 注册/complete端点（用于完整请求流程测试）
        handler_->post("/complete", [this](const HttpRequest& request) {
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
                int max_new_tokens = root.isMember("max_new_tokens") ? root["max_new_tokens"].asInt() : 20;
                
                // 模拟文本补全：在输入文本后添加"... completed by server"并截断到max_new_tokens
                std::string completedText = text + "... completed by server";
                if (completedText.size() > static_cast<size_t>(max_new_tokens)) {
                    completedText = completedText.substr(0, max_new_tokens);
                }
                
                Json::Value result;
                result["text"] = completedText;
                result["generated_tokens"] = max_new_tokens;
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
};

// Phase 3: Complete System Integration Tests
TEST_F(HttpServerCompleteTest, CompleteRequestFlow) {
    CLLM_INFO("=== Test: Complete Request Flow ===");
    
    // Test text
    std::string testText = "Hello, this is a complete flow test";
    
    // Step 1: Tokenize
    Json::Value tokenizeReqJson;
    tokenizeReqJson["text"] = testText;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string tokenizeBody = Json::writeString(writerBuilder, tokenizeReqJson);
    
    HttpRequest tokenizeReq;
    tokenizeReq.setMethod("POST");
    tokenizeReq.setPath("/tokenize");
    tokenizeReq.setBody(tokenizeBody);
    
    HttpResponse tokenizeResp = handler_->handleRequest(tokenizeReq);
    EXPECT_EQ(tokenizeResp.getStatusCode(), 200);
    
    // Step 2: Parse tokens
    Json::Reader reader;
    Json::Value tokenizeRoot;
    ASSERT_TRUE(reader.parse(tokenizeResp.getBody(), tokenizeRoot));
    ASSERT_TRUE(tokenizeRoot.isMember("tokens"));
    ASSERT_TRUE(tokenizeRoot["tokens"].isArray());
    
    std::vector<int> tokenIds;
    for (const auto& tokenValue : tokenizeRoot["tokens"]) {
        tokenIds.push_back(tokenValue.asInt());
    }
    
    // Step 3: Detokenize
    Json::Value detokenizeReqJson;
    Json::Value tokensArray(Json::arrayValue);
    for (int id : tokenIds) {
        tokensArray.append(id);
    }
    detokenizeReqJson["tokens"] = tokensArray;
    
    std::string detokenizeBody = Json::writeString(writerBuilder, detokenizeReqJson);
    
    HttpRequest detokenizeReq;
    detokenizeReq.setMethod("POST");
    detokenizeReq.setPath("/detokenize");
    detokenizeReq.setBody(detokenizeBody);
    
    HttpResponse detokenizeResp = handler_->handleRequest(detokenizeReq);
    EXPECT_EQ(detokenizeResp.getStatusCode(), 200);
    
    // Step 4: Complete text generation
    Json::Value completeReqJson;
    completeReqJson["text"] = testText;
    completeReqJson["max_new_tokens"] = 30;
    
    std::string completeBody = Json::writeString(writerBuilder, completeReqJson);
    
    HttpRequest completeReq;
    completeReq.setMethod("POST");
    completeReq.setPath("/complete");
    completeReq.setBody(completeBody);
    
    HttpResponse completeResp = handler_->handleRequest(completeReq);
    EXPECT_EQ(completeResp.getStatusCode(), 200);
    
    CLLM_INFO("Complete flow test passed successfully");
}

TEST_F(HttpServerCompleteTest, ConcurrencyStability) {
    CLLM_INFO("=== Test: Concurrency Stability ===");
    
    const int numThreads = 10;
    const int requestsPerThread = 5;
    std::atomic<int> successCount(0);
    
    // Function to make requests in a thread
    auto makeRequests = [this, &successCount](int threadId) {
        Json::StreamWriterBuilder writerBuilder;
        
        for (int i = 0; i < requestsPerThread; ++i) {
            // Create tokenize request
            Json::Value reqJson;
            reqJson["text"] = "Concurrent request test " + std::to_string(threadId) + ":" + std::to_string(i);
            
            std::string requestBody = Json::writeString(writerBuilder, reqJson);
            
            HttpRequest req;
            req.setMethod("POST");
            req.setPath("/tokenize");
            req.setBody(requestBody);
            
            // Handle request
            HttpResponse resp = handler_->handleRequest(req);
            
            if (resp.getStatusCode() == 200) {
                successCount++;
            }
            
            // Small delay to simulate real-world conditions
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    
    // Create and start threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(makeRequests, i);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Verify all requests succeeded
    int totalRequests = numThreads * requestsPerThread;
    EXPECT_EQ(successCount.load(), totalRequests) << "Expected " << totalRequests << " successful requests, got " << successCount.load();
    
    CLLM_INFO("Concurrency test completed: %d/%d requests successful", successCount.load(), totalRequests);
}

TEST_F(HttpServerCompleteTest, PerformanceMeasurement) {
    CLLM_INFO("=== Test: Performance Measurement ===");
    
    const int numRequests = 100;
    std::vector<double> responseTimes;
    
    Json::StreamWriterBuilder writerBuilder;
    Json::Value reqJson;
    reqJson["text"] = "Performance measurement test";
    std::string requestBody = Json::writeString(writerBuilder, reqJson);
    
    for (int i = 0; i < numRequests; ++i) {
        HttpRequest req;
        req.setMethod("POST");
        req.setPath("/tokenize");
        req.setBody(requestBody);
        
        // Measure response time
        auto startTime = std::chrono::high_resolution_clock::now();
        HttpResponse resp = handler_->handleRequest(req);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        responseTimes.push_back(duration.count() / 1000.0); // Convert to milliseconds
        
        EXPECT_EQ(resp.getStatusCode(), 200);
    }
    
    // Calculate statistics
    double total = 0.0;
    double min = responseTimes[0];
    double max = responseTimes[0];
    
    for (double time : responseTimes) {
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }
    
    double average = total / responseTimes.size();
    
    CLLM_INFO("Performance Results:");
    CLLM_INFO("  Requests: %d", numRequests);
    CLLM_INFO("  Average Response Time: %.2f ms", average);
    CLLM_INFO("  Min Response Time: %.2f ms", min);
    CLLM_INFO("  Max Response Time: %.2f ms", max);
    
    // Performance threshold - should be under 10ms per request
    EXPECT_LT(average, 10.0) << "Average response time exceeded threshold of 10ms";
}

TEST_F(HttpServerCompleteTest, ResourceMonitoring) {
    CLLM_INFO("=== Test: Resource Monitoring ===");
    
    // This test monitors memory usage by tracking object sizes and request counts
    const int numRequests = 500;
    
    Json::StreamWriterBuilder writerBuilder;
    Json::Value reqJson;
    reqJson["text"] = "Resource monitoring test for memory usage";
    std::string requestBody = Json::writeString(writerBuilder, reqJson);
    
    // Track the number of requests handled to ensure no memory leaks in request/response processing
    int successfulRequests = 0;
    
    for (int i = 0; i < numRequests; ++i) {
        HttpRequest req;
        req.setMethod("POST");
        req.setPath("/tokenize");
        req.setBody(requestBody);
        
        HttpResponse resp = handler_->handleRequest(req);
        
        if (resp.getStatusCode() == 200) {
            successfulRequests++;
        }
        
        // For a real resource monitoring test, we would use platform-specific APIs to check memory usage
        // Here we're just verifying that the server can handle many requests without crashing
    }
    
    EXPECT_EQ(successfulRequests, numRequests) << "Expected " << numRequests << " successful requests, got " << successfulRequests;
    
    CLLM_INFO("Resource monitoring test completed: %d requests processed successfully", successfulRequests);
}

TEST_F(HttpServerCompleteTest, HealthCheck) {
    CLLM_INFO("=== Test: Health Check ===");
    
    HttpRequest req;
    req.setMethod("GET");
    req.setPath("/health");
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 200);
    
    Json::Reader reader;
    Json::Value root;
    ASSERT_TRUE(reader.parse(resp.getBody(), root));
    
    EXPECT_TRUE(root.isMember("status"));
    EXPECT_EQ(root["status"].asString(), "healthy");
    
    CLLM_INFO("Health check passed");
}

TEST_F(HttpServerCompleteTest, EndpointErrorHandling) {
    CLLM_INFO("=== Test: Endpoint Error Handling ===");
    
    // Test 1: Invalid JSON to /tokenize
    HttpRequest invalidJsonReq;
    invalidJsonReq.setMethod("POST");
    invalidJsonReq.setPath("/tokenize");
    invalidJsonReq.setBody("{invalid json}");
    
    HttpResponse invalidJsonResp = handler_->handleRequest(invalidJsonReq);
    EXPECT_EQ(invalidJsonResp.getStatusCode(), 400);
    
    // Test 2: Missing parameter to /detokenize
    Json::Value missingParamReqJson;
    // Missing "tokens" parameter
    
    Json::StreamWriterBuilder writerBuilder;
    std::string missingParamBody = Json::writeString(writerBuilder, missingParamReqJson);
    
    HttpRequest missingParamReq;
    missingParamReq.setMethod("POST");
    missingParamReq.setPath("/detokenize");
    missingParamReq.setBody(missingParamBody);
    
    HttpResponse missingParamResp = handler_->handleRequest(missingParamReq);
    EXPECT_EQ(missingParamResp.getStatusCode(), 400);
    
    // Test 3: Invalid token format to /detokenize
    Json::Value invalidTokenReqJson;
    Json::Value invalidTokensArray(Json::arrayValue);
    invalidTokensArray.append("not an integer"); // Invalid token type
    invalidTokenReqJson["tokens"] = invalidTokensArray;
    
    std::string invalidTokenBody = Json::writeString(writerBuilder, invalidTokenReqJson);
    
    HttpRequest invalidTokenReq;
    invalidTokenReq.setMethod("POST");
    invalidTokenReq.setPath("/detokenize");
    invalidTokenReq.setBody(invalidTokenBody);
    
    HttpResponse invalidTokenResp = handler_->handleRequest(invalidTokenReq);
    EXPECT_EQ(invalidTokenResp.getStatusCode(), 400);
    
    CLLM_INFO("Endpoint error handling tests passed");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}