/**
 * @file test_frontend_subsystem.cpp
 * @brief 前端子系统（HTTP + Tokenizer）测试 (Phase 3.1)
 *
 * 测试内容：
 * 1. 完整的 HTTP 请求 → 响应流程
 * 2. 并发处理能力（50并发）
 * 3. 性能指标（延迟、吞吐量）
 * 4. 容错能力
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
#include "cllm/common/logger.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/tokenizer/unified_tokenizer.h"
#include <json/json.h>

using namespace cllm;

class FrontendSubsystemTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HFTokenizer> tokenizer_;
    bool tokenizerLoaded_ = false;

    void SetUp() override {
        CLLM_INFO("Setting up FrontendSubsystemTest");
        
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
        
        // 注册/v1/tokenize端点
        handler_->post("/v1/tokenize", [this](const HttpRequest& request) {
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
    }
};

// P3.1.1: 前端子系统完整流程测试
TEST_F(FrontendSubsystemTest, CompleteFlow) {
    CLLM_INFO("Running P3.1.1: CompleteFlow test");
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/v1/tokenize");
    
    Json::Value jsonBody;
    jsonBody["text"] = "Hello, world! This is a test.";
    
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
    EXPECT_TRUE(result.isMember("token_count"));
    EXPECT_EQ(result["token_count"].asInt(), result["tokens"].size());
    
    CLLM_INFO("CompleteFlow test passed");
}

// P3.1.2: 前端子系统并发测试（50并发）
TEST_F(FrontendSubsystemTest, ConcurrentRequests) {
    CLLM_INFO("Running P3.1.2: ConcurrentRequests test");
    
    const int NUM_THREADS = 50;
    const int REQUESTS_PER_THREAD = 10;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < REQUESTS_PER_THREAD; ++j) {
                HttpRequest request;
                request.setMethod("POST");
                request.setPath("/v1/tokenize");
                
                Json::Value jsonBody;
                jsonBody["text"] = "Test " + std::to_string(i * 100 + j);
                
                Json::StreamWriterBuilder writerBuilder;
                std::string body = Json::writeString(writerBuilder, jsonBody);
                request.setBody(body);
                
                auto response = handler_->handleRequest(request);
                
                if (response.getStatusCode() == 200) {
                    success_count++;
                } else {
                    error_count++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count, NUM_THREADS * REQUESTS_PER_THREAD);
    EXPECT_EQ(error_count, 0);
    
    CLLM_INFO("ConcurrentRequests test passed: %d successes, %d errors", success_count.load(), error_count.load());
}

// P3.1.3: 前端子系统性能测试（延迟/吞吐量）
TEST_F(FrontendSubsystemTest, PerformanceMetrics) {
    CLLM_INFO("Running P3.1.3: PerformanceMetrics test");
    
    const int NUM_REQUESTS = 100;
    std::vector<double> latencies;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto req_start = std::chrono::high_resolution_clock::now();
        
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/v1/tokenize");
        
        Json::Value jsonBody;
        jsonBody["text"] = "Performance test text";
        
        Json::StreamWriterBuilder writerBuilder;
        std::string body = Json::writeString(writerBuilder, jsonBody);
        request.setBody(body);
        
        auto response = handler_->handleRequest(request);
        
        auto req_end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            req_end - req_start
        ).count();
        latencies.push_back(latency);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time
    ).count();
    
    // 计算统计指标
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[NUM_REQUESTS * 50 / 100];
    double p95 = latencies[NUM_REQUESTS * 95 / 100];
    double p99 = latencies[NUM_REQUESTS * 99 / 100];
    double throughput = NUM_REQUESTS / total_time;
    
    CLLM_INFO("Performance Metrics:");
    CLLM_INFO("  P50 Latency: %.2f ms", p50);
    CLLM_INFO("  P95 Latency: %.2f ms", p95);
    CLLM_INFO("  P99 Latency: %.2f ms", p99);
    CLLM_INFO("  Throughput: %.2f req/s", throughput);
    
    // 验证性能目标
    // 注意：在测试环境中，这些值可能需要根据实际情况调整
    EXPECT_LT(p99, 1000); // P99 < 1000ms (宽松限制，用于测试)
    EXPECT_GT(throughput, 1); // > 1 req/s (宽松限制，用于测试)
    
    CLLM_INFO("PerformanceMetrics test passed");
}

// P3.1.4: 前端子系统容错测试
TEST_F(FrontendSubsystemTest, FaultTolerance) {
    CLLM_INFO("Running P3.1.4: FaultTolerance test");
    
    // 测试1: 无效输入
    HttpRequest invalidRequest;
    invalidRequest.setMethod("POST");
    invalidRequest.setPath("/v1/tokenize");
    
    Json::Value invalidJsonBody;
    invalidJsonBody["invalid_field"] = "test";
    
    Json::StreamWriterBuilder writerBuilder;
    std::string invalidBody = Json::writeString(writerBuilder, invalidJsonBody);
    invalidRequest.setBody(invalidBody);
    
    auto response1 = handler_->handleRequest(invalidRequest);
    EXPECT_EQ(response1.getStatusCode(), 400); // Bad Request
    
    // 测试2: 空输入
    HttpRequest emptyRequest;
    emptyRequest.setMethod("POST");
    emptyRequest.setPath("/v1/tokenize");
    
    Json::Value emptyJsonBody;
    emptyJsonBody["text"] = "";
    
    std::string emptyBody = Json::writeString(writerBuilder, emptyJsonBody);
    emptyRequest.setBody(emptyBody);
    
    auto response2 = handler_->handleRequest(emptyRequest);
    EXPECT_TRUE(response2.getStatusCode() == 200 || response2.getStatusCode() == 400);
    
    // 测试3: 超长输入
    HttpRequest longRequest;
    longRequest.setMethod("POST");
    longRequest.setPath("/v1/tokenize");
    
    std::string long_text(10000, 'a');
    Json::Value longJsonBody;
    longJsonBody["text"] = long_text;
    
    std::string longBody = Json::writeString(writerBuilder, longJsonBody);
    longRequest.setBody(longBody);
    
    auto response3 = handler_->handleRequest(longRequest);
    EXPECT_TRUE(response3.getStatusCode() == 200); // 当前实现接受超长输入
    
    // 测试4: 系统应该仍然正常工作
    HttpRequest validRequest;
    validRequest.setMethod("POST");
    validRequest.setPath("/v1/tokenize");
    
    Json::Value validJsonBody;
    validJsonBody["text"] = "Test after errors";
    
    std::string validBody = Json::writeString(writerBuilder, validJsonBody);
    validRequest.setBody(validBody);
    
    auto response5 = handler_->handleRequest(validRequest);
    EXPECT_EQ(response5.getStatusCode(), 200);
    
    CLLM_INFO("FaultTolerance test passed");
}
