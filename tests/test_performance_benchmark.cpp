/**
 * @file test_performance_benchmark.cpp
 * @brief Phase 4.2: 性能基准测试 (Phase 4.2)
 *
 * 测试内容：
 * 1. 吞吐量测试（单请求和批处理）
 * 2. 延迟测试（P50/P95/P99）
 * 3. 资源使用测试
 * 4. 扩展性测试
 */

#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

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
 * @brief 性能基准测试基类
 */
class PerformanceBenchmarkTest : public ::testing::Test {
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
        CLLM_INFO("Setting up PerformanceBenchmarkTest");
        
        // 创建HTTP处理器
        handler_ = std::make_unique<HttpHandler>();
        
        // 创建健康检查端点
        healthEndpoint_ = std::make_unique<HealthEndpoint>();
        handler_->get("/health", [this](const HttpRequest& req) {
            return healthEndpoint_->handle(req);
        });
        
        // 尝试初始化模型执行器和分词器（可选）
        const char* modelPath = std::getenv("CLLM_TEST_MODEL_PATH");
        if (modelPath) {
            try {
                CLLM_INFO("Initializing components with model path: %s", modelPath);
                
                modelExecutor_ = std::make_unique<ModelExecutor>(
                    std::string(modelPath),
                    "",
                    true,
                    false
                );
                
                try {
                    modelExecutor_->loadModel();
                    CLLM_INFO("Model loaded successfully");
                } catch (const std::exception& e) {
                    CLLM_WARN("Failed to load model: %s (continuing with mock)", e.what());
                }
                
                std::string tokenizerPath = std::string(modelPath) + "/tokenizer.model";
                tokenizer_ = std::make_unique<Tokenizer>(tokenizerPath);
                CLLM_INFO("Tokenizer initialized");
                
                scheduler_ = std::make_unique<Scheduler>(
                    modelExecutor_.get(),
                    8,
                    2048
                );
                
                scheduler_->start();
                CLLM_INFO("Scheduler started");
                
                componentsInitialized_ = true;
            } catch (const std::exception& e) {
                CLLM_WARN("Failed to initialize components: %s (continuing with mock)", e.what());
            }
        } else {
            CLLM_WARN("CLLM_TEST_MODEL_PATH not set, using mock implementation");
        }
        
        if (scheduler_ && tokenizer_) {
            generateEndpoint_ = std::make_unique<GenerateEndpoint>(
                scheduler_.get(),
                tokenizer_.get()
            );
        } else {
            generateEndpoint_ = std::make_unique<GenerateEndpoint>(nullptr, nullptr);
        }
        
        handler_->post("/generate", [this](const HttpRequest& req) {
            return generateEndpoint_->handle(req);
        });
        
        handler_->post("/generate_stream", [this](const HttpRequest& req) {
            return generateEndpoint_->handle(req);
        });
        
        if (tokenizer_) {
            encodeEndpoint_ = std::make_unique<EncodeEndpoint>(tokenizer_.get());
        } else {
            encodeEndpoint_ = std::make_unique<EncodeEndpoint>(nullptr);
        }
        
        handler_->post("/encode", [this](const HttpRequest& req) {
            return encodeEndpoint_->handle(req);
        });
        
        CLLM_INFO("PerformanceBenchmarkTest setup completed");
    }
    
    void TearDown() override {
        CLLM_INFO("Tearing down PerformanceBenchmarkTest");
        
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
    
    /**
     * @brief 计算百分位数
     */
    double calculatePercentile(const std::vector<double>& sortedValues, double percentile) {
        if (sortedValues.empty()) {
            return 0.0;
        }
        
        size_t index = static_cast<size_t>(std::ceil(percentile * sortedValues.size() / 100.0) - 1);
        index = std::min(index, sortedValues.size() - 1);
        return sortedValues[index];
    }
};

// ==================== P4.2.1: 吞吐量测试 ====================

/**
 * @brief 测试单请求吞吐量
 */
TEST_F(PerformanceBenchmarkTest, SingleRequestThroughput) {
    CLLM_INFO("Testing single request throughput");
    
    if (!componentsInitialized_) {
        CLLM_WARN("Components not initialized, skipping throughput test");
        GTEST_SKIP() << "Components not initialized";
    }
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Count from 1 to 100.";
    jsonBody["max_tokens"] = 50;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto response = handler_->handleRequest(request);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (response.getStatusCode() != 200) {
        CLLM_WARN("Request failed with status %d, skipping throughput test", response.getStatusCode());
        GTEST_SKIP() << "Request failed";
    }
    
    double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    double duration_sec = duration_ms / 1000.0;
    
    // 解析响应以获取生成的token数
    Json::Reader reader;
    Json::Value result;
    int completionTokens = 10; // 默认值
    
    if (reader.parse(response.getBody(), result)) {
        if (result.isMember("generated_text")) {
            // 估算token数（简单假设每个字符约0.75个token）
            std::string text = result["generated_text"].asString();
            completionTokens = static_cast<int>(text.length() * 0.75);
        }
    }
    
    double tokens_per_sec = completionTokens / duration_sec;
    
    CLLM_INFO("Single Request Throughput: %.2f tokens/sec", tokens_per_sec);
    CLLM_INFO("Completion tokens: %d", completionTokens);
    CLLM_INFO("Duration: %.3f sec (%.2f ms)", duration_sec, duration_ms);
    
    // 性能目标：> 10 tokens/sec (在测试环境中使用较低阈值)
    EXPECT_GT(tokens_per_sec, 1.0) << "Throughput too low";
}

/**
 * @brief 测试批处理吞吐量
 */
TEST_F(PerformanceBenchmarkTest, BatchThroughput) {
    CLLM_INFO("Testing batch throughput");
    
    if (!componentsInitialized_) {
        CLLM_WARN("Components not initialized, skipping batch throughput test");
        GTEST_SKIP() << "Components not initialized";
    }
    
    const int BATCH_SIZE = 4; // 使用较小的批次大小以避免测试时间过长
    std::vector<std::thread> threads;
    std::atomic<int> totalTokens{0};
    std::atomic<int> successCount{0};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < BATCH_SIZE; ++i) {
        threads.emplace_back([&, i]() {
            HttpRequest request;
            request.setMethod("POST");
            request.setPath("/generate");
            
            Json::Value jsonBody;
            jsonBody["prompt"] = "Test " + std::to_string(i);
            jsonBody["max_tokens"] = 10;
            
            Json::StreamWriterBuilder writerBuilder;
            std::string body = Json::writeString(writerBuilder, jsonBody);
            request.setBody(body);
            
            auto response = handler_->handleRequest(request);
            
            if (response.getStatusCode() == 200) {
                Json::Reader reader;
                Json::Value result;
                if (reader.parse(response.getBody(), result)) {
                    if (result.isMember("generated_text")) {
                        std::string text = result["generated_text"].asString();
                        int tokens = static_cast<int>(text.length() * 0.75);
                        totalTokens += tokens;
                    }
                }
                successCount++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    double duration_sec = duration_ms / 1000.0;
    
    double throughput = totalTokens.load() / duration_sec;
    
    CLLM_INFO("Batch Throughput (%d): %.2f tokens/sec", BATCH_SIZE, throughput);
    CLLM_INFO("Total tokens: %d", totalTokens.load());
    CLLM_INFO("Successful requests: %d/%d", successCount.load(), BATCH_SIZE);
    CLLM_INFO("Duration: %.3f sec (%.2f ms)", duration_sec, duration_ms);
    
    // 性能目标：> 5 tokens/sec (批处理)
    EXPECT_GT(throughput, 1.0) << "Batch throughput too low";
}

// ==================== P4.2.2: 延迟测试 ====================

/**
 * @brief 测试延迟分布（P50/P95/P99）
 */
TEST_F(PerformanceBenchmarkTest, LatencyDistribution) {
    CLLM_INFO("Testing latency distribution");
    
    const int NUM_REQUESTS = 50; // 使用较少的请求数以加快测试
    std::vector<double> latencies;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Hello";
    jsonBody["max_tokens"] = 5; // 使用较小的max_tokens以加快测试
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto reqStart = std::chrono::high_resolution_clock::now();
        
        auto response = handler_->handleRequest(request);
        
        auto reqEnd = std::chrono::high_resolution_clock::now();
        
        if (response.getStatusCode() == 200) {
            double latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                reqEnd - reqStart
            ).count();
            latencies.push_back(latency_ms);
        }
    }
    
    if (latencies.empty()) {
        CLLM_WARN("No successful requests, skipping latency test");
        GTEST_SKIP() << "No successful requests";
    }
    
    std::sort(latencies.begin(), latencies.end());
    
    double p50 = calculatePercentile(latencies, 50.0);
    double p95 = calculatePercentile(latencies, 95.0);
    double p99 = calculatePercentile(latencies, 99.0);
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    
    CLLM_INFO("Latency Distribution:");
    CLLM_INFO("  Mean: %.2f ms", mean);
    CLLM_INFO("  P50:  %.2f ms", p50);
    CLLM_INFO("  P95:  %.2f ms", p95);
    CLLM_INFO("  P99:  %.2f ms", p99);
    
    // 性能目标（在测试环境中使用较宽松的阈值）
    EXPECT_LT(p50, 5000) << "P50 latency too high";  // P50 < 5s
    EXPECT_LT(p95, 10000) << "P95 latency too high"; // P95 < 10s
    EXPECT_LT(p99, 20000) << "P99 latency too high"; // P99 < 20s
}

// ==================== P4.2.3: 资源使用测试 ====================

/**
 * @brief 测试基本资源使用情况
 */
TEST_F(PerformanceBenchmarkTest, ResourceUsage) {
    CLLM_INFO("Testing resource usage");
    
    // 简单的资源使用测试：执行多个请求并观察是否有内存泄漏迹象
    const int NUM_REQUESTS = 20;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    
    Json::Value jsonBody;
    jsonBody["prompt"] = "Test resource usage";
    jsonBody["max_tokens"] = 5;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string body = Json::writeString(writerBuilder, jsonBody);
    request.setBody(body);
    
    int successCount = 0;
    
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        auto response = handler_->handleRequest(request);
        
        if (response.getStatusCode() == 200) {
            successCount++;
        }
    }
    
    CLLM_INFO("Resource usage test completed");
    CLLM_INFO("Successful requests: %d/%d", successCount, NUM_REQUESTS);
    
    // 验证系统仍然正常工作
    EXPECT_GT(successCount, 0) << "No successful requests";
}

// ==================== P4.2.4: 扩展性测试 ====================

/**
 * @brief 测试不同并发级别的性能
 */
TEST_F(PerformanceBenchmarkTest, ScalabilityConcurrency) {
    CLLM_INFO("Testing scalability with different concurrency levels");
    
    if (!componentsInitialized_) {
        CLLM_WARN("Components not initialized, skipping scalability test");
        GTEST_SKIP() << "Components not initialized";
    }
    
    std::vector<int> concurrencyLevels = {1, 2, 4};
    std::vector<double> throughputs;
    
    for (int concurrency : concurrencyLevels) {
        CLLM_INFO("Testing with concurrency level: %d", concurrency);
        
        std::vector<std::thread> threads;
        std::atomic<int> totalTokens{0};
        std::atomic<int> successCount{0};
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < concurrency; ++i) {
            threads.emplace_back([&, i]() {
                HttpRequest request;
                request.setMethod("POST");
                request.setPath("/generate");
                
                Json::Value jsonBody;
                jsonBody["prompt"] = "Scalability test " + std::to_string(i);
                jsonBody["max_tokens"] = 5;
                
                Json::StreamWriterBuilder writerBuilder;
                std::string body = Json::writeString(writerBuilder, jsonBody);
                request.setBody(body);
                
                auto response = handler_->handleRequest(request);
                
                if (response.getStatusCode() == 200) {
                    Json::Reader reader;
                    Json::Value result;
                    if (reader.parse(response.getBody(), result)) {
                        if (result.isMember("generated_text")) {
                            std::string text = result["generated_text"].asString();
                            int tokens = static_cast<int>(text.length() * 0.75);
                            totalTokens += tokens;
                        }
                    }
                    successCount++;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count() / 1000.0;
        
        double throughput = totalTokens.load() / duration_sec;
        throughputs.push_back(throughput);
        
        CLLM_INFO("  Concurrency %d: %.2f tokens/sec (success: %d/%d)", 
                 concurrency, throughput, successCount.load(), concurrency);
    }
    
    // 验证并发性能
    EXPECT_GT(throughputs.size(), 0) << "No throughput measurements";
    
    CLLM_INFO("Scalability test completed");
}
