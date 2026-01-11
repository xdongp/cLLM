#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <future>
#include <atomic>
#include <sys/resource.h>
#include "cllm/http/server.h"
#include "cllm/http/handler.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/logger.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include <jsoncpp/json/json.h>

using namespace cllm;

class HttpServerResourcesTest : public ::testing::Test {
protected:
    std::unique_ptr<HttpHandler> handler_;
    std::unique_ptr<HFTokenizer> tokenizer_;
    std::atomic<bool> running_;

    void SetUp() override {
        CLLM_INFO("Setting up HttpServerResourcesTest");
        
        // 创建HTTP处理器
        handler_ = std::make_unique<HttpHandler>();
        
        // 创建并初始化Tokenizer
        tokenizer_ = std::make_unique<HFTokenizer>();
        
        // 获取环境变量中的测试模型路径
        const char* tokenizerPath = std::getenv("CLLM_TEST_MODEL_PATH");
        bool tokenizerLoaded = false;
        
        if (tokenizerPath) {
            CLLM_INFO("Using tokenizer from: %s", tokenizerPath);
            tokenizerLoaded = tokenizer_->load(tokenizerPath);
        }
        
        if (!tokenizerLoaded) {
            CLLM_WARNING("Failed to load tokenizer, using mock implementation");
        }
        
        // 注册端点
        setupEndpoints();
        
        running_ = true;
    }

    void TearDown() override {
        CLLM_INFO("Tearing down HttpServerResourcesTest");
        running_ = false;
        handler_.reset();
        tokenizer_.reset();
    }

private:
    void setupEndpoints() {
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
                if (tokenizer_->isLoaded()) {
                    tokenIds = tokenizer_->encode(text);
                } else {
                    // Mock tokenization
                    for (char c : text) {
                        tokenIds.push_back(static_cast<int>(c));
                    }
                }
                
                Json::Value result;
                result["tokens"] = Json::Value(Json::arrayValue);
                for (int id : tokenIds) {
                    result["tokens"].append(id);
                }
                result["token_count"] = static_cast<int>(tokenIds.size());
                result["model"] = "test-model";
                
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
                if (tokenizer_->isLoaded()) {
                    text = tokenizer_->decode(tokenIds);
                } else {
                    // Mock detokenization
                    for (int id : tokenIds) {
                        if (id >= 32 && id <= 126) {
                            text += static_cast<char>(id);
                        }
                    }
                }
                
                Json::Value result;
                result["text"] = text;
                result["model"] = "test-model";
                
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

    // 获取当前内存使用情况（以KB为单位）
    long getCurrentMemoryUsage() {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            return usage.ru_maxrss;
        }
        return -1;
    }

    // 获取CPU时间（以秒为单位）
    double getCurrentCPUTime() {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0 +
                   usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0;
        }
        return -1;
    }

public:
    // 执行内存泄漏测试的辅助函数
    void performMemoryTest(int numIterations) {
        Json::StreamWriterBuilder writerBuilder;
        
        for (int i = 0; i < numIterations && running_; ++i) {
            // 创建测试请求
            Json::Value tokenizeReqJson;
            tokenizeReqJson["text"] = "Memory test iteration " + std::to_string(i);
            
            std::string tokenizeRequestBody = Json::writeString(writerBuilder, tokenizeReqJson);
            
            HttpRequest tokenizeReq;
            tokenizeReq.setMethod("POST");
            tokenizeReq.setPath("/tokenize");
            tokenizeReq.setBody(tokenizeRequestBody);
            
            // 处理请求
            HttpResponse tokenizeResp = handler_->handleRequest(tokenizeReq);
            
            // 验证响应
            if (tokenizeResp.getStatusCode() != 200) {
                CLLM_ERROR("Memory test iteration %d failed: %d", i, tokenizeResp.getStatusCode());
            }
            
            // 解析响应以确保完整处理
            Json::Reader reader;
            Json::Value root;
            if (reader.parse(tokenizeResp.getBody(), root)) {
                // 验证响应结构
                if (root.isMember("tokens")) {
                    // 尝试将token转换为detokenize请求
                    Json::Value detokenizeReqJson;
                    detokenizeReqJson["tokens"] = root["tokens"];
                    
                    std::string detokenizeRequestBody = Json::writeString(writerBuilder, detokenizeReqJson);
                    
                    HttpRequest detokenizeReq;
                    detokenizeReq.setMethod("POST");
                    detokenizeReq.setPath("/detokenize");
                    detokenizeReq.setBody(detokenizeRequestBody);
                    
                    HttpResponse detokenizeResp = handler_->handleRequest(detokenizeReq);
                    
                    if (detokenizeResp.getStatusCode() != 200) {
                        CLLM_ERROR("Detokenize in memory test failed: %d", detokenizeResp.getStatusCode());
                    }
                }
            }
            
            // 每100次迭代打印一次进度
            if (i % 100 == 0) {
                CLLM_INFO("Memory test iteration %d completed", i);
            }
        }
    }

    // 获取资源使用情况的方法
    long getMemoryUsage() {
        return getCurrentMemoryUsage();
    }

    double getCPUTime() {
        return getCurrentCPUTime();
    }
};

TEST_F(HttpServerResourcesTest, MemoryUsageStability) {
    CLLM_INFO("=== Test: Memory Usage Stability ===");
    
    // 初始内存使用情况
    long initialMemory = getMemoryUsage();
    CLLM_INFO("Initial memory usage: %ld KB", initialMemory);
    
    // 执行大量请求
    const int numIterations = 1000;
    performMemoryTest(numIterations);
    
    // 最终内存使用情况
    long finalMemory = getMemoryUsage();
    CLLM_INFO("Final memory usage: %ld KB", finalMemory);
    
    // 计算内存增长
    long memoryGrowth = finalMemory - initialMemory;
    CLLM_INFO("Memory growth: %ld KB after %d requests", memoryGrowth, numIterations);
    
    // 验证内存增长在可接受范围内（例如，不超过5MB）
    EXPECT_LT(memoryGrowth, 5000);  // 5MB
    
    // 等待一段时间，看看是否有内存回收
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    long memoryAfterGC = getMemoryUsage();
    CLLM_INFO("Memory usage after GC: %ld KB", memoryAfterGC);
    
    // 内存回收应该会减少一些使用量
    EXPECT_LE(memoryAfterGC, finalMemory);
}

TEST_F(HttpServerResourcesTest, CPUUsage) {
    CLLM_INFO("=== Test: CPU Usage ===");
    
    // 记录开始时间和CPU时间
    double startTime = getCPUTime();
    auto wallStartTime = std::chrono::high_resolution_clock::now();
    
    // 执行大量请求
    const int numIterations = 500;
    performMemoryTest(numIterations);
    
    // 记录结束时间和CPU时间
    double endTime = getCPUTime();
    auto wallEndTime = std::chrono::high_resolution_clock::now();
    
    // 计算CPU时间使用
    double cpuTimeUsed = endTime - startTime;
    auto wallTimeUsed = std::chrono::duration_cast<std::chrono::seconds>(wallEndTime - wallStartTime).count();
    
    CLLM_INFO("CPU time used: %.2f seconds", cpuTimeUsed);
    CLLM_INFO("Wall time used: %ld seconds", wallTimeUsed);
    
    // 计算CPU使用率
    double cpuUsage = (cpuTimeUsed / wallTimeUsed) * 100.0;
    CLLM_INFO("CPU usage: %.2f%%", cpuUsage);
    
    // 验证CPU使用率不是异常高
    EXPECT_LT(cpuUsage, 1000.0);  // 防止异常高的CPU使用率（多线程可能超过100%）
    
    // 计算每个请求的平均CPU时间
    double avgCPUTimePerRequest = cpuTimeUsed / numIterations;
    CLLM_INFO("Average CPU time per request: %.4f seconds", avgCPUTimePerRequest);
}

TEST_F(HttpServerResourcesTest, LongRunningStability) {
    CLLM_INFO("=== Test: Long Running Stability ===");
    
    // 初始内存使用情况
    long initialMemory = getMemoryUsage();
    CLLM_INFO("Initial memory usage: %ld KB", initialMemory);
    
    // 创建多个线程来执行请求
    const int numThreads = 5;
    const int numIterations = 200;
    std::vector<std::future<void>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < numThreads; ++t) {
        futures.emplace_back(std::async(std::launch::async, [this, t, numIterations]() {
            Json::StreamWriterBuilder writerBuilder;
            
            for (int i = 0; i < numIterations; ++i) {
                // 创建测试请求
                Json::Value tokenizeReqJson;
                tokenizeReqJson["text"] = "Long running test - thread " + std::to_string(t) + ", iteration " + std::to_string(i);
                
                std::string tokenizeRequestBody = Json::writeString(writerBuilder, tokenizeReqJson);
                
                HttpRequest tokenizeReq;
                tokenizeReq.setMethod("POST");
                tokenizeReq.setPath("/tokenize");
                tokenizeReq.setBody(tokenizeRequestBody);
                
                // 处理请求
                HttpResponse tokenizeResp = handler_->handleRequest(tokenizeReq);
                
                // 验证响应
                ASSERT_EQ(tokenizeResp.getStatusCode(), 200);
            }
        }));
    }
    
    // 等待所有线程完成
    for (auto& future : futures) {
        future.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // 最终内存使用情况
    long finalMemory = getMemoryUsage();
    CLLM_INFO("Final memory usage: %ld KB", finalMemory);
    
    // 计算内存增长
    long memoryGrowth = finalMemory - initialMemory;
    CLLM_INFO("Memory growth after long running test: %ld KB", memoryGrowth);
    
    // 验证内存增长在可接受范围内
    EXPECT_LT(memoryGrowth, 10000);  // 10MB
    
    CLLM_INFO("Long running test completed in %ld seconds with %ld KB memory growth", duration, memoryGrowth);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
