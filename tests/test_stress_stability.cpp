/**
 * @file test_stress_stability.cpp
 * @brief Phase 4.3: 压力和稳定性测试 (Phase 4.3)
 *
 * 测试内容：
 * 1. 高并发测试（8并发）
 * 2. 长时间运行测试（5-10分钟）
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <memory>
#include <curl/curl.h>
#include <json/json.h>
#include <vector>
#include <atomic>
#include <iostream>

#include "cllm/http/http_server.h"
#include "cllm/http/handler.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/common/logger.h"

namespace cllm {
namespace test {

// 用于存储 HTTP 响应的回调函数
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

/**
 * @brief HTTP 客户端辅助类
 */
class HttpClient {
public:
    HttpClient() {
        curl_global_init(CURL_GLOBAL_ALL);
    }
    
    ~HttpClient() {
        curl_global_cleanup();
    }
    
    /**
     * @brief 发送 POST 请求
     */
    std::pair<int, std::string> post(const std::string& url, const std::string& data) {
        CURL* curl = curl_easy_init();
        std::string response;
        int httpCode = 0;
        
        if (curl) {
            struct curl_slist* headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
            
            CURLcode res = curl_easy_perform(curl);
            
            if (res == CURLE_OK) {
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
            }
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }
        
        return {httpCode, response};
    }
    
    /**
     * @brief 发送 GET 请求
     */
    std::pair<int, std::string> get(const std::string& url) {
        CURL* curl = curl_easy_init();
        std::string response;
        int httpCode = 0;
        
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
            
            CURLcode res = curl_easy_perform(curl);
            
            if (res == CURLE_OK) {
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
            }
            
            curl_easy_cleanup(curl);
        }
        
        return {httpCode, response};
    }
};

/**
 * @brief 压力和稳定性测试类
 */
class StressStabilityTest : public ::testing::Test {
protected:
    static std::unique_ptr<HttpHandler> httpHandler_;
    static std::unique_ptr<Scheduler> scheduler_;
    static std::unique_ptr<Tokenizer> tokenizer_;
    static std::unique_ptr<ModelExecutor> modelExecutor_;
    static std::thread serverThread_;
    static int port_;
    static std::string baseUrl_;
    
    static void SetUpTestSuite() {
        CLLM_INFO("Setting up StressStabilityTest suite");
        
        // 随机选择一个可用端口
        port_ = 8080 + (rand() % 1000);
        baseUrl_ = "http://localhost:" + std::to_string(port_);
        
        try {
            // 加载模型和 tokenizer（如果环境变量中指定了模型路径）
            const char* modelPath = std::getenv("CLLM_TEST_MODEL_PATH");
            if (modelPath) {
                CLLM_INFO("Loading model from: %s", modelPath);
                
                // 初始化模型执行器
                modelExecutor_ = std::make_unique<ModelExecutor>(modelPath, "", true, false);
                
                // 初始化 tokenizer
                tokenizer_ = std::make_unique<Tokenizer>(std::string(modelPath) + "/tokenizer.model");
                
                // 初始化调度器
                scheduler_ = std::make_unique<Scheduler>(
                    modelExecutor_.get(),
                    8,    // maxBatchSize
                    2048  // maxContextLength
                );
                scheduler_->start();
                
                // 创建 HTTP 处理器和端点
                httpHandler_ = std::make_unique<HttpHandler>();
                
                auto healthEndpoint = std::make_shared<HealthEndpoint>();
                httpHandler_->get("/health", [healthEndpoint](const HttpRequest& req) {
                    return healthEndpoint->handle(req);
                });
                
                auto generateEndpoint = std::make_shared<GenerateEndpoint>(
                    scheduler_.get(),
                    tokenizer_.get()
                );
                httpHandler_->post("/generate", [generateEndpoint](const HttpRequest& req) {
                    return generateEndpoint->handle(req);
                });
                
                auto encodeEndpoint = std::make_shared<EncodeEndpoint>(tokenizer_.get());
                httpHandler_->post("/encode", [encodeEndpoint](const HttpRequest& req) {
                    return encodeEndpoint->handle(req);
                });
                
                // 初始化并启动服务器
                HttpServer::init("127.0.0.1", port_, httpHandler_.get());
                
                // 在单独的线程中启动服务器
                serverThread_ = std::thread([]() {
                    HttpServer::start();
                });
                
                // 等待服务器启动
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                CLLM_INFO("Test server started successfully on port %d", port_);
                
            } else {
                CLLM_ERROR("CLLM_TEST_MODEL_PATH environment variable not set");
                FAIL() << "CLLM_TEST_MODEL_PATH environment variable not set";
            }
        } catch (const std::exception& e) {
            CLLM_ERROR("Failed to start test server: %s", e.what());
            FAIL() << "Failed to start test server: " << e.what();
        }
    }
    
    static void TearDownTestSuite() {
        CLLM_INFO("Tearing down StressStabilityTest suite");
        
        // 停止调度器
        if (scheduler_) {
            scheduler_->stop();
        }
        
        // 停止服务器
        HttpServer::stop();
        
        // 等待服务器线程结束
        if (serverThread_.joinable()) {
            serverThread_.join();
        }
        
        // 清理资源
        httpHandler_.reset();
        scheduler_.reset();
        tokenizer_.reset();
        modelExecutor_.reset();
        
        CLLM_INFO("StressStabilityTest suite teardown completed");
    }
    
    void SetUp() override {
        CLLM_INFO("Setting up StressStabilityTest");
        
        // 每个测试用例的设置
        CLLM_INFO("StressStabilityTest setup completed");
    }
    
    void TearDown() override {
        CLLM_INFO("Tearing down StressStabilityTest");
        
        // 每个测试用例的清理
        CLLM_INFO("StressStabilityTest teardown completed");
    }
};

// 静态成员初始化
std::unique_ptr<HttpHandler> StressStabilityTest::httpHandler_;
std::unique_ptr<Scheduler> StressStabilityTest::scheduler_;
std::unique_ptr<Tokenizer> StressStabilityTest::tokenizer_;
std::unique_ptr<ModelExecutor> StressStabilityTest::modelExecutor_;
std::thread StressStabilityTest::serverThread_;
int StressStabilityTest::port_;
std::string StressStabilityTest::baseUrl_;

/**
 * @brief 发送单个测试请求的辅助函数
 */
bool sendTestRequest(HttpClient& client, const std::string& baseUrl, const std::string& requestId) {
    Json::Value request;
    request["prompt"] = "Hello, how are you?";
    request["max_tokens"] = 10;
    request["temperature"] = 0.7;
    
    Json::FastWriter writer;
    std::string requestData = writer.write(request);
    
    std::string url = baseUrl + "/generate";
    
    auto [httpCode, response] = client.post(url, requestData);
    
    if (httpCode == 200) {
        CLLM_INFO("Request %s succeeded, response length: %zu", requestId.c_str(), response.size());
        return true;
    } else {
        CLLM_ERROR("Request %s failed, status code: %d, response: %s", requestId.c_str(), httpCode, response.c_str());
        return false;
    }
}

/**
 * @brief 高并发测试（8并发）
 */
TEST_F(StressStabilityTest, ConcurrentRequests) {
    const int CONCURRENT_COUNT = 8;
    const int REQUESTS_PER_THREAD = 5;
    
    std::vector<std::thread> threads;
    std::atomic<int> successfulRequests(0);
    std::atomic<int> failedRequests(0);
    
    CLLM_INFO("Starting %d concurrent threads, each sending %d requests", CONCURRENT_COUNT, REQUESTS_PER_THREAD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 启动并发线程
    for (int i = 0; i < CONCURRENT_COUNT; ++i) {
        threads.emplace_back([&, i]() {
            HttpClient client;
            for (int j = 0; j < REQUESTS_PER_THREAD; ++j) {
                std::string requestId = "thread_" + std::to_string(i) + "_req_" + std::to_string(j);
                if (sendTestRequest(client, baseUrl_, requestId)) {
                    successfulRequests++;
                } else {
                    failedRequests++;
                }
                
                // 短暂延迟，避免请求过于密集
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    
    CLLM_INFO("Concurrent test completed in %.2f seconds", duration);
    CLLM_INFO("Successful requests: %d", successfulRequests.load());
    CLLM_INFO("Failed requests: %d", failedRequests.load());
    
    // 验证所有请求都成功
    EXPECT_EQ(failedRequests.load(), 0) << "Some requests failed during concurrent test";
    EXPECT_GT(successfulRequests.load(), 0) << "No successful requests";
}

/**
 * @brief 长时间运行测试（5-10分钟，实际使用5分钟）
 */
TEST_F(StressStabilityTest, LongRunningTest) {
    const int TEST_DURATION_MINUTES = 5;
    const int REQUEST_INTERVAL_MS = 1000;
    
    std::atomic<int> successfulRequests(0);
    std::atomic<int> failedRequests(0);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::minutes(TEST_DURATION_MINUTES);
    
    CLLM_INFO("Starting long running test for %d minutes", TEST_DURATION_MINUTES);
    
    // 创建客户端
    HttpClient client;
    
    // 持续发送请求，直到达到测试时长
    int requestCount = 0;
    while (std::chrono::high_resolution_clock::now() < endTime) {
        std::string requestId = "longrun_req_" + std::to_string(requestCount++);
        if (sendTestRequest(client, baseUrl_, requestId)) {
            successfulRequests++;
        } else {
            failedRequests++;
        }
        
        // 等待一段时间再发送下一个请求
        std::this_thread::sleep_for(std::chrono::milliseconds(REQUEST_INTERVAL_MS));
        
        // 每100个请求输出一次状态
        if (requestCount % 100 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - startTime).count();
            CLLM_INFO("Long running test: %d minutes elapsed, %d requests sent", 
                     static_cast<int>(elapsed), requestCount);
        }
    }
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    
    CLLM_INFO("Long running test completed in %d seconds", static_cast<int>(elapsed));
    CLLM_INFO("Total requests: %d", requestCount);
    CLLM_INFO("Successful requests: %d", successfulRequests.load());
    CLLM_INFO("Failed requests: %d", failedRequests.load());
    
    // 验证成功率
    double successRate = static_cast<double>(successfulRequests.load()) / requestCount;
    CLLM_INFO("Success rate: %.2f%%", successRate * 100);
    
    EXPECT_GT(successfulRequests.load(), 0) << "No successful requests";
    EXPECT_LT(failedRequests.load(), requestCount * 0.1) << "Too many failed requests (>10%)";
}

} // namespace test
} // namespace cllm
