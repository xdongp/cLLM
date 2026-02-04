/**
 * @file test_server_integration.cpp
 * @brief cLLM 服务器集成测试
 * @author cLLM Team
 * @date 2026-01-10
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <memory>
#include <curl/curl.h>
#include <json/json.h>

#include "cllm/http/http_server.h"
#include "cllm/http/handler.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/manager.h"

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
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
            
            CURLcode res = curl_easy_perform(curl);
            
            if (res == CURLE_OK) {
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
            }
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }
        
        return {httpCode, response};
    }
};

/**
 * @brief 服务器集成测试夹具
 */
class ServerIntegrationTest : public ::testing::Test {
protected:
    static std::unique_ptr<Scheduler> scheduler_;
    static std::unique_ptr<ModelExecutor> modelExecutor_;
    static std::unique_ptr<TokenizerManager> tokenizerManager_;
    static std::unique_ptr<HttpHandler> httpHandler_;
    static std::thread serverThread_;
    static std::string baseUrl_;
    static int port_;
    
    static void SetUpTestSuite() {
        port_ = 18080; // 使用测试端口避免冲突
        baseUrl_ = "http://127.0.0.1:" + std::to_string(port_);
        
        std::cout << "[TEST] Setting up test server on port " << port_ << std::endl;
        
        try {
            // 加载测试配置文件
            std::cout << "[TEST] Loading configuration files..." << std::endl;
            Config::instance().load("../config/sampler_config.yaml");
            
            // 创建一个模拟的模型执行器（用于测试，不加载实际模型）
            ModelConfig config;
            config.vocabSize = 32000;
            config.hiddenSize = 768;
            config.numLayers = 12;
            config.numAttentionHeads = 12;
            config.maxSequenceLength = 2048;
            
            // 使用占位模型路径（Kylin 后端支持）
            modelExecutor_ = std::make_unique<ModelExecutor>("", "", true, false);
            
            // 创建模拟分词器（tests/ 目录里提供 tokenizer.model）
            tokenizerManager_ = std::make_unique<TokenizerManager>("../tests", modelExecutor_.get());
            ITokenizer* tokenizer = tokenizerManager_->getTokenizer();
            
            // 创建调度器
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
                tokenizer
            );
            httpHandler_->post("/generate", [generateEndpoint](const HttpRequest& req) {
                return generateEndpoint->handle(req);
            });
            
            auto encodeEndpoint = std::make_shared<EncodeEndpoint>(tokenizer);
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
            
            std::cout << "[TEST] Test server started successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[TEST ERROR] Failed to start test server: " << e.what() << std::endl;
            throw;
        }
    }
    
    static void TearDownTestSuite() {
        std::cout << "[TEST] Shutting down test server..." << std::endl;
        
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
        tokenizerManager_.reset();
        modelExecutor_.reset();
        
        std::cout << "[TEST] Test server shutdown complete" << std::endl;
    }
};

// 静态成员初始化
std::unique_ptr<Scheduler> ServerIntegrationTest::scheduler_;
std::unique_ptr<ModelExecutor> ServerIntegrationTest::modelExecutor_;
std::unique_ptr<TokenizerManager> ServerIntegrationTest::tokenizerManager_;
std::unique_ptr<HttpHandler> ServerIntegrationTest::httpHandler_;
std::thread ServerIntegrationTest::serverThread_;
std::string ServerIntegrationTest::baseUrl_;
int ServerIntegrationTest::port_;

/**
 * @brief 测试健康检查端点
 */
TEST_F(ServerIntegrationTest, HealthEndpoint) {
    HttpClient client;
    
    auto [statusCode, response] = client.get(baseUrl_ + "/health");
    
    // 验证状态码
    EXPECT_EQ(statusCode, 200) << "Health endpoint should return 200 OK";
    
    // 验证响应内容
    EXPECT_FALSE(response.empty()) << "Response should not be empty";
    
    // 解析 JSON 响应
    Json::Value root;
    Json::Reader reader;
    bool parsingSuccessful = reader.parse(response, root);
    
    ASSERT_TRUE(parsingSuccessful) << "Failed to parse JSON response: " << response;
    EXPECT_TRUE(root.isMember("status")) << "Response should contain 'status' field";
    EXPECT_EQ(root["status"].asString(), "healthy") << "Status should be 'healthy'";
    
    std::cout << "[TEST] Health check response: " << response << std::endl;
}

/**
 * @brief 测试编码端点
 */
TEST_F(ServerIntegrationTest, EncodeEndpoint) {
    HttpClient client;
    
    // 构造请求
    std::string requestBody = R"({"text": "Hello, world!"})";
    
    auto [statusCode, response] = client.post(baseUrl_ + "/encode", requestBody);
    
    // 验证状态码
    EXPECT_EQ(statusCode, 200) << "Encode endpoint should return 200 OK";
    
    // 验证响应内容
    EXPECT_FALSE(response.empty()) << "Response should not be empty";
    
    // 解析 JSON 响应
    Json::Value root;
    Json::Reader reader;
    bool parsingSuccessful = reader.parse(response, root);
    
    ASSERT_TRUE(parsingSuccessful) << "Failed to parse JSON response: " << response;
    EXPECT_TRUE(root.isMember("tokens")) << "Response should contain 'tokens' field";
    EXPECT_TRUE(root.isMember("length")) << "Response should contain 'length' field";
    EXPECT_TRUE(root["tokens"].isArray()) << "Tokens should be an array";
    EXPECT_GT(root["length"].asInt(), 0) << "Token length should be greater than 0";
    
    std::cout << "[TEST] Encode response: " << response << std::endl;
}

/**
 * @brief 测试生成端点 - 简单请求
 */
TEST_F(ServerIntegrationTest, GenerateEndpointSimple) {
    HttpClient client;
    
    // 构造请求
    std::string requestBody = R"({
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0.7
    })";
    
    auto [statusCode, response] = client.post(baseUrl_ + "/generate", requestBody);
    
    // 验证状态码
    EXPECT_EQ(statusCode, 200) << "Generate endpoint should return 200 OK";
    
    // 验证响应内容
    EXPECT_FALSE(response.empty()) << "Response should not be empty";
    
    // 解析 JSON 响应
    Json::Value root;
    Json::Reader reader;
    bool parsingSuccessful = reader.parse(response, root);
    
    ASSERT_TRUE(parsingSuccessful) << "Failed to parse JSON response: " << response;
    EXPECT_TRUE(root.isMember("id")) << "Response should contain 'id' field";
    EXPECT_TRUE(root.isMember("text")) << "Response should contain 'text' field";
    EXPECT_TRUE(root.isMember("response_time")) << "Response should contain 'response_time' field";
    
    std::cout << "[TEST] Generate response: " << response << std::endl;
}

/**
 * @brief 测试生成端点 - 参数验证
 */
TEST_F(ServerIntegrationTest, GenerateEndpointWithParameters) {
    HttpClient client;
    
    // 构造请求，包含所有参数
    std::string requestBody = R"({
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": false
    })";
    
    auto [statusCode, response] = client.post(baseUrl_ + "/generate", requestBody);
    
    // 验证状态码
    EXPECT_EQ(statusCode, 200) << "Generate endpoint should return 200 OK";
    
    // 解析响应
    Json::Value root;
    Json::Reader reader;
    bool parsingSuccessful = reader.parse(response, root);
    
    ASSERT_TRUE(parsingSuccessful) << "Failed to parse JSON response: " << response;
    EXPECT_TRUE(root.isMember("tokens_per_second")) << "Response should contain performance metrics";
    
    // 验证性能指标
    float tokensPerSecond = root["tokens_per_second"].asFloat();
    EXPECT_GE(tokensPerSecond, 0.0f) << "Tokens per second should be non-negative";
    
    std::cout << "[TEST] Tokens per second: " << tokensPerSecond << std::endl;
}

/**
 * @brief 测试无效端点
 */
TEST_F(ServerIntegrationTest, InvalidEndpoint) {
    HttpClient client;
    
    auto [statusCode, response] = client.get(baseUrl_ + "/invalid");
    
    // 应该返回 404
    EXPECT_EQ(statusCode, 404) << "Invalid endpoint should return 404 Not Found";
}

/**
 * @brief 测试编码端点 - 缺少参数
 */
TEST_F(ServerIntegrationTest, EncodeEndpointMissingParameter) {
    HttpClient client;
    
    // 发送空请求
    std::string requestBody = "{}";
    
    auto [statusCode, response] = client.post(baseUrl_ + "/encode", requestBody);
    
    // 应该返回错误状态码
    EXPECT_NE(statusCode, 200) << "Should return error for missing parameters";
}

/**
 * @brief 测试并发请求
 */
TEST_F(ServerIntegrationTest, ConcurrentRequests) {
    const int numRequests = 5;
    std::vector<std::thread> threads;
    std::vector<int> statusCodes(numRequests);
    
    // 发送并发请求
    for (int i = 0; i < numRequests; ++i) {
        threads.emplace_back([this, i, &statusCodes]() {
            HttpClient client;
            auto [statusCode, response] = client.get(baseUrl_ + "/health");
            statusCodes[i] = statusCode;
        });
    }
    
    // 等待所有请求完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 验证所有请求都成功
    for (int i = 0; i < numRequests; ++i) {
        EXPECT_EQ(statusCodes[i], 200) << "Request " << i << " should succeed";
    }
    
    std::cout << "[TEST] All " << numRequests << " concurrent requests succeeded" << std::endl;
}

} // namespace test
} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
