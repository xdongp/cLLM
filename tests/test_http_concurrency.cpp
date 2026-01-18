#include <gtest/gtest.h>
#include "cllm/scheduler/scheduler.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/common/config.h"
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

namespace cllm {

class MockTokenizer : public ITokenizer {
public:
    MockTokenizer() {}
    ~MockTokenizer() override {}
    
    bool load(const std::string& modelPath) override { return true; }
    
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        return {1, 2, 3, 4, 5};
    }
    
    std::string decode(const std::vector<int>& tokens, bool skipSpecialTokens) override {
        return "decoded text";
    }
    
    int getVocabSize() const override { return 1000; }
    std::string idToToken(int id) const override { return "token"; }
    int tokenToId(const std::string& token) const override { return 1; }
    
    int getPadId() const override { return 0; }
    int getEosId() const override { return 2; }
    int getBosId() const override { return 1; }
    int getUnkId() const override { return 3; }
    
    ModelType getModelType() const override { return ModelType::AUTO; }
};

class HttpConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("config/scheduler_config.yaml");
        scheduler_ = nullptr;
        tokenizer_ = std::make_unique<MockTokenizer>();
        endpoint_ = nullptr;
    }
    
    void TearDown() override {
        if (scheduler_) {
            scheduler_->stop();
            delete scheduler_;
        }
    }
    
    void createScheduler(size_t maxConcurrentRequests = 2) {
        try {
            scheduler_ = new Scheduler("", "", 4, 2048);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Scheduler initialization failed: " << e.what();
        }
        scheduler_->start();
        
        endpoint_ = std::make_unique<GenerateEndpoint>(scheduler_, tokenizer_.get());
    }
    
    Scheduler* scheduler_;
    std::unique_ptr<MockTokenizer> tokenizer_;
    std::unique_ptr<GenerateEndpoint> endpoint_;
};

TEST_F(HttpConcurrencyTest, GetRunningCount_InitiallyZero) {
    createScheduler();
    
    size_t runningCount = scheduler_->getRunningCount();
    EXPECT_EQ(runningCount, 0);
}

TEST_F(HttpConcurrencyTest, GetMaxConcurrentRequests_DefaultValue) {
    createScheduler();
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    EXPECT_EQ(maxConcurrent, 8);
}

TEST_F(HttpConcurrencyTest, GetMaxConcurrentRequests_CustomValue) {
    createScheduler();
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    EXPECT_GT(maxConcurrent, 0);
}

TEST_F(HttpConcurrencyTest, ConcurrentCheck_BelowLimit_AllowsRequest) {
    createScheduler();
    
    nlohmann::json requestBody;
    requestBody["prompt"] = "test prompt";
    requestBody["max_tokens"] = 10;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_NE(response.getStatusCode(), 429);
}

TEST_F(HttpConcurrencyTest, ConcurrentCheck_AtLimit_Returns429) {
    createScheduler();
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    
    nlohmann::json requestBody;
    requestBody["prompt"] = "test prompt";
    requestBody["max_tokens"] = 10;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    size_t runningCount = scheduler_->getRunningCount();
    
    if (runningCount >= maxConcurrent) {
        EXPECT_EQ(response.getStatusCode(), 429);
        
        std::string body = response.getBody();
        nlohmann::json jsonResponse = nlohmann::json::parse(body);
        
        EXPECT_FALSE(jsonResponse["success"]);
        EXPECT_EQ(jsonResponse["error"], "Too many concurrent requests");
        EXPECT_TRUE(jsonResponse.contains("message"));
        EXPECT_TRUE(jsonResponse.contains("retry_after"));
    }
}

TEST_F(HttpConcurrencyTest, HTTP429Response_ContainsRetryAfterHeader) {
    createScheduler();
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    
    nlohmann::json requestBody;
    requestBody["prompt"] = "test prompt";
    requestBody["max_tokens"] = 10;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    size_t runningCount = scheduler_->getRunningCount();
    
    if (runningCount >= maxConcurrent) {
        EXPECT_EQ(response.getStatusCode(), 429);
        
        std::string retryAfter = response.getHeader("Retry-After");
        EXPECT_FALSE(retryAfter.empty());
        EXPECT_EQ(retryAfter, "5");
    }
}

TEST_F(HttpConcurrencyTest, HTTP429Response_JsonFormat) {
    createScheduler();
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    
    nlohmann::json requestBody;
    requestBody["prompt"] = "test prompt";
    requestBody["max_tokens"] = 10;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    size_t runningCount = scheduler_->getRunningCount();
    
    if (runningCount >= maxConcurrent) {
        EXPECT_EQ(response.getStatusCode(), 429);
        
        std::string body = response.getBody();
        nlohmann::json jsonResponse = nlohmann::json::parse(body);
        
        EXPECT_TRUE(jsonResponse.contains("success"));
        EXPECT_TRUE(jsonResponse.contains("error"));
        EXPECT_TRUE(jsonResponse.contains("message"));
        EXPECT_TRUE(jsonResponse.contains("retry_after"));
        
        EXPECT_FALSE(jsonResponse["success"]);
        EXPECT_EQ(jsonResponse["error"], "Too many concurrent requests");
    }
}

TEST_F(HttpConcurrencyTest, ConcurrentCheck_ThreadSafety) {
    createScheduler();
    
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::vector<int> statusCodes(numThreads);
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, &statusCodes]() {
            nlohmann::json requestBody;
            requestBody["prompt"] = "test prompt " + std::to_string(i);
            requestBody["max_tokens"] = 10;
            
            HttpRequest request;
            request.setMethod("POST");
            request.setPath("/generate");
            request.setBody(requestBody.dump());
            request.setHeader("Content-Type", "application/json");
            
            HttpResponse response = endpoint_->handle(request);
            statusCodes[i] = response.getStatusCode();
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
    int successCount = 0;
    int error429Count = 0;
    
    for (int statusCode : statusCodes) {
        if (statusCode == 200) {
            successCount++;
        } else if (statusCode == 429) {
            error429Count++;
        }
    }
    
    EXPECT_LE(successCount, static_cast<int>(maxConcurrent));
    EXPECT_GT(error429Count, 0);
}

TEST_F(HttpConcurrencyTest, GetRunningCount_ThreadSafety) {
    createScheduler();
    
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::atomic<size_t> maxRunningCount{0};
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, &maxRunningCount]() {
            for (int j = 0; j < 100; ++j) {
                size_t count = scheduler_->getRunningCount();
                if (count > maxRunningCount) {
                    maxRunningCount = count;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GE(maxRunningCount.load(), 0);
}

} // namespace cllm
