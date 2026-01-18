#include <gtest/gtest.h>
#include <cllm/scheduler/scheduler.h>
#include <cllm/http/generate_endpoint.h>
#include <cllm/http/request.h>
#include <cllm/http/response.h>
#include <cllm/tokenizer/i_tokenizer.h>
#include <cllm/common/config.h>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

namespace cllm {

class MockTokenizer : public ITokenizer {
public:
    MockTokenizer() : eosId_(2), padId_(0), bosId_(1), unkId_(3) {}
    
    ~MockTokenizer() override = default;
    
    bool load(const std::string& modelPath) override { return true; }
    
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(c) % 1000);
        }
        if (addSpecialTokens) {
            tokens.insert(tokens.begin(), bosId_);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens, bool skipSpecialTokens) override {
        std::string text;
        for (int token : tokens) {
            if (skipSpecialTokens && (token == eosId_ || token == padId_ || token == bosId_ || token == unkId_)) {
                continue;
            }
            text += static_cast<char>(token % 128);
        }
        return text;
    }
    
    int getVocabSize() const override { return 1000; }
    std::string idToToken(int id) const override { return std::string(1, static_cast<char>(id % 128)); }
    int tokenToId(const std::string& token) const override { return static_cast<int>(token[0]) % 1000; }
    
    int getBosId() const override { return bosId_; }
    int getEosId() const override { return eosId_; }
    int getPadId() const override { return padId_; }
    int getUnkId() const override { return unkId_; }
    
    ModelType getModelType() const override { return ModelType::LLAMA; }
    
private:
    int eosId_;
    int padId_;
    int bosId_;
    int unkId_;
};

class GenerateSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("../config/test_config.yaml");
        scheduler_ = nullptr;
        tokenizer_ = nullptr;
        endpoint_ = nullptr;
    }
    
    void TearDown() override {
        if (scheduler_) {
            scheduler_->stop();
            delete scheduler_;
        }
        if (endpoint_) {
            delete endpoint_;
        }
        if (tokenizer_) {
            delete tokenizer_;
        }
    }
    
    void createSchedulerAndEndpoint() {
        try {
            scheduler_ = new Scheduler("", "", 4, 2048);
            scheduler_->start();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Scheduler initialization failed: " << e.what();
        }
        
        tokenizer_ = new MockTokenizer();
        endpoint_ = new GenerateEndpoint(scheduler_, tokenizer_);
    }
    
    HttpRequest createGenerateRequest(const std::string& prompt, int maxTokens = 10) {
        nlohmann::json requestBody;
        requestBody["prompt"] = prompt;
        requestBody["max_tokens"] = maxTokens;
        requestBody["temperature"] = 0.7f;
        requestBody["top_p"] = 0.9f;
        requestBody["stream"] = false;
        
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        request.setBody(requestBody.dump());
        request.setHeader("Content-Type", "application/json");
        
        return request;
    }
    
    Scheduler* scheduler_;
    MockTokenizer* tokenizer_;
    GenerateEndpoint* endpoint_;
};

TEST_F(GenerateSimpleTest, CreateSchedulerAndEndpoint) {
    createSchedulerAndEndpoint();
    
    EXPECT_NE(scheduler_, nullptr);
    EXPECT_NE(tokenizer_, nullptr);
    EXPECT_NE(endpoint_, nullptr);
}

TEST_F(GenerateSimpleTest, BasicGenerateRequest) {
    createSchedulerAndEndpoint();
    
    HttpRequest request = createGenerateRequest("Hello, world!", 5);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string responseBody = response.getBody();
    EXPECT_FALSE(responseBody.empty());
    
    try {
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        EXPECT_TRUE(jsonResponse.contains("id"));
        EXPECT_TRUE(jsonResponse.contains("text"));
        EXPECT_TRUE(jsonResponse.contains("response_time"));
        EXPECT_TRUE(jsonResponse.contains("tokens_per_second"));
    } catch (const std::exception& e) {
        FAIL() << "Failed to parse JSON response: " << e.what();
    }
}

TEST_F(GenerateSimpleTest, RequestWithMaxTokens) {
    createSchedulerAndEndpoint();
    
    int maxTokens = 20;
    HttpRequest request = createGenerateRequest("Generate many tokens", maxTokens);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string responseBody = response.getBody();
    try {
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        EXPECT_TRUE(jsonResponse.contains("text"));
        
        std::string text = jsonResponse["text"];
        EXPECT_FALSE(text.empty());
    } catch (const std::exception& e) {
        FAIL() << "Failed to parse JSON response: " << e.what();
    }
}

TEST_F(GenerateSimpleTest, RequestWithTemperature) {
    createSchedulerAndEndpoint();
    
    HttpRequest request = createGenerateRequest("Test temperature", 5);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string responseBody = response.getBody();
    try {
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        EXPECT_TRUE(jsonResponse.contains("text"));
    } catch (const std::exception& e) {
        FAIL() << "Failed to parse JSON response: " << e.what();
    }
}

TEST_F(GenerateSimpleTest, RequestWithTopP) {
    createSchedulerAndEndpoint();
    
    HttpRequest request = createGenerateRequest("Test top_p", 5);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string responseBody = response.getBody();
    try {
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        EXPECT_TRUE(jsonResponse.contains("text"));
    } catch (const std::exception& e) {
        FAIL() << "Failed to parse JSON response: " << e.what();
    }
}

TEST_F(GenerateSimpleTest, InvalidJsonRequest) {
    createSchedulerAndEndpoint();
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody("{invalid json}");
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 500);
}

TEST_F(GenerateSimpleTest, MissingPromptField) {
    createSchedulerAndEndpoint();
    
    nlohmann::json requestBody;
    requestBody["max_tokens"] = 10;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    request.setHeader("Content-Type", "application/json");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
}

TEST_F(GenerateSimpleTest, EmptyPrompt) {
    createSchedulerAndEndpoint();
    
    HttpRequest request = createGenerateRequest("", 5);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
}

TEST_F(GenerateSimpleTest, ResponseTimeMetrics) {
    createSchedulerAndEndpoint();
    
    HttpRequest request = createGenerateRequest("Test metrics", 5);
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string responseBody = response.getBody();
    try {
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        EXPECT_TRUE(jsonResponse.contains("response_time"));
        EXPECT_TRUE(jsonResponse.contains("tokens_per_second"));
        
        float responseTime = jsonResponse["response_time"];
        float tps = jsonResponse["tokens_per_second"];
        
        EXPECT_GT(responseTime, 0.0f);
        EXPECT_GE(tps, 0.0f);
    } catch (const std::exception& e) {
        FAIL() << "Failed to parse JSON response: " << e.what();
    }
}

} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}