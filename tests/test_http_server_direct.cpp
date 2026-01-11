/**
 * @file test_http_server_direct.cpp
 * @brief HTTP Server测试 - 直接集成Tokenizer和ModelExecutor (跳过Scheduler)
 * @author cLLM Team
 * @date 2026-01-11
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <json/json.h>

#include "cllm/http/drogon_server.h"
#include "cllm/http/handler.h"
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/model/executor.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include "cllm/model/config.h"

using namespace cllm;

class HttpServerDirectTest : public ::testing::Test {
protected:
    void SetUp() override {
        CLLM_INFO("=== Setting up HTTP Server Direct Test ===");
        
        // 1. 初始化Tokenizer
        std::string tokenizerPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/Qwen3-0.6B";
        tokenizer_ = std::make_unique<Tokenizer>(tokenizerPath);
        CLLM_INFO("Tokenizer initialized from: %s", tokenizerPath.c_str());
        
        // 2. 初始化ModelExecutor (使用LibTorch backend)
        std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
        std::string quantization = "";
        bool enableSIMD = true;
        bool useLibTorch = true;
        
        executor_ = std::make_unique<ModelExecutor>(modelPath, quantization, enableSIMD, useLibTorch);
        CLLM_INFO("ModelExecutor initialized with LibTorch backend");
        CLLM_INFO("Actual vocab_size: %zu", executor_->getConfig().vocabSize);
        
        // 3. 创建HttpHandler并注册/generate端点
        handler_ = std::make_unique<HttpHandler>();
        
        // 注册健康检查端点
        handler_->get("/health", [](const HttpRequest& req) -> HttpResponse {
            HttpResponse resp;
            resp.setStatusCode(200);
            resp.setBody(R"({"status":"healthy","server":"direct_integration"})");
            resp.setContentType("application/json");
            return resp;
        });
        
        // 注册/generate端点 - 直接使用Tokenizer和ModelExecutor
        handler_->post("/generate", [this](const HttpRequest& req) -> HttpResponse {
            return handleGenerate(req);
        });
        
        CLLM_INFO("HTTP Handler configured");
        
        // 4. 初始化DrogonServer (不启动,只用于测试Handler逻辑)
        serverHost_ = "0.0.0.0";
        serverPort_ = 8765;
        
        CLLM_INFO("=== Setup Complete ===");
    }
    
    void TearDown() override {
        CLLM_INFO("=== Tearing down HTTP Server Direct Test ===");
        handler_.reset();
        executor_.reset();
        tokenizer_.reset();
    }
    
    // 处理/generate请求的核心逻辑
    HttpResponse handleGenerate(const HttpRequest& request) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        try {
            // 1. 解析请求
            std::string body = request.getBody();
            Json::Value root;
            Json::CharReaderBuilder builder;
            std::istringstream bodyStream(body);
            std::string errs;
            
            if (!Json::parseFromStream(builder, bodyStream, &root, &errs)) {
                CLLM_ERROR("Failed to parse JSON: %s", errs.c_str());
                HttpResponse resp;
                resp.setStatusCode(400);
                resp.setBody(R"({"error":"Invalid JSON"})");
                resp.setContentType("application/json");
                return resp;
            }
            
            std::string prompt = root.get("prompt", "").asString();
            int maxTokens = root.get("max_tokens", 3).asInt();
            float temperature = root.get("temperature", 0.7f).asFloat();
            float topP = root.get("top_p", 0.9f).asFloat();
            
            CLLM_INFO("Generate request: prompt='%s', max_tokens=%d", prompt.c_str(), maxTokens);
            
            if (prompt.empty()) {
                HttpResponse resp;
                resp.setStatusCode(400);
                resp.setBody(R"({"error":"prompt is required"})");
                resp.setContentType("application/json");
                return resp;
            }
            
            // 2. Tokenize prompt
            std::vector<int> inputTokens = tokenizer_->encode(prompt, true);
            CLLM_INFO("Tokenized prompt: %zu tokens", inputTokens.size());
            
            // 限制输入长度(LibTorch模型trace限制)
            const size_t MAX_INPUT_TOKENS = 7;
            if (inputTokens.size() > MAX_INPUT_TOKENS) {
                CLLM_WARN("Input tokens (%zu) exceeds limit (%zu), truncating", 
                         inputTokens.size(), MAX_INPUT_TOKENS);
                inputTokens.resize(MAX_INPUT_TOKENS);
            }
            
            // 3. 执行推理 (生成maxTokens个token)
            std::vector<int> generatedTokens;
            std::vector<int> currentSequence = inputTokens;
            
            for (int i = 0; i < maxTokens && currentSequence.size() < 10; ++i) {
                // 构建BatchInput
                BatchInput batchInput;
                batchInput.inputIds = currentSequence;
                batchInput.requestPositions.push_back({0, currentSequence.size()});
                batchInput.batchSize = 1;
                batchInput.sequenceIds.push_back(0);
                
                // 执行前向传播
                BatchOutput batchOutput = executor_->forward(batchInput);
                
                // 获取最后一个位置的logits
                size_t vocabSize = executor_->getConfig().vocabSize;
                std::vector<float> lastLogits(vocabSize);
                size_t lastPos = currentSequence.size() - 1;
                
                // BatchOutput.logits 是 [batch_size, seq_len, vocab_size]
                // 对于batch_size=1, 提取最后一个位置
                for (size_t v = 0; v < vocabSize; ++v) {
                    lastLogits[v] = batchOutput.logits[lastPos * vocabSize + v];
                }
                
                // 简单采样: 取top-1 (greedy)
                int nextToken = std::distance(
                    lastLogits.begin(),
                    std::max_element(lastLogits.begin(), lastLogits.end())
                );
                
                CLLM_DEBUG("Generated token: %d", nextToken);
                
                // 检查是否是特殊token
                if (tokenizer_->isSpecialToken(nextToken)) {
                    CLLM_INFO("Hit special token, stopping generation");
                    break;
                }
                
                generatedTokens.push_back(nextToken);
                currentSequence.push_back(nextToken);
            }
            
            CLLM_INFO("Generated %zu tokens", generatedTokens.size());
            
            // 4. Decode生成的tokens
            std::string generatedText;
            if (!generatedTokens.empty()) {
                generatedText = tokenizer_->decode(generatedTokens, true);
                CLLM_INFO("Generated text: '%s'", generatedText.c_str());
            } else {
                generatedText = "";
                CLLM_WARN("No tokens generated");
            }
            
            // 5. 构建响应
            auto endTime = std::chrono::high_resolution_clock::now();
            float responseTime = std::chrono::duration<float>(endTime - startTime).count();
            float tokensPerSecond = generatedTokens.size() / std::max(responseTime, 0.001f);
            
            Json::Value responseJson;
            responseJson["id"] = "direct_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            responseJson["text"] = generatedText;
            responseJson["tokens_generated"] = static_cast<int>(generatedTokens.size());
            responseJson["response_time"] = responseTime;
            responseJson["tokens_per_second"] = tokensPerSecond;
            
            Json::StreamWriterBuilder writerBuilder;
            std::string responseBody = Json::writeString(writerBuilder, responseJson);
            
            HttpResponse resp;
            resp.setStatusCode(200);
            resp.setBody(responseBody);
            resp.setContentType("application/json");
            return resp;
            
        } catch (const std::exception& e) {
            CLLM_ERROR("Error processing generate request: %s", e.what());
            
            HttpResponse resp;
            resp.setStatusCode(500);
            
            Json::Value errorJson;
            errorJson["error"] = std::string("Internal error: ") + e.what();
            
            Json::StreamWriterBuilder writerBuilder;
            std::string errorBody = Json::writeString(writerBuilder, errorJson);
            
            resp.setBody(errorBody);
            resp.setContentType("application/json");
            return resp;
        }
    }
    
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ModelExecutor> executor_;
    std::unique_ptr<HttpHandler> handler_;
    std::string serverHost_;
    int serverPort_;
};

// ==================== 测试用例 ====================

TEST_F(HttpServerDirectTest, HealthCheck) {
    CLLM_INFO("=== Test: Health Check ===");
    
    HttpRequest req;
    req.setMethod("GET");
    req.setPath("/health");
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 200);
    EXPECT_TRUE(resp.getBody().find("healthy") != std::string::npos);
    
    CLLM_INFO("Health check response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerDirectTest, GenerateBasic) {
    CLLM_INFO("=== Test: Basic Generate ===");
    
    // 构建请求
    Json::Value requestJson;
    requestJson["prompt"] = "Hello";
    requestJson["max_tokens"] = 3;
    requestJson["temperature"] = 0.7;
    requestJson["top_p"] = 0.9;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/generate");
    req.setBody(requestBody);
    
    // 执行请求
    HttpResponse resp = handler_->handleRequest(req);
    
    // 验证响应
    EXPECT_EQ(resp.getStatusCode(), 200);
    EXPECT_EQ(resp.getContentType(), "application/json");
    
    // 解析响应
    Json::Value responseJson;
    Json::CharReaderBuilder readerBuilder;
    std::istringstream respStream(resp.getBody());
    std::string errs;
    ASSERT_TRUE(Json::parseFromStream(readerBuilder, respStream, &responseJson, &errs));
    
    EXPECT_TRUE(responseJson.isMember("text"));
    EXPECT_TRUE(responseJson.isMember("tokens_generated"));
    EXPECT_TRUE(responseJson.isMember("response_time"));
    
    CLLM_INFO("Generated text: %s", responseJson["text"].asString().c_str());
    CLLM_INFO("Tokens generated: %d", responseJson["tokens_generated"].asInt());
    CLLM_INFO("Response time: %.3f seconds", responseJson["response_time"].asFloat());
    CLLM_INFO("Tokens/sec: %.2f", responseJson["tokens_per_second"].asFloat());
}

TEST_F(HttpServerDirectTest, GenerateWithLongerPrompt) {
    CLLM_INFO("=== Test: Generate with Longer Prompt ===");
    
    Json::Value requestJson;
    requestJson["prompt"] = "The quick brown fox";
    requestJson["max_tokens"] = 5;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/generate");
    req.setBody(requestBody);
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 200);
    
    Json::Value responseJson;
    Json::CharReaderBuilder readerBuilder;
    std::istringstream respStream(resp.getBody());
    std::string errs;
    ASSERT_TRUE(Json::parseFromStream(readerBuilder, respStream, &responseJson, &errs));
    
    CLLM_INFO("Generated text: %s", responseJson["text"].asString().c_str());
    CLLM_INFO("Tokens generated: %d", responseJson["tokens_generated"].asInt());
}

TEST_F(HttpServerDirectTest, GenerateEmptyPrompt) {
    CLLM_INFO("=== Test: Generate with Empty Prompt (Error Handling) ===");
    
    Json::Value requestJson;
    requestJson["prompt"] = "";
    requestJson["max_tokens"] = 3;
    
    Json::StreamWriterBuilder writerBuilder;
    std::string requestBody = Json::writeString(writerBuilder, requestJson);
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/generate");
    req.setBody(requestBody);
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 400);
    EXPECT_TRUE(resp.getBody().find("error") != std::string::npos);
    
    CLLM_INFO("Error response: %s", resp.getBody().c_str());
}

TEST_F(HttpServerDirectTest, GenerateInvalidJSON) {
    CLLM_INFO("=== Test: Generate with Invalid JSON (Error Handling) ===");
    
    HttpRequest req;
    req.setMethod("POST");
    req.setPath("/generate");
    req.setBody("{invalid json");
    
    HttpResponse resp = handler_->handleRequest(req);
    
    EXPECT_EQ(resp.getStatusCode(), 400);
    EXPECT_TRUE(resp.getBody().find("error") != std::string::npos);
    
    CLLM_INFO("Error response: %s", resp.getBody().c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
