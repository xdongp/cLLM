#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer_manager.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/endpoint/generate_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/logger.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <memory>

using json = nlohmann::json;

int main() {
    try {
        // 设置日志
        cllm::Logger::instance().setLevel(cllm::LogLevel::INFO);
        
        // 创建模型配置
        cllm::ModelConfig modelConfig;
        modelConfig.vocabSize = 151936; // Qwen3 词表大小
        modelConfig.maxSequenceLength = 2048;
        modelConfig.defaultMaxTokens = 10;
        modelConfig.temperature = 0.7;
        modelConfig.topK = 50;
        modelConfig.topP = 0.9;
        
        // 创建模型执行器（使用空模型路径，启用占位权重模式）
        cllm::ModelExecutor modelExecutor("", "fp16", true, false);
        
        // 加载模型（实际会使用占位权重）
        modelExecutor.loadModel();
        
        // 初始化分词器（使用默认的 vocab.txt）
        // 注意：这里我们需要一个真实的分词器文件路径，或者模拟一个
        // 为了简单起见，我们直接使用一个不存在的路径，看看代码如何处理
        try {
            cllm::TokenizerManager tokenizerManager("/tmp", &modelExecutor);
            cllm::ITokenizer* tokenizer = tokenizerManager.getTokenizer();
            
            // 创建调度器
            cllm::Scheduler scheduler(&modelExecutor, 8, 2048);
            scheduler.start();
            
            // 创建生成端点
            cllm::GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
            
            // 创建测试请求
            std::string requestBody = R"({
                "prompt": "hello",
                "max_tokens": 10,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            })";
            
            cllm::HttpRequest request;
            request.setMethod(cllm::HttpMethod::POST);
            request.setPath("/generate");
            request.setBody(requestBody);
            request.setHeader("Content-Type", "application/json");
            
            // 处理请求
            std::unique_ptr<cllm::HttpResponse> response = generateEndpoint.handle(request);
            
            // 输出结果
            std::cout << "Status Code: " << static_cast<int>(response->getStatus()) << std::endl;
            std::cout << "Response Body: " << response->getBody() << std::endl;
            
            // 停止调度器
            scheduler.stop();
            
        } catch (const std::exception& e) {
            std::cerr << "Tokenizer initialization error: " << e.what() << std::endl;
            std::cerr << "Creating mock tokenizer instead..." << std::endl;
            
            // 创建一个非常简单的模拟分词器
            class MockTokenizer : public cllm::ITokenizer {
            public:
                size_t getVocabSize() const override { return 151936; }
                std::vector<int> encode(const std::string& text) const override {
                    // 简单的模拟编码
                    std::vector<int> tokens;
                    for (char c : text) {
                        tokens.push_back(static_cast<int>(c) + 100);
                    }
                    return tokens;
                }
                std::string decode(const std::vector<int>& tokens) const override {
                    // 简单的模拟解码
                    std::string text;
                    for (int token : tokens) {
                        text += static_cast<char>(token - 100);
                    }
                    return text;
                }
            };
            
            MockTokenizer mockTokenizer;
            
            // 创建调度器
            cllm::Scheduler scheduler(&modelExecutor, 8, 2048);
            scheduler.start();
            
            // 创建生成端点
            cllm::GenerateEndpoint generateEndpoint(&scheduler, &mockTokenizer);
            
            // 创建测试请求
            std::string requestBody = R"({
                "prompt": "hello",
                "max_tokens": 10,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            })";
            
            cllm::HttpRequest request;
            request.setMethod(cllm::HttpMethod::POST);
            request.setPath("/generate");
            request.setBody(requestBody);
            request.setHeader("Content-Type", "application/json");
            
            // 处理请求
            std::unique_ptr<cllm::HttpResponse> response = generateEndpoint.handle(request);
            
            // 输出结果
            std::cout << "Status Code: " << static_cast<int>(response->getStatus()) << std::endl;
            std::cout << "Response Body: " << response->getBody() << std::endl;
            
            // 停止调度器
            scheduler.stop();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}