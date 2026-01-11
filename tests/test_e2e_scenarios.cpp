/**
 * @file test_e2e_scenarios.cpp
 * @brief cLLM E2E 场景测试
 * @author cLLM Team
 * @date 2026-01-11
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <memory>
#include <curl/curl.h>
#include <json/json.h>
#include <algorithm>
#include <vector>
#include <tuple>

#include "cllm/http/drogon_server.h"
#include "cllm/http/handler.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer.h"

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
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L); // 增加超时时间以处理模型推理
            
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
 * @brief E2E 场景测试夹具
 */
class E2EScenarios : public ::testing::Test {
protected:
    static std::unique_ptr<Scheduler> scheduler_;
    static std::unique_ptr<ModelExecutor> modelExecutor_;
    static std::unique_ptr<Tokenizer> tokenizer_;
    static std::unique_ptr<HttpHandler> httpHandler_;
    static std::thread serverThread_;
    static std::string baseUrl_;
    static int port_;
    
    static void SetUpTestSuite() {
        port_ = 18081; // 使用测试端口避免冲突
        baseUrl_ = "http://127.0.0.1:" + std::to_string(port_);
        
        std::cout << "[TEST] Setting up E2E test server on port " << port_ << std::endl;
        
        try {
            // 加载测试配置文件
            std::cout << "[TEST] Loading configuration files..." << std::endl;
            Config::instance().load("../config/sampler_config.yaml");
            
            // 获取模型路径
            const char* modelPathEnv = std::getenv("CLLM_TEST_MODEL_PATH");
            if (!modelPathEnv) {
                FAIL() << "CLLM_TEST_MODEL_PATH environment variable not set";
                return;
            }
            std::string modelPath = modelPathEnv;
            std::cout << "[TEST] Using model path: " << modelPath << std::endl;
            
            // 初始化模型执行器
            modelExecutor_ = std::make_unique<ModelExecutor>(modelPath, "kylin", false, true);
            
            // 初始化分词器
            tokenizer_ = std::make_unique<Tokenizer>(modelPath);
            
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
                tokenizer_.get()
            );
            httpHandler_->post("/generate", [generateEndpoint](const HttpRequest& req) {
                return generateEndpoint->handle(req);
            });
            
            // 启动服务器线程
            serverThread_ = std::thread([&]() {
                std::cout << "[TEST] Starting HTTP server..." << std::endl;
                DrogonServer::init("0.0.0.0", port_, httpHandler_.get());
                DrogonServer::start();
                std::cout << "[TEST] HTTP server stopped" << std::endl;
            });
            
            // 等待服务器启动
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << "[TEST] Server setup completed" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[TEST] Setup failed: " << e.what() << std::endl;
            FAIL() << "Server setup failed: " << e.what();
        }
    }
    
    static void TearDownTestSuite() {
        std::cout << "[TEST] Tearing down test server..." << std::endl;
        
        try {
            // 停止服务器
            DrogonServer::stop();
            
            // 等待服务器线程结束
            if (serverThread_.joinable()) {
                serverThread_.join();
            }
            
            // 清理资源
            httpHandler_.reset();
            scheduler_.reset();
            modelExecutor_.reset();
            tokenizer_.reset();
            
            std::cout << "[TEST] Teardown completed" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[TEST] Teardown failed: " << e.what() << std::endl;
            FAIL() << "Server teardown failed: " << e.what();
        }
    }
    
    // 用于合并字符串向量
    std::string join(const std::vector<std::string>& vec, const std::string& delimiter) {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            result += vec[i];
            if (i != vec.size() - 1) {
                result += delimiter;
            }
        }
        return result;
    }
};

// 静态成员初始化
std::unique_ptr<Scheduler> E2EScenarios::scheduler_;
std::unique_ptr<ModelExecutor> E2EScenarios::modelExecutor_;
std::unique_ptr<Tokenizer> E2EScenarios::tokenizer_;
std::unique_ptr<HttpHandler> E2EScenarios::httpHandler_;
std::thread E2EScenarios::serverThread_;
std::string E2EScenarios::baseUrl_;
int E2EScenarios::port_;

/**
 * @brief P5.1.1: 单轮问答场景 - 事实问答
 */
TEST_F(E2EScenarios, FactualQuestions) {
    HttpClient client;
    
    std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
        {"What is the capital of China?", {"Beijing", "北京"}},
        {"Who invented the telephone?", {"Bell", "Alexander"}},
        {"What is the largest planet in our solar system?", {"Jupiter", "木星"}},
        {"When did World War II end?", {"1945"}},
        {"What is the speed of light?", {"300000", "3×10^8", "299792458"}}
    };
    
    int passed = 0;
    int total = test_cases.size();
    
    for (const auto& [question, expected_keywords] : test_cases) {
        Json::Value request;
        Json::Value messages;
        Json::Value userMessage;
        userMessage["role"] = "user";
        userMessage["content"] = question;
        messages.append(userMessage);
        
        request["model"] = "qwen2-0.5b";
        request["messages"] = messages;
        request["max_tokens"] = 100;
        request["temperature"] = 0.3;
        
        Json::FastWriter writer;
        std::string requestData = writer.write(request);
        
        auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
        ASSERT_EQ(httpCode, 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response, result));
        
        std::string answer = result["choices"][0]["message"]["content"].asString();
        
        std::cout << "Q: " << question << std::endl;
        std::cout << "A: " << answer << std::endl;
        
        bool correct = false;
        for (const auto& keyword : expected_keywords) {
            if (answer.find(keyword) != std::string::npos) {
                correct = true;
                break;
            }
        }
        
        if (correct) {
            passed++;
            std::cout << "✅ PASS" << std::endl;
        } else {
            std::cout << "❌ FAIL (expected keywords: " << join(expected_keywords, ", ") << ")" << std::endl;
        }
        std::cout << "---" << std::endl;
    }
    
    double accuracy = static_cast<double>(passed) / total;
    std::cout << "Factual QA Accuracy: " << accuracy * 100 << "% (" << passed << "/" << total << ")" << std::endl;
    
    EXPECT_GE(accuracy, 0.6);
}

/**
 * @brief P5.1.1: 单轮问答场景 - 推理问答
 */
TEST_F(E2EScenarios, ReasoningQuestions) {
    HttpClient client;
    
    std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
        {
            "If A is taller than B, and B is taller than C, who is the tallest?",
            {"A"}
        },
        {
            "If all roses are flowers, and some flowers fade quickly, can we conclude all roses fade quickly?",
            {"No", "cannot", "not necessarily"}
        },
        {
            "A train leaves Station A at 60 km/h, and another train leaves Station B (120km away) at 40 km/h towards each other. When do they meet?",
            {"1.2", "72", "minutes"}
        }
    };
    
    int passed = 0;
    int total = test_cases.size();
    
    for (const auto& [question, expected_indicators] : test_cases) {
        Json::Value request;
        Json::Value messages;
        Json::Value userMessage;
        userMessage["role"] = "user";
        userMessage["content"] = question;
        messages.append(userMessage);
        
        request["model"] = "qwen2-0.5b";
        request["messages"] = messages;
        request["max_tokens"] = 200;
        request["temperature"] = 0.5;
        
        Json::FastWriter writer;
        std::string requestData = writer.write(request);
        
        auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
        ASSERT_EQ(httpCode, 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response, result));
        
        std::string answer = result["choices"][0]["message"]["content"].asString();
        
        std::cout << "Q: " << question << std::endl;
        std::cout << "A: " << answer << std::endl;
        
        bool correct = false;
        for (const auto& indicator : expected_indicators) {
            if (answer.find(indicator) != std::string::npos) {
                correct = true;
                break;
            }
        }
        
        if (correct) {
            passed++;
            std::cout << "✅ PASS" << std::endl;
        } else {
            std::cout << "❌ FAIL" << std::endl;
        }
        std::cout << "---" << std::endl;
    }
    
    double accuracy = static_cast<double>(passed) / total;
    std::cout << "Reasoning QA Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    EXPECT_GE(accuracy, 0.4);
}

/**
 * @brief P5.1.1: 单轮问答场景 - 数学问答
 */
TEST_F(E2EScenarios, MathQuestions) {
    HttpClient client;
    
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"What is 15 + 27?", "42"},
        {"What is 8 × 7?", "56"},
        {"What is 100 - 37?", "63"},
        {"What is 144 ÷ 12?", "12"},
        {"What is 2^10?", "1024"}
    };
    
    int passed = 0;
    
    for (const auto& [question, expected_answer] : test_cases) {
        Json::Value request;
        Json::Value messages;
        Json::Value userMessage;
        userMessage["role"] = "user";
        userMessage["content"] = question;
        messages.append(userMessage);
        
        request["model"] = "qwen2-0.5b";
        request["messages"] = messages;
        request["max_tokens"] = 50;
        request["temperature"] = 0.1;
        
        Json::FastWriter writer;
        std::string requestData = writer.write(request);
        
        auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
        ASSERT_EQ(httpCode, 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response, result));
        
        std::string answer = result["choices"][0]["message"]["content"].asString();
        
        std::cout << "Q: " << question << std::endl;
        std::cout << "A: " << answer << std::endl;
        
        if (answer.find(expected_answer) != std::string::npos) {
            passed++;
            std::cout << "✅ PASS" << std::endl;
        } else {
            std::cout << "❌ FAIL (expected: " << expected_answer << ")" << std::endl;
        }
        std::cout << "---" << std::endl;
    }
    
    double accuracy = static_cast<double>(passed) / test_cases.size();
    std::cout << "Math QA Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    EXPECT_GE(accuracy, 0.7);
}

/**
 * @brief P5.1.2: 多轮对话场景 - 上下文保持
 */
TEST_F(E2EScenarios, ContextRetention) {
    HttpClient client;
    
    Json::Value request1;
    request1["model"] = "qwen2-0.5b";
    request1["messages"] = Json::Value(Json::arrayValue);
    
    Json::Value userMsg1;
    userMsg1["role"] = "user";
    userMsg1["content"] = "My name is Alice and I'm 25 years old.";
    request1["messages"].append(userMsg1);
    request1["max_tokens"] = 50;
    
    Json::FastWriter writer;
    std::string requestData1 = writer.write(request1);
    
    auto [httpCode1, response1] = client.post(baseUrl_ + "/generate", requestData1);
    ASSERT_EQ(httpCode1, 200);
    
    Json::Reader reader;
    Json::Value result1;
    ASSERT_TRUE(reader.parse(response1, result1));
    
    std::string answer1 = result1["choices"][0]["message"]["content"].asString();
    
    std::cout << "Round 1 - User: My name is Alice and I'm 25 years old." << std::endl;
    std::cout << "Round 1 - Assistant: " << answer1 << std::endl;
    
    Json::Value request2;
    request2["model"] = "qwen2-0.5b";
    request2["messages"] = Json::Value(Json::arrayValue);
    
    // Repeat the first exchange
    Json::Value userMsg1Repeat;
    userMsg1Repeat["role"] = "user";
    userMsg1Repeat["content"] = "My name is Alice and I'm 25 years old.";
    request2["messages"].append(userMsg1Repeat);
    
    Json::Value assistantMsg1;
    assistantMsg1["role"] = "assistant";
    assistantMsg1["content"] = answer1;
    request2["messages"].append(assistantMsg1);
    
    Json::Value userMsg2;
    userMsg2["role"] = "user";
    userMsg2["content"] = "What's my name?";
    request2["messages"].append(userMsg2);
    request2["max_tokens"] = 20;
    
    std::string requestData2 = writer.write(request2);
    auto [httpCode2, response2] = client.post(baseUrl_ + "/generate", requestData2);
    auto result2 = Json::Value();
    ASSERT_TRUE(reader.parse(response2, result2));
    
    std::string answer2 = result2["choices"][0]["message"]["content"].asString();
    
    std::cout << "Round 2 - User: What's my name?" << std::endl;
    std::cout << "Round 2 - Assistant: " << answer2 << std::endl;
    
    EXPECT_TRUE(answer2.find("Alice") != std::string::npos);
    
    Json::Value request3;
    request3["model"] = "qwen2-0.5b";
    request3["messages"] = Json::Value(Json::arrayValue);
    
    // Repeat the first two exchanges
    Json::Value userMsg1Repeat3;
    userMsg1Repeat3["role"] = "user";
    userMsg1Repeat3["content"] = "My name is Alice and I'm 25 years old.";
    request3["messages"].append(userMsg1Repeat3);
    
    Json::Value assistantMsg1Repeat3;
    assistantMsg1Repeat3["role"] = "assistant";
    assistantMsg1Repeat3["content"] = answer1;
    request3["messages"].append(assistantMsg1Repeat3);
    
    Json::Value userMsg2Repeat3;
    userMsg2Repeat3["role"] = "user";
    userMsg2Repeat3["content"] = "What's my name?";
    request3["messages"].append(userMsg2Repeat3);
    
    Json::Value assistantMsg2;
    assistantMsg2["role"] = "assistant";
    assistantMsg2["content"] = answer2;
    request3["messages"].append(assistantMsg2);
    
    Json::Value userMsg3;
    userMsg3["role"] = "user";
    userMsg3["content"] = "How old am I?";
    request3["messages"].append(userMsg3);
    request3["max_tokens"] = 20;
    
    std::string requestData3 = writer.write(request3);
    auto [httpCode3, response3] = client.post(baseUrl_ + "/generate", requestData3);
    auto result3 = Json::Value();
    ASSERT_TRUE(reader.parse(response3, result3));
    
    std::string answer3 = result3["choices"][0]["message"]["content"].asString();
    
    std::cout << "Round 3 - User: How old am I?" << std::endl;
    std::cout << "Round 3 - Assistant: " << answer3 << std::endl;
    
    EXPECT_TRUE(answer3.find("25") != std::string::npos);
}

/**
 * @brief P5.1.2: 多轮对话场景 - 指代消解
 */
TEST_F(E2EScenarios, Coreference) {
    HttpClient client;
    
    Json::Value request1;
    request1["model"] = "qwen2-0.5b";
    request1["messages"] = Json::Value(Json::arrayValue);
    
    Json::Value userMsg1;
    userMsg1["role"] = "user";
    userMsg1["content"] = "Tell me about Python programming language.";
    request1["messages"].append(userMsg1);
    request1["max_tokens"] = 100;
    
    Json::FastWriter writer;
    std::string requestData1 = writer.write(request1);
    
    auto [httpCode1, response1] = client.post(baseUrl_ + "/generate", requestData1);
    ASSERT_EQ(httpCode1, 200);
    
    Json::Reader reader;
    Json::Value result1;
    ASSERT_TRUE(reader.parse(response1, result1));
    
    std::string answer1 = result1["choices"][0]["message"]["content"].asString();
    
    Json::Value request2;
    request2["model"] = "qwen2-0.5b";
    request2["messages"] = Json::Value(Json::arrayValue);
    
    // Repeat the first exchange
    Json::Value userMsg1Repeat;
    userMsg1Repeat["role"] = "user";
    userMsg1Repeat["content"] = "Tell me about Python programming language.";
    request2["messages"].append(userMsg1Repeat);
    
    Json::Value assistantMsg1;
    assistantMsg1["role"] = "assistant";
    assistantMsg1["content"] = answer1;
    request2["messages"].append(assistantMsg1);
    
    Json::Value userMsg2;
    userMsg2["role"] = "user";
    userMsg2["content"] = "What is it mainly used for?";
    request2["messages"].append(userMsg2);
    request2["max_tokens"] = 100;
    
    std::string requestData2 = writer.write(request2);
    auto [httpCode2, response2] = client.post(baseUrl_ + "/generate", requestData2);
    ASSERT_EQ(httpCode2, 200);
    
    Json::Value result2;
    ASSERT_TRUE(reader.parse(response2, result2));
    
    std::string answer2 = result2["choices"][0]["message"]["content"].asString();
    
    std::cout << "User: What is it mainly used for?" << std::endl;
    std::cout << "Assistant: " << answer2 << std::endl;
    
    bool relevant = answer2.find("Python") != std::string::npos ||
                   answer2.find("programming") != std::string::npos ||
                   answer2.find("web") != std::string::npos ||
                   answer2.find("data") != std::string::npos ||
                   answer2.find("AI") != std::string::npos;
    
    EXPECT_TRUE(relevant);
}

/**
 * @brief P5.1.2: 多轮对话场景 - 话题切换
 */
TEST_F(E2EScenarios, TopicSwitch) {
    HttpClient client;
    
    Json::Value request1;
    request1["model"] = "qwen2-0.5b";
    request1["messages"] = Json::Value(Json::arrayValue);
    
    Json::Value userMsg1;
    userMsg1["role"] = "user";
    userMsg1["content"] = "What's the weather like today?";
    request1["messages"].append(userMsg1);
    request1["max_tokens"] = 50;
    
    Json::FastWriter writer;
    std::string requestData1 = writer.write(request1);
    
    auto [httpCode1, response1] = client.post(baseUrl_ + "/generate", requestData1);
    ASSERT_EQ(httpCode1, 200);
    
    Json::Reader reader;
    Json::Value result1;
    ASSERT_TRUE(reader.parse(response1, result1));
    
    Json::Value request2;
    request2["model"] = "qwen2-0.5b";
    request2["messages"] = Json::Value(Json::arrayValue);
    
    // Repeat the first exchange
    Json::Value userMsg1Repeat;
    userMsg1Repeat["role"] = "user";
    userMsg1Repeat["content"] = "What's the weather like today?";
    request2["messages"].append(userMsg1Repeat);
    
    Json::Value assistantMsg1;
    assistantMsg1["role"] = "assistant";
    assistantMsg1["content"] = result1["choices"][0]["message"]["content"].asString();
    request2["messages"].append(assistantMsg1);
    
    Json::Value userMsg2;
    userMsg2["role"] = "user";
    userMsg2["content"] = "By the way, can you help me write a Python function?";
    request2["messages"].append(userMsg2);
    
    request2["max_tokens"] = 100;
    
    std::string requestData2 = writer.write(request2);
    auto [httpCode2, response2] = client.post(baseUrl_ + "/generate", requestData2);
    ASSERT_EQ(httpCode2, 200);
    
    Json::Value result2;
    ASSERT_TRUE(reader.parse(response2, result2));
    
    std::string answer2 = result2["choices"][0]["message"]["content"].asString();
    
    std::cout << "Assistant (after topic switch): " << answer2 << std::endl;
    
    bool handled = answer2.find("Python") != std::string::npos ||
                   answer2.find("function") != std::string::npos ||
                   answer2.find("def") != std::string::npos ||
                   answer2.find("code") != std::string::npos;
    
    EXPECT_TRUE(handled);
}

/**
 * @brief P5.1.3: 专业任务场景 - 代码生成
 */
TEST_F(E2EScenarios, CodeGeneration) {
    HttpClient client;
    
    std::vector<std::string> prompts = {
        "Write a Python function to calculate Fibonacci numbers.",
        "Write a JavaScript function to check if a string is a palindrome.",
        "Write a C++ function to sort an array using quicksort."
    };
    
    for (const auto& prompt : prompts) {
        Json::Value request;
        Json::Value messages;
        Json::Value userMsg;
        userMsg["role"] = "user";
        userMsg["content"] = prompt;
        messages.append(userMsg);
        
        request["model"] = "qwen2-0.5b";
        request["messages"] = messages;
        request["max_tokens"] = 300;
        
        Json::FastWriter writer;
        std::string requestData = writer.write(request);
        
        auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
        ASSERT_EQ(httpCode, 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response, result));
        
        std::string code = result["choices"][0]["message"]["content"].asString();
        
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Generated code:\n" << code << std::endl;
        std::cout << "---" << std::endl;
        
        bool has_function = code.find("def ") != std::string::npos ||
                           code.find("function ") != std::string::npos ||
                           code.find("void ") != std::string::npos;
        
        EXPECT_TRUE(has_function);
    }
}

/**
 * @brief P5.1.3: 专业任务场景 - 文本摘要
 */
TEST_F(E2EScenarios, TextSummarization) {
    HttpClient client;
    
    std::string long_text = R"(
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals. The term "artificial intelligence" 
        had previously been used to describe machines that mimic and display "human" 
        cognitive skills that are associated with the human mind, such as "learning" 
        and "problem-solving". This definition has since been rejected by major AI 
        researchers who now describe AI in terms of rationality and acting rationally, 
        which does not limit how intelligence can be articulated.
    )";
    
    Json::Value request;
    Json::Value messages;
    Json::Value userMsg;
    userMsg["role"] = "user";
    userMsg["content"] = "Please summarize the following text in 2-3 sentences: " + long_text;
    messages.append(userMsg);
    
    request["model"] = "qwen2-0.5b";
    request["messages"] = messages;
    request["max_tokens"] = 150;
    
    Json::FastWriter writer;
    std::string requestData = writer.write(request);
    
    auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
    ASSERT_EQ(httpCode, 200);
    
    Json::Reader reader;
    Json::Value result;
    ASSERT_TRUE(reader.parse(response, result));
    
    std::string summary = result["choices"][0]["message"]["content"].asString();
    
    std::cout << "Original length: " << long_text.length() << std::endl;
    std::cout << "Summary: " << summary << std::endl;
    std::cout << "Summary length: " << summary.length() << std::endl;
    
    EXPECT_LT(summary.length(), long_text.length() * 0.5);
    
    bool has_keywords = summary.find("AI") != std::string::npos ||
                       summary.find("artificial intelligence") != std::string::npos ||
                       summary.find("intelligence") != std::string::npos;
    
    EXPECT_TRUE(has_keywords);
}

/**
 * @brief P5.1.3: 专业任务场景 - 翻译
 */
TEST_F(E2EScenarios, Translation) {
    HttpClient client;
    
    Json::Value requestEnToZh;
    Json::Value messagesEnToZh;
    Json::Value userMsgEnToZh;
    userMsgEnToZh["role"] = "user";
    userMsgEnToZh["content"] = "Translate to Chinese: Artificial Intelligence is changing the world.";
    messagesEnToZh.append(userMsgEnToZh);
    
    requestEnToZh["model"] = "qwen2-0.5b";
    requestEnToZh["messages"] = messagesEnToZh;
    requestEnToZh["max_tokens"] = 100;
    
    Json::FastWriter writer;
    std::string requestDataEnToZh = writer.write(requestEnToZh);
    
    auto [httpCodeEnToZh, responseEnToZh] = client.post(baseUrl_ + "/generate", requestDataEnToZh);
    ASSERT_EQ(httpCodeEnToZh, 200);
    
    Json::Reader reader;
    Json::Value resultEnToZh;
    ASSERT_TRUE(reader.parse(responseEnToZh, resultEnToZh));
    
    std::string translationZh = resultEnToZh["choices"][0]["message"]["content"].asString();
    
    std::cout << "EN->ZH: " << translationZh << std::endl;
    
    bool has_chinese = std::any_of(translationZh.begin(), translationZh.end(), 
        [](char c) { return static_cast<unsigned char>(c) > 127; });
    
    EXPECT_TRUE(has_chinese);
    
    Json::Value requestZhToEn;
    Json::Value messagesZhToEn;
    Json::Value userMsgZhToEn;
    userMsgZhToEn["role"] = "user";
    userMsgZhToEn["content"] = "翻译成英文：人工智能正在改变世界。";
    messagesZhToEn.append(userMsgZhToEn);
    
    requestZhToEn["model"] = "qwen2-0.5b";
    requestZhToEn["messages"] = messagesZhToEn;
    requestZhToEn["max_tokens"] = 100;
    
    std::string requestDataZhToEn = writer.write(requestZhToEn);
    auto [httpCodeZhToEn, responseZhToEn] = client.post(baseUrl_ + "/generate", requestDataZhToEn);
    ASSERT_EQ(httpCodeZhToEn, 200);
    
    Json::Value resultZhToEn;
    ASSERT_TRUE(reader.parse(responseZhToEn, resultZhToEn));
    
    std::string translationEn = resultZhToEn["choices"][0]["message"]["content"].asString();
    
    std::cout << "ZH->EN: " << translationEn << std::endl;
    
    bool has_keywords = translationEn.find("AI") != std::string::npos ||
                       translationEn.find("artificial") != std::string::npos ||
                       translationEn.find("intelligence") != std::string::npos ||
                       translationEn.find("world") != std::string::npos;
    
    EXPECT_TRUE(has_keywords);
}

/**
 * @brief P5.1.4: 质量评估
 */
TEST_F(E2EScenarios, QualityEvaluation) {
    HttpClient client;
    
    struct QualityScore {
        double accuracy = 0.0;
        double fluency = 0.0;
        double relevance = 0.0;
        double completeness = 0.0;
    };
    
    std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> test_set = {
        {
            "Question", 
            "What is machine learning?",
            {"machine", "learn", "data", "algorithm", "model"}
        },
        {
            "Question",
            "Explain quantum computing.",
            {"quantum", "qubit", "superposition", "computing"}
        },
        {
            "Question",
            "What is programming?",
            {"program", "code", "computer", "language"}
        },
        {
            "Question",
            "Explain artificial intelligence.",
            {"AI", "artificial", "intelligence", "machine", "learn"}
        }
    };
    
    std::vector<QualityScore> scores;
    
    for (const auto& [type, question, keywords] : test_set) {
        Json::Value request;
        Json::Value messages;
        Json::Value userMsg;
        userMsg["role"] = "user";
        userMsg["content"] = question;
        messages.append(userMsg);
        
        request["model"] = "qwen2-0.5b";
        request["messages"] = messages;
        request["max_tokens"] = 200;
        
        Json::FastWriter writer;
        std::string requestData = writer.write(request);
        
        auto [httpCode, response] = client.post(baseUrl_ + "/generate", requestData);
        ASSERT_EQ(httpCode, 200);
        
        Json::Reader reader;
        Json::Value result;
        ASSERT_TRUE(reader.parse(response, result));
        
        std::string answer = result["choices"][0]["message"]["content"].asString();
        
        QualityScore score;
        
        int keywords_found = 0;
        for (const auto& keyword : keywords) {
            if (answer.find(keyword) != std::string::npos) {
                keywords_found++;
            }
        }
        score.accuracy = static_cast<double>(keywords_found) / keywords.size();
        
        bool has_punctuation = answer.find(".") != std::string::npos || 
                               answer.find("。") != std::string::npos;
        bool reasonable_length = answer.length() > 20 && answer.length() < 1000;
        score.fluency = (has_punctuation && reasonable_length) ? 1.0 : 0.5;
        
        score.relevance = (answer.length() > 30) ? 1.0 : 0.5;
        score.completeness = (answer.length() > 50) ? 1.0 : 0.5;
        
        scores.push_back(score);
        
        std::cout << "Question: " << question << std::endl;
        std::cout << "Answer: " << answer << std::endl;
        std::cout << "Scores - Accuracy: " << score.accuracy 
                  << ", Fluency: " << score.fluency
                  << ", Relevance: " << score.relevance
                  << ", Completeness: " << score.completeness << std::endl;
        std::cout << "---" << std::endl;
    }
    
    double avg_accuracy = 0.0, avg_fluency = 0.0, avg_relevance = 0.0, avg_completeness = 0.0;
    
    for (const auto& score : scores) {
        avg_accuracy += score.accuracy;
        avg_fluency += score.fluency;
        avg_relevance += score.relevance;
        avg_completeness += score.completeness;
    }
    
    int n = scores.size();
    avg_accuracy /= n;
    avg_fluency /= n;
    avg_relevance /= n;
    avg_completeness /= n;
    
    double overall_score = (avg_accuracy + avg_fluency + avg_relevance + avg_completeness) / 4.0;
    
    std::cout << "===== Quality Evaluation Results =====" << std::endl;
    std::cout << "Average Accuracy: " << avg_accuracy * 5 << " / 5.0" << std::endl;
    std::cout << "Average Fluency: " << avg_fluency * 5 << " / 5.0" << std::endl;
    std::cout << "Average Relevance: " << avg_relevance * 5 << " / 5.0" << std::endl;
    std::cout << "Average Completeness: " << avg_completeness * 5 << " / 5.0" << std::endl;
    std::cout << "Overall Score: " << overall_score * 5 << " / 5.0" << std::endl;
    
    EXPECT_GT(overall_score * 5, 4.0);
}

} // namespace test
} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}