#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include "cllm/common/json.hpp"

using json = nlohmann::json;

// 回调函数用于处理HTTP响应
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// 测试健康检查端点
bool testHealthEndpoint() {
    std::cout << "Testing /health endpoint..." << std::endl;
    
    CURL* curl;
    CURLcode res;
    std::string response_data;
    
    curl = curl_easy_init();
    if(curl) {
        std::string url = "http://localhost:8080/health";
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10秒超时
        
        res = curl_easy_perform(curl);
        
        curl_easy_cleanup(curl);
        
        if(res != CURLE_OK) {
            std::cout << "Failed to perform request: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
        
        std::cout << "Health response: " << response_data << std::endl;
        
        // 解析JSON响应
        try {
            json response = json::parse(response_data);
            if (response.contains("status") && response["status"] == "healthy") {
                std::cout << "Health check passed!" << std::endl;
                return true;
            } else {
                std::cout << "Health check failed - unexpected response" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to parse JSON response: " << e.what() << std::endl;
            return false;
        }
    }
    
    return false;
}

// 测试文本生成端点
bool testGenerateEndpoint() {
    std::cout << "\nTesting /generate endpoint..." << std::endl;
    
    CURL* curl;
    CURLcode res;
    std::string response_data;
    
    curl = curl_easy_init();
    if(curl) {
        std::string url = "http://localhost:8080/generate";
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // 创建请求JSON
        json request_json;
        request_json["prompt"] = "Hello, how are you?";
        request_json["max_tokens"] = 50;
        request_json["temperature"] = 0.7;
        
        std::string request_body = request_json.dump();
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L); // 30秒超时
        
        res = curl_easy_perform(curl);
        
        if (headers) {
            curl_slist_free_all(headers);
        }
        curl_easy_cleanup(curl);
        
        if(res != CURLE_OK) {
            std::cout << "Failed to perform request: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
        
        std::cout << "Generate response: " << response_data << std::endl;
        
        // 解析JSON响应
        try {
            json response = json::parse(response_data);
            if (response.contains("generated_text")) {
                std::cout << "Generate test passed! Generated text length: " 
                         << response["generated_text"].get<std::string>().length() << std::endl;
                return true;
            } else {
                std::cout << "Generate test failed - no generated_text in response" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to parse JSON response: " << e.what() << std::endl;
            return false;
        }
    }
    
    return false;
}

// 测试文本编码端点
bool testEncodeEndpoint() {
    std::cout << "\nTesting /encode endpoint..." << std::endl;
    
    CURL* curl;
    CURLcode res;
    std::string response_data;
    
    curl = curl_easy_init();
    if(curl) {
        std::string url = "http://localhost:8080/encode";
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // 创建请求JSON
        json request_json;
        request_json["text"] = "Hello, world!";
        
        std::string request_body = request_json.dump();
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10秒超时
        
        res = curl_easy_perform(curl);
        
        if (headers) {
            curl_slist_free_all(headers);
        }
        curl_easy_cleanup(curl);
        
        if(res != CURLE_OK) {
            std::cout << "Failed to perform request: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
        
        std::cout << "Encode response: " << response_data << std::endl;
        
        // 解析JSON响应
        try {
            json response = json::parse(response_data);
            if (response.contains("token_ids") && response["token_ids"].is_array()) {
                std::cout << "Encode test passed! Token count: " 
                         << response["token_ids"].size() << std::endl;
                return true;
            } else {
                std::cout << "Encode test failed - invalid response format" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to parse JSON response: " << e.what() << std::endl;
            return false;
        }
    }
    
    return false;
}

int main() {
    std::cout << "Starting cLLM API Integration Tests..." << std::endl;
    
    // 初始化curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // 等待服务器启动
    std::cout << "Waiting for server to start..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    bool all_tests_passed = true;
    
    // 执行健康检查测试
    if (!testHealthEndpoint()) {
        all_tests_passed = false;
        std::cout << "Health endpoint test FAILED" << std::endl;
    } else {
        std::cout << "Health endpoint test PASSED" << std::endl;
    }
    
    // 执行编码测试
    if (!testEncodeEndpoint()) {
        all_tests_passed = false;
        std::cout << "Encode endpoint test FAILED" << std::endl;
    } else {
        std::cout << "Encode endpoint test PASSED" << std::endl;
    }
    
    // 执行生成测试
    if (!testGenerateEndpoint()) {
        all_tests_passed = false;
        std::cout << "Generate endpoint test FAILED" << std::endl;
    } else {
        std::cout << "Generate endpoint test PASSED" << std::endl;
    }
    
    // 输出最终结果
    std::cout << "\n=== API Integration Test Summary ===" << std::endl;
    if (all_tests_passed) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED!" << std::endl;
    }
    
    // 清理curl
    curl_global_cleanup();
    
    return all_tests_passed ? 0 : 1;
}