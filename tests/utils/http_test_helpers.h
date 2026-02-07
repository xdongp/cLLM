#pragma once

#include <cllm/http/request.h>
#include <cllm/http/response.h>
#include <nlohmann/json.hpp>
#include <string>

namespace cllm {
namespace test {

/**
 * @brief HTTP测试辅助工具类
 * 提供创建和验证HTTP请求/响应的便捷方法
 */
class HttpTestHelpers {
public:
    /**
     * @brief 创建生成请求
     * @param prompt 提示文本
     * @param maxTokens 最大token数
     * @param temperature 温度参数
     * @param topP top_p参数
     * @param stream 是否流式输出
     * @return HTTP请求对象
     */
    static HttpRequest createGenerateRequest(
        const std::string& prompt,
        int maxTokens = 10,
        float temperature = 0.7f,
        float topP = 0.9f,
        bool stream = false) {
        
        nlohmann::json requestBody;
        requestBody["prompt"] = prompt;
        requestBody["max_tokens"] = maxTokens;
        requestBody["temperature"] = temperature;
        requestBody["top_p"] = topP;
        requestBody["stream"] = stream;
        
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        request.setBody(requestBody.dump());
        request.setHeader("Content-Type", "application/json");
        
        return request;
    }
    
    /**
     * @brief 创建tokenize请求
     * @param text 要tokenize的文本
     * @param addSpecialTokens 是否添加特殊token
     * @return HTTP请求对象
     */
    static HttpRequest createTokenizeRequest(
        const std::string& text,
        bool addSpecialTokens = true) {
        
        nlohmann::json requestBody;
        requestBody["text"] = text;
        requestBody["add_special_tokens"] = addSpecialTokens;
        
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/tokenize");
        request.setBody(requestBody.dump());
        request.setHeader("Content-Type", "application/json");
        
        return request;
    }
    
    /**
     * @brief 创建detokenize请求
     * @param tokens token序列
     * @param skipSpecialTokens 是否跳过特殊token
     * @return HTTP请求对象
     */
    static HttpRequest createDetokenizeRequest(
        const std::vector<int>& tokens,
        bool skipSpecialTokens = true) {
        
        nlohmann::json requestBody;
        requestBody["tokens"] = tokens;
        requestBody["skip_special_tokens"] = skipSpecialTokens;
        
        HttpRequest request;
        request.setMethod("POST");
        request.setPath("/detokenize");
        request.setBody(requestBody.dump());
        request.setHeader("Content-Type", "application/json");
        
        return request;
    }
    
    /**
     * @brief 验证成功响应
     * @param response HTTP响应对象
     * @param expectedStatusCode 期望的状态码
     * @return JSON响应体
     */
    static nlohmann::json verifySuccessResponse(
        const HttpResponse& response,
        int expectedStatusCode = 200) {
        
        if (response.getStatusCode() != expectedStatusCode) {
            throw std::runtime_error("Unexpected status code: " + 
                std::to_string(response.getStatusCode()));
        }
        
        std::string responseBody = response.getBody();
        if (responseBody.empty()) {
            throw std::runtime_error("Empty response body");
        }
        
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        
        if (!jsonResponse.contains("success")) {
            throw std::runtime_error("Response missing 'success' field");
        }
        
        if (!jsonResponse["success"].get<bool>()) {
            throw std::runtime_error("Request failed: " + 
                (jsonResponse.contains("error") ? 
                    jsonResponse["error"].get<std::string>() : "unknown error"));
        }
        
        return jsonResponse;
    }
    
    /**
     * @brief 验证错误响应
     * @param response HTTP响应对象
     * @param expectedStatusCode 期望的状态码
     * @return JSON响应体
     */
    static nlohmann::json verifyErrorResponse(
        const HttpResponse& response,
        int expectedStatusCode) {
        
        if (response.getStatusCode() != expectedStatusCode) {
            throw std::runtime_error("Unexpected status code: " + 
                std::to_string(response.getStatusCode()));
        }
        
        std::string responseBody = response.getBody();
        if (responseBody.empty()) {
            throw std::runtime_error("Empty response body");
        }
        
        nlohmann::json jsonResponse = nlohmann::json::parse(responseBody);
        
        if (!jsonResponse.contains("success")) {
            throw std::runtime_error("Response missing 'success' field");
        }
        
        if (jsonResponse["success"].get<bool>()) {
            throw std::runtime_error("Expected error response, but got success");
        }
        
        return jsonResponse;
    }
    
    /**
     * @brief 验证生成响应包含必需字段
     * @param jsonResponse JSON响应对象
     */
    static void verifyGenerateResponseFields(const nlohmann::json& jsonResponse) {
        if (!jsonResponse.contains("data")) {
            throw std::runtime_error("Response missing 'data' field");
        }
        
        auto data = jsonResponse["data"];
        
        if (!data.contains("id")) {
            throw std::runtime_error("Response data missing 'id' field");
        }
        
        if (!data.contains("text")) {
            throw std::runtime_error("Response data missing 'text' field");
        }
        
        if (!data.contains("response_time")) {
            throw std::runtime_error("Response data missing 'response_time' field");
        }
        
        if (!data.contains("tokens_per_second")) {
            throw std::runtime_error("Response data missing 'tokens_per_second' field");
        }
    }
    
    /**
     * @brief 验证并发限制响应
     * @param response HTTP响应对象
     */
    static void verifyConcurrencyLimitResponse(const HttpResponse& response) {
        if (response.getStatusCode() != 429) {
            throw std::runtime_error("Expected 429 status code for concurrency limit");
        }
        
        nlohmann::json jsonResponse = verifyErrorResponse(response, 429);
        
        if (!jsonResponse.contains("error")) {
            throw std::runtime_error("Concurrency limit response missing 'error' field");
        }
        
        if (jsonResponse["error"] != "Too many concurrent requests") {
            throw std::runtime_error("Unexpected error message for concurrency limit");
        }
        
        std::string retryAfter = response.getHeader("Retry-After");
        if (retryAfter.empty()) {
            throw std::runtime_error("Concurrency limit response missing 'Retry-After' header");
        }
    }
};

} // namespace test
} // namespace cllm
