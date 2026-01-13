#include <iostream>
#include <cllm/http/handler.h>
#include <cllm/http/request.h>
#include <cllm/http/response.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace cllm;

int main() {
    std::cout << "cLLM Generate API Test" << std::endl;
    std::cout << "=========================" << std::endl;

    HttpHandler handler;

    // 注册生成API - POST /generate
    handler.post("/generate", [](const HttpRequest& request) {
        HttpResponse response;
        std::string body = request.getBody();
        
        // 设置响应内容
        std::stringstream ss;
        ss << "{\"id\":\"test_req_123\",\"object\":\"text_completion\",\"created\":1623456789,\"model\":\"test-model\",\"choices\":[{\"text\":\"hello world this is a test response\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"stop\"}]}";
        
        response.setStatusCode(200);
        response.setContentType("application/json");
        response.setBody(ss.str());
        
        return response;
    });

    // 模拟请求
    std::string testBody = "{\"prompt\":\"hello\",\"max_tokens\":50,\"temperature\":0.7}";
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(testBody);
    request.setHeader("Content-Type", "application/json");

    // 处理请求
    HttpResponse response = handler.handleRequest(request);

    // 输出结果
    std::cout << "Request Body: " << testBody << std::endl;
    std::cout << "\nResponse Status Code: " << response.getStatusCode() << std::endl;
    std::cout << "Response Body: " << response.getBody() << std::endl;
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}