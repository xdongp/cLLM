#include <iostream>
#include <cllm/http/server.h>
#include <cllm/http/handler.h>
#include <cllm/http/health_endpoint.h>
#include <thread>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>

using namespace cllm;

// Helper function to generate request ID
std::string generateRequestId() {
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_int_distribution<> distribution(0, 0xFFFF);
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::stringstream ss;
    ss << "req_" << timestamp << "_" << std::setw(4) << std::setfill('0') << std::hex << distribution(generator);
    return ss.str();
}

// Helper function to generate random text
std::string generateRandomText(size_t length) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distribution(0, charset.size() - 1);
    
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += charset[distribution(generator)];
    }
    return result;
}

int main() {
    std::cout << "cLLM HTTP Server Example" << std::endl;
    std::cout << "=========================" << std::endl;

    HttpServer server("0.0.0.0", 8080);
    HttpHandler handler;

    // 注册健康检查API
    handler.get("/health", [](const HttpRequest& request) {
        HttpResponse response;
        response.setStatusCode(200);
        response.setBody(R"({"status":"healthy","model_loaded":true})");
        response.setContentType("application/json");
        return response;
    });

    // 注册测试API
    handler.get("/test", [](const HttpRequest& request) {
        HttpResponse response;
        response.setStatusCode(200);
        response.setBody("cLLM HTTP Server is running!");
        return response;
    });

    // 注册生成API - POST /generate
    handler.post("/generate", [](const HttpRequest& request) {
        HttpResponse response;
        std::string body = request.getBody();
        
        // Check if this is a streaming request (account for possible spaces)
        bool isStream = (body.find("\"stream\":true") != std::string::npos || 
                         body.find("\"stream\": true") != std::string::npos);
        
        if (isStream) {
            // Handle streaming response
            response.setStatusCode(200);
            response.setContentType("text/event-stream");
            response.setHeader("Cache-Control", "no-cache");
            response.setHeader("Connection", "keep-alive");
            
            // Generate streaming response with multiple chunks
            std::stringstream ss;
            ss << "data: {\"id\":\"" << generateRequestId() << "\",\"object\":\"text_completion\",\"created\":1623456789,\"model\":\"test-model\",\"choices\":[{\"text\":\"Hello \",\"index\":0,\"logprobs\":null,\"finish_reason\":null}]}" << std::endl << std::endl;
            ss << "data: {\"id\":\"" << generateRequestId() << "\",\"object\":\"text_completion\",\"created\":1623456790,\"model\":\"test-model\",\"choices\":[{\"text\":\"world!\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"stop\"}]}" << std::endl << std::endl;
            ss << "data: [DONE]" << std::endl << std::endl;
            
            response.setBody(ss.str());
        } else {
            // Handle non-streaming response
            response.setStatusCode(200);
            response.setContentType("application/json");
            
            std::string text = "Hello world! " + generateRandomText(20);
            
            // Properly format the JSON response
            std::stringstream ss;
            ss << "{";
            ss << "\"id\":\"" << generateRequestId() << "\",";
            ss << "\"object\":\"text_completion\",";
            ss << "\"created\":1623456789,";
            ss << "\"model\":\"test-model\",";
            ss << "\"choices\":[{";
            ss << "\"text\":\"" << text << "\",";
            ss << "\"index\":0,";
            ss << "\"logprobs\":null,";
            ss << "\"finish_reason\":\"stop\"";
            ss << "}]";
            ss << "}";
            
            response.setBody(ss.str());
        }
        
        return response;
    });

    // 注册编码API - POST /encode
    handler.post("/encode", [](const HttpRequest& request) {
        HttpResponse response;
        std::string body = request.getBody();
        
        // Check if text parameter is present
        if (body.find("\"text\":") == std::string::npos) {
            response.setStatusCode(400);
            response.setBody(R"({"error":"Missing required parameter: text"})");
            response.setContentType("application/json");
            return response;
        }
        
        // Generate random embeddings
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<> distribution(-1.0, 1.0);
        
        std::stringstream embeddings;
        embeddings << "[";
        for (int i = 0; i < 1536; ++i) {
            if (i > 0) embeddings << ",";
            embeddings << distribution(generator);
        }
        embeddings << "]";
        
        response.setStatusCode(200);
        response.setContentType("application/json");
        response.setBody(
            R"({"object":"list","data":[{"object":"embedding","embedding":)" + embeddings.str() + 
            R"(,"index":0}],"model":"test-model","usage":{"prompt_tokens":10,"total_tokens":10}})");
        
        return response;
    });

    // 设置路由并启动服务器
    server.setHandler(&handler);
    
    std::cout << "Server starting on 0.0.0.0:8080..." << std::endl;
    std::cout << "Available endpoints:" << std::endl;
    std::cout << "  GET    /health    - Health check" << std::endl;
    std::cout << "  GET    /test      - Test endpoint" << std::endl;
    std::cout << "  POST   /generate  - Text generation" << std::endl;
    std::cout << "  POST   /encode    - Text encoding" << std::endl;
    
    // 启动服务器
    server.start();
    
    std::cout << "Server started successfully!" << std::endl;
    std::cout << "Press Ctrl+C to stop the server..." << std::endl;
    
    // 保持服务器运行
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
