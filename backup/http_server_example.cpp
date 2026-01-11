#include <iostream>
#include <cllm/http/server.h>
#include <cllm/http/handler.h>
#include <cllm/http/health_endpoint.h>
#include <thread>
#include <chrono>

using namespace cllm;

int main() {
    std::cout << "cLLM HTTP Server Example" << std::endl;
    std::cout << "=========================" << std::endl;

    HttpServer server("0.0.0.0", 8080);
    HttpHandler handler;

    // 注册健康检查API
    handler.get("/health", [](const HttpRequest& request) {
        HttpResponse response;
        response.setStatusCode(200);
        response.setBody("{\"status\":\"healthy\",\"model_loaded\":true}");
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

    // 注册生成API
    handler.post("/generate", [](const HttpRequest& request) {
        HttpResponse response;
        response.setStatusCode(200);
        response.setBody(R"({"text":"This is a sample response from cLLM server."})";