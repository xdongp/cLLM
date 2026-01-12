#include "cllm/http/drogon_server.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/config.h"
#include <mutex>
#include <thread>
#include <algorithm>

namespace cllm {

HttpHandler* DrogonServer::handler_ = nullptr;
std::string DrogonServer::host_;
int DrogonServer::port_;

std::mutex DrogonServer::handler_mutex_;

void DrogonServer::init(const std::string& host, int port, HttpHandler* handler) {
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        handler_ = handler;
        host_ = host;
        port_ = port;
    }

    // Drogon 默认线程数可能较小，/generate 为同步阻塞处理时会导致 /health 等请求卡住
    // 这里显式配置线程数，保证健康检查等轻量请求可并发响应
    unsigned int threads = static_cast<unsigned int>(cllm::Config::instance().serverNumThreads());
    const unsigned int minThreads = static_cast<unsigned int>(cllm::Config::instance().serverMinThreads());
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    if (threads == 0) {
        threads = hw;
    }
    threads = std::max(threads, minThreads);
    threads = std::max(threads, 2u);
    drogon::app().setThreadNum(static_cast<int>(threads));

    // 显式注册路由到 HttpHandler（避免依赖 Controller 自动注册失败导致 404）
    drogon::app().registerHandler(
        cllm::Config::instance().apiEndpointHealthPath(),
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            DrogonServer controller;
            controller.health(req, std::move(callback));
        },
        {drogon::Get});

    drogon::app().registerHandler(
        cllm::Config::instance().apiEndpointGeneratePath(),
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            DrogonServer controller;
            controller.generate(req, std::move(callback));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        cllm::Config::instance().apiEndpointGenerateStreamPath(),
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            DrogonServer controller;
            controller.generateStream(req, std::move(callback));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        cllm::Config::instance().apiEndpointEncodePath(),
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            DrogonServer controller;
            controller.encode(req, std::move(callback));
        },
        {drogon::Post});

    drogon::app().addListener(host, port);
}

void DrogonServer::start() {
    // ⚠️ listener 已在 init() 中注册，避免重复 addListener 导致 EADDRINUSE
    drogon::app().run();
}

void DrogonServer::stop() {
    drogon::app().quit();
}

void DrogonServer::health(const drogon::HttpRequestPtr& req,
                         std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    HttpHandler* handler_ptr;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        handler_ptr = handler_;
    }
    
    if (!handler_ptr) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }

    HttpRequest request;
    request.setMethod("GET");
    request.setPath(cllm::Config::instance().apiEndpointHealthPath());
    
    HttpResponse response = handler_ptr->handleRequest(request);
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(static_cast<drogon::HttpStatusCode>(response.getStatusCode()));
    resp->setBody(response.getBody());
    
    // 设置响应头
    for (const auto& header : response.getAllHeaders()) {
        resp->addHeader(header.first, header.second);
    }
    
    callback(resp);
}

void DrogonServer::generate(const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    HttpHandler* handler_ptr;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        handler_ptr = handler_;
    }
    
    if (!handler_ptr) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }

    HttpRequest request;
    request.setMethod("POST");
    request.setPath(cllm::Config::instance().apiEndpointGeneratePath());
    request.setBody(std::string(req->getBody()));
    
    HttpResponse response = handler_ptr->handleRequest(request);
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(static_cast<drogon::HttpStatusCode>(response.getStatusCode()));
    resp->setBody(response.getBody());
    
    // 设置响应头
    for (const auto& header : response.getAllHeaders()) {
        resp->addHeader(header.first, header.second);
    }
    
    callback(resp);
}

void DrogonServer::generateStream(const drogon::HttpRequestPtr& req,
                                 std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    HttpHandler* handler_ptr;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        handler_ptr = handler_;
    }
    
    if (!handler_ptr) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }

    // For streaming response, we'll just call the handler normally
    // Streaming implementation would be more complex and depend on the actual handler implementation
    HttpRequest request;
    request.setMethod("POST");
    request.setPath(cllm::Config::instance().apiEndpointGenerateStreamPath());
    request.setBody(std::string(req->getBody()));
    
    HttpResponse response = handler_ptr->handleRequest(request);
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(static_cast<drogon::HttpStatusCode>(response.getStatusCode()));
    resp->setBody(response.getBody());
    
    // 设置响应头
    for (const auto& header : response.getAllHeaders()) {
        resp->addHeader(header.first, header.second);
    }
    
    callback(resp);
}

void DrogonServer::encode(const drogon::HttpRequestPtr& req,
                         std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    HttpHandler* handler_ptr;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        handler_ptr = handler_;
    }
    
    if (!handler_ptr) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }

    HttpRequest request;
    request.setMethod("POST");
    request.setPath(cllm::Config::instance().apiEndpointEncodePath());
    request.setBody(std::string(req->getBody()));
    
    HttpResponse response = handler_ptr->handleRequest(request);
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(static_cast<drogon::HttpStatusCode>(response.getStatusCode()));
    resp->setBody(response.getBody());
    
    // 设置响应头
    for (const auto& header : response.getAllHeaders()) {
        resp->addHeader(header.first, header.second);
    }
    
    callback(resp);
}

} // namespace cllm