#pragma once
#include <drogon/HttpController.h>
#include <drogon/HttpAppFramework.h>
#include "cllm/http/handler.h"

namespace cllm {

class DrogonServer : public drogon::HttpController<DrogonServer> {
public:
    METHOD_LIST_BEGIN
    METHOD_ADD(DrogonServer::health, "/health", drogon::Get);
    METHOD_ADD(DrogonServer::generate, "/generate", drogon::Post);
    METHOD_ADD(DrogonServer::generateStream, "/generate_stream", drogon::Post);
    METHOD_ADD(DrogonServer::encode, "/encode", drogon::Post);
    METHOD_LIST_END

    static void init(const std::string& host, int port, HttpHandler* handler);
    static void start();
    static void stop();

    // Handler methods
    void health(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    void generate(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    void generateStream(const drogon::HttpRequestPtr& req,
                       std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    void encode(const drogon::HttpRequestPtr& req,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback);

private:
    static std::mutex handler_mutex_;
    static HttpHandler* handler_;
    static std::string host_;
    static int port_;
};

} // namespace cllm