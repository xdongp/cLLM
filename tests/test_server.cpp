#include "cllm/http/http_server.h"
#include "cllm/http/handler.h"
#include <iostream>
#include <memory>

int main() {
    // 初始化模型执行器、分词器和采样器
    try {
        // 创建处理器实例
        auto handler = std::make_shared<cllm::HttpHandler>();
        
        // 初始化HTTP服务器
        cllm::HttpServer::init("127.0.0.1", 8080, handler.get());
        
        std::cout << "Starting HTTP server on 127.0.0.1:8080..." << std::endl;
        
        // 启动服务器
        cllm::HttpServer::start();
        
        // 等待用户输入停止服务器
        std::cout << "Press Enter to stop the server..." << std::endl;
        std::cin.get();
        
        // 停止服务器
        cllm::HttpServer::stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Error starting server: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
