#pragma once

#include "cllm/http/handler.h"
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <map>

#ifdef __linux__
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>
#include <sys/time.h>
#endif

namespace cllm {

/**
 * @brief 高性能HTTP服务器实现
 * 
 * 基于原生socket实现，零依赖，高性能
 * 支持HTTP/1.1 Keep-Alive，多线程并发处理
 */
class HttpServer {
public:
    /**
     * @brief 初始化服务器
     * @param host 监听地址
     * @param port 监听端口
     * @param handler HTTP请求处理器
     */
    static void init(const std::string& host, int port, HttpHandler* handler);
    
    /**
     * @brief 启动服务器
     */
    static void start();
    
    /**
     * @brief 停止服务器
     */
    static void stop();
    
    /**
     * @brief 检查服务器是否运行中
     */
    static bool isRunning();

private:
    HttpServer() = default;
    ~HttpServer() = default;
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    
    // 内部实现
    void run();
    void eventLoop(int workerId);
    void handleConnection(int clientFd);
    void handleReadEvent(int clientFd);
    void handleWriteEvent(int clientFd);
    
    // epoll/kqueue相关
    void setupEventLoop();
    void addEvent(int fd, uint32_t events);
    void modEvent(int fd, uint32_t events);
    void delEvent(int fd);
    
    // 连接状态管理
    struct ConnectionState {
        std::string readBuffer;
        std::string writeBuffer;
        HttpRequest request;
        HttpResponse response;
        enum { READING_HEADER, READING_BODY, WRITING } state;
        size_t contentLength;
        bool keepAlive;
        
        ConnectionState() : state(READING_HEADER), contentLength(0), keepAlive(true) {}
    };
    
    // 静态成员
    static HttpServer* instance_;
    static std::mutex instance_mutex_;
    
    // 服务器配置
    std::string host_;
    int port_;
    HttpHandler* handler_;
    int serverFd_;
    
    // 事件循环
    int epollFd_;  // Linux: epoll fd, macOS: kqueue fd
    std::vector<int> epollFds_;  // 每个worker一个epoll/kqueue实例
    
    // 线程管理
    std::atomic<bool> running_;
    std::vector<std::thread> workerThreads_;
    unsigned int numThreads_;
    
    // 连接状态管理
    std::map<int, ConnectionState> connections_;
    std::mutex connectionsMutex_;
    static constexpr size_t MAX_CONNECTIONS = 1024;  // 最大连接数限制
};

} // namespace cllm
