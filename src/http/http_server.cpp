#include "cllm/http/http_server.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/common/config.h"
#include "cllm/common/logger.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <errno.h>
#include <chrono>
#include <thread>
#include <atomic>

#ifdef __linux__
#include <sys/epoll.h>
#define EPOLL_IN EPOLLIN
#define EPOLL_OUT EPOLLOUT
#define EPOLL_ET EPOLLET
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>
#include <sys/time.h>
// 为macOS定义兼容常量
#define EPOLLIN 1
#define EPOLLOUT 4
#define EPOLLET 0x80000000
#define EPOLL_IN EPOLLIN
#define EPOLL_OUT EPOLLOUT
#define EPOLL_ET EPOLLET
#endif

namespace cllm {

// 静态成员初始化
HttpServer* HttpServer::instance_ = nullptr;
std::mutex HttpServer::instance_mutex_;

// HTTP解析辅助函数
static std::string readLine(int fd, std::string& buffer) {
    // 先检查buffer中是否已有完整行
    size_t pos = buffer.find("\r\n");
    if (pos != std::string::npos) {
        std::string line = buffer.substr(0, pos);
        buffer.erase(0, pos + 2);
        return line;
    }
    
    // 需要读取更多数据（最多读取64KB）
    const size_t maxBufferSize = 64 * 1024;
    if (buffer.length() > maxBufferSize) {
        return ""; // 缓冲区过大，可能是恶意请求
    }
    
    char buf[4096];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        if (n == 0) {
            // 连接关闭
            return "";
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 超时，返回空
            return "";
        }
        return "";
    }
    buf[n] = '\0';
    buffer += buf;
    
    pos = buffer.find("\r\n");
    if (pos != std::string::npos) {
        std::string line = buffer.substr(0, pos);
        buffer.erase(0, pos + 2);
        return line;
    }
    return "";
}

static bool parseRequestLine(const std::string& line, HttpRequest& request) {
    std::istringstream iss(line);
    std::string method, path, version;
    
    if (!(iss >> method >> path >> version)) {
        return false;
    }
    
    request.setMethod(method);
    
    // 解析路径和查询参数
    size_t queryPos = path.find('?');
    if (queryPos != std::string::npos) {
        std::string queryString = path.substr(queryPos + 1);
        path = path.substr(0, queryPos);
        
        // 解析查询参数
        std::istringstream qss(queryString);
        std::string param;
        while (std::getline(qss, param, '&')) {
            size_t eqPos = param.find('=');
            if (eqPos != std::string::npos) {
                std::string key = param.substr(0, eqPos);
                std::string value = param.substr(eqPos + 1);
                // URL解码（简化版）
                request.setQuery(key, value);
            }
        }
    }
    
    request.setPath(path);
    return true;
}

static bool parseHeaders(int fd, std::string& buffer, HttpRequest& request) {
    std::string line;
    while ((line = readLine(fd, buffer)) != "") {
        if (line.empty()) {
            break; // 空行表示头部结束
        }
        
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string name = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 1);
            
            // 去除前导空格
            size_t firstNonSpace = value.find_first_not_of(" \t");
            if (firstNonSpace != std::string::npos) {
                value = value.substr(firstNonSpace);
            }
            
            // 转换为小写（HTTP头部不区分大小写）
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            request.setHeader(name, value);
        }
    }
    return true;
}

static bool readBody(int fd, std::string& buffer, HttpRequest& request, size_t contentLength) {
    std::string body;
    
    // 先使用buffer中已有的数据
    if (buffer.length() > 0) {
        size_t toRead = std::min(buffer.length(), contentLength);
        body = buffer.substr(0, toRead);
        buffer.erase(0, toRead);
        contentLength -= toRead;
    }
    
    // 读取剩余数据
    while (contentLength > 0 && body.length() < contentLength) {
        char buf[4096];
        size_t toRead = std::min(sizeof(buf) - 1, contentLength - body.length());
        ssize_t n = recv(fd, buf, toRead, 0);
        if (n <= 0) {
            break;
        }
        body.append(buf, n);
    }
    
    request.setBody(body);
    return body.length() == contentLength;
}

static bool parseHttpRequest(int fd, HttpRequest& request) {
    std::string buffer;
    
    // 设置socket超时（5秒）
    // 增加超时时间以支持长时间运行的请求（生成50 tokens可能需要更长时间）
    struct timeval timeout;
    timeout.tv_sec = 60;  // 增加到60秒
    timeout.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // 解析请求行
    std::string requestLine = readLine(fd, buffer);
    if (requestLine.empty()) {
        return false;
    }
    
    if (!parseRequestLine(requestLine, request)) {
        return false;
    }
    
    // 解析头部
    if (!parseHeaders(fd, buffer, request)) {
        return false;
    }
    
    // 读取请求体（如果有）
    std::string contentLengthStr = request.getHeader("content-length");
    if (!contentLengthStr.empty()) {
        try {
            size_t contentLength = std::stoul(contentLengthStr);
            if (contentLength > 0 && contentLength < 10 * 1024 * 1024) { // 限制10MB
                if (!readBody(fd, buffer, request, contentLength)) {
                    return false;
                }
            }
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    return true;
}

// 🔥 优化：使用预分配字符串，减少ostringstream开销
static std::string buildHttpResponse(const HttpResponse& response) {
    // 预分配足够大的缓冲区（减少重新分配）
    std::string result;
    result.reserve(1024);  // 预分配1KB
    
    // 状态行
    result += "HTTP/1.1 ";
    result += std::to_string(response.getStatusCode());
    result += " ";
    switch (response.getStatusCode()) {
        case 200: result += "OK"; break;
        case 400: result += "Bad Request"; break;
        case 404: result += "Not Found"; break;
        case 500: result += "Internal Server Error"; break;
        default: result += "Unknown"; break;
    }
    result += "\r\n";
    
    // 头部
    auto headers = response.getAllHeaders();
    for (const auto& header : headers) {
        result += header.first;
        result += ": ";
        result += header.second;
        result += "\r\n";
    }
    
    // 如果没有Content-Type，默认设置
    if (!response.getContentType().empty()) {
        result += "Content-Type: ";
        result += response.getContentType();
        result += "\r\n";
    }
    
    // Content-Length
    std::string body = response.getBody();
    if (response.isStreaming()) {
        // 流式响应：合并所有chunks
        for (const auto& chunk : response.getChunks()) {
            body += chunk;
        }
    }
    result += "Content-Length: ";
    result += std::to_string(body.length());
    result += "\r\n";
    
    // Connection: keep-alive（默认支持）
    result += "Connection: keep-alive\r\n";
    
    // 空行
    result += "\r\n";
    
    // 响应体
    result += body;
    
    return result;
}

void HttpServer::init(const std::string& host, int port, HttpHandler* handler) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (instance_ != nullptr) {
        CLLM_ERROR("HttpServer already initialized");
        return;
    }
    
    instance_ = new HttpServer();
    instance_->host_ = host;
    instance_->port_ = port;
    instance_->handler_ = handler;
    instance_->serverFd_ = -1;
    instance_->epollFd_ = -1;
    instance_->running_.store(false);
    
    // 计算线程数
    unsigned int threads = static_cast<unsigned int>(Config::instance().serverNumThreads());
    const unsigned int minThreads = static_cast<unsigned int>(Config::instance().serverMinThreads());
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    if (threads == 0) {
        threads = hw;
    }
    threads = std::max(threads, minThreads);
    threads = std::max(threads, 2u);
    instance_->numThreads_ = threads;
    
    CLLM_INFO("HttpServer initialized: %s:%d, threads=%u", host.c_str(), port, threads);
}

void HttpServer::start() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (instance_ == nullptr) {
        CLLM_ERROR("HttpServer not initialized");
        return;
    }
    
    if (instance_->running_.load()) {
        CLLM_WARN("HttpServer already running");
        return;
    }
    
    instance_->running_.store(true);
    instance_->run();
}

void HttpServer::stop() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (instance_ == nullptr || !instance_->running_.load()) {
        return;
    }
    
    instance_->running_.store(false);
    
    // 关闭服务器socket
    if (instance_->serverFd_ >= 0) {
        close(instance_->serverFd_);
        instance_->serverFd_ = -1;
    }
    
    // 关闭epoll/kqueue
    for (int epfd : instance_->epollFds_) {
        if (epfd >= 0) {
            close(epfd);
        }
    }
    instance_->epollFds_.clear();
    
    // 关闭所有连接
    {
        std::lock_guard<std::mutex> lock(instance_->connectionsMutex_);
        for (auto& pair : instance_->connections_) {
            close(pair.first);
        }
        instance_->connections_.clear();
    }
    
    // 等待线程结束
    for (auto& thread : instance_->workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    CLLM_INFO("HttpServer stopped");
}

bool HttpServer::isRunning() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    return instance_ != nullptr && instance_->running_.load();
}

void HttpServer::run() {
    // 创建socket
    serverFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd_ < 0) {
        CLLM_ERROR("Failed to create socket: %s", strerror(errno));
        return;
    }
    
    // 设置socket选项
    int opt = 1;
    setsockopt(serverFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 设置为非阻塞
    int flags = fcntl(serverFd_, F_GETFL, 0);
    fcntl(serverFd_, F_SETFL, flags | O_NONBLOCK);
    
    // 绑定地址
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    
    if (host_ == "0.0.0.0" || host_.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    }
    
    if (bind(serverFd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        CLLM_ERROR("Failed to bind socket: %s", strerror(errno));
        close(serverFd_);
        return;
    }
    
    // 监听 - 增加backlog以支持更高并发
    // backlog = 512 可以支持更多pending连接
    if (listen(serverFd_, 512) < 0) {
        CLLM_ERROR("Failed to listen: %s", strerror(errno));
        close(serverFd_);
        return;
    }
    
    CLLM_INFO("HttpServer listening on %s:%d", host_.c_str(), port_);
    
    // 设置事件循环
    setupEventLoop();
    
    // 启动worker线程（每个线程运行独立的事件循环）
    for (unsigned int i = 0; i < numThreads_; ++i) {
        workerThreads_.emplace_back(&HttpServer::eventLoop, this, i);
    }
    
    CLLM_INFO("HttpServer started with %u event-driven worker threads (epoll/kqueue)", numThreads_);
}

// 这些函数已被eventLoop替代，保留为空实现以保持兼容
// 事件循环（epoll/kqueue）
void HttpServer::eventLoop(int workerId) {
    int epfd = epollFds_[workerId];
    CLLM_DEBUG("Event loop %d started (epfd=%d)", workerId, epfd);
    
    const int MAX_EVENTS = 64;
    
#ifdef __linux__
    struct epoll_event events[MAX_EVENTS];
#elif defined(__APPLE__) || defined(__FreeBSD__)
    struct kevent events[MAX_EVENTS];
#endif
    
    while (running_.load()) {
        int nfds;
        
#ifdef __linux__
        nfds = epoll_wait(epfd, events, MAX_EVENTS, 100);  // 100ms超时
        if (nfds < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (running_.load()) {
                CLLM_ERROR("epoll_wait failed: %s", strerror(errno));
            }
            break;
        }
        
        for (int i = 0; i < nfds; ++i) {
            int fd = events[i].data.fd;
            uint32_t ev = events[i].events;
            
            if (fd == serverFd_) {
                // 接受新连接
                while (true) {
                    struct sockaddr_in clientAddr;
                    socklen_t clientLen = sizeof(clientAddr);
                    int clientFd = accept(serverFd_, (struct sockaddr*)&clientAddr, &clientLen);
                    if (clientFd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;  // 没有更多连接
                        }
                        if (running_.load()) {
                            CLLM_ERROR("Accept failed: %s", strerror(errno));
                        }
                        break;
                    }
                    
                    // 设置非阻塞
                    int flags = fcntl(clientFd, F_GETFL, 0);
                    fcntl(clientFd, F_SETFL, flags | O_NONBLOCK);
                    
                    // 设置socket选项
                    int opt = 1;
                    setsockopt(clientFd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
                    
                    // 添加到连接状态
                    {
                        std::lock_guard<std::mutex> lock(connectionsMutex_);
                        connections_[clientFd] = ConnectionState();
                    }
                    
                    // 添加到事件循环
                    addEvent(clientFd, EPOLL_IN);
                }
            } else {
                // 处理客户端连接
                if (ev & (EPOLL_IN | EPOLLERR | EPOLLHUP)) {
                    handleReadEvent(fd);
                }
                if (ev & EPOLL_OUT) {
                    handleWriteEvent(fd);
                }
            }
        }
        
#elif defined(__APPLE__) || defined(__FreeBSD__)
        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = 100000000;  // 100ms
        
        nfds = kevent(epfd, nullptr, 0, events, MAX_EVENTS, &timeout);
        if (nfds < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (running_.load()) {
                CLLM_ERROR("kevent failed: %s", strerror(errno));
            }
            break;
        }
        
        for (int i = 0; i < nfds; ++i) {
            int fd = static_cast<int>(events[i].ident);
            int filter = events[i].filter;
            int flags = events[i].flags;
            
            if (fd == serverFd_) {
                // 接受新连接
                while (true) {
                    struct sockaddr_in clientAddr;
                    socklen_t clientLen = sizeof(clientAddr);
                    int clientFd = accept(serverFd_, (struct sockaddr*)&clientAddr, &clientLen);
                    if (clientFd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;
                        }
                        if (running_.load()) {
                            CLLM_ERROR("Accept failed: %s", strerror(errno));
                        }
                        break;
                    }
                    
                    // 检查连接数限制
                    {
                        std::lock_guard<std::mutex> lock(connectionsMutex_);
                        if (connections_.size() >= MAX_CONNECTIONS) {
                            // 连接数已达上限，拒绝新连接
                            CLLM_WARN("Max connections (%zu) reached, rejecting new connection", MAX_CONNECTIONS);
                            close(clientFd);
                            continue;
                        }
                    }
                    
                    // 设置非阻塞
                    int flags = fcntl(clientFd, F_GETFL, 0);
                    if (fcntl(clientFd, F_SETFL, flags | O_NONBLOCK) < 0) {
                        CLLM_ERROR("Failed to set non-blocking: %s", strerror(errno));
                        close(clientFd);
                        continue;
                    }
                    
                    // 设置socket选项
                    int opt = 1;
                    setsockopt(clientFd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
                    
                    // 设置接收/发送缓冲区大小（提升性能）
                    int recvBufSize = 64 * 1024;  // 64KB
                    int sendBufSize = 64 * 1024;  // 64KB
                    setsockopt(clientFd, SOL_SOCKET, SO_RCVBUF, &recvBufSize, sizeof(recvBufSize));
                    setsockopt(clientFd, SOL_SOCKET, SO_SNDBUF, &sendBufSize, sizeof(sendBufSize));
                    
                    // 添加到连接状态
                    {
                        std::lock_guard<std::mutex> lock(connectionsMutex_);
                        connections_[clientFd] = ConnectionState();
                    }
                    
                    // 添加到事件循环
                    addEvent(clientFd, EPOLL_IN);
                }
            } else {
                // 处理客户端连接
                if (filter == EVFILT_READ || (flags & EV_EOF)) {
                    handleReadEvent(fd);
                }
                if (filter == EVFILT_WRITE) {
                    handleWriteEvent(fd);
                }
            }
        }
#endif
    }
    
    CLLM_DEBUG("Event loop %d stopped", workerId);
}

void HttpServer::handleConnection(int clientFd) {
    // 此函数已被handleReadEvent/handleWriteEvent替代
    // 保留以保持兼容性
}

// 实现epoll/kqueue相关函数
void HttpServer::setupEventLoop() {
    epollFds_.clear();
    
    // 为每个worker线程创建独立的epoll/kqueue实例
    for (unsigned int i = 0; i < numThreads_; ++i) {
#ifdef __linux__
        int epfd = epoll_create1(EPOLL_CLOEXEC);
        if (epfd < 0) {
            CLLM_ERROR("Failed to create epoll instance: %s", strerror(errno));
            return;
        }
        
        // 将server socket添加到第一个epoll实例
        if (i == 0) {
            struct epoll_event ev;
            ev.events = EPOLL_IN | EPOLL_ET;  // 边缘触发
            ev.data.fd = serverFd_;
            if (epoll_ctl(epfd, EPOLL_CTL_ADD, serverFd_, &ev) < 0) {
                CLLM_ERROR("Failed to add server socket to epoll: %s", strerror(errno));
                close(epfd);
                return;
            }
            epollFd_ = epfd;  // 保存主epoll fd
        }
        epollFds_.push_back(epfd);
        
#elif defined(__APPLE__) || defined(__FreeBSD__)
        int kq = kqueue();
        if (kq < 0) {
            CLLM_ERROR("Failed to create kqueue: %s", strerror(errno));
            return;
        }
        
        // 将server socket添加到第一个kqueue实例
        if (i == 0) {
            struct kevent ev;
            EV_SET(&ev, serverFd_, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            if (kevent(kq, &ev, 1, nullptr, 0, nullptr) < 0) {
                CLLM_ERROR("Failed to add server socket to kqueue: %s", strerror(errno));
                close(kq);
                return;
            }
            epollFd_ = kq;  // 保存主kqueue fd
        }
        epollFds_.push_back(kq);
#endif
    }
    
    CLLM_INFO("Event loop setup complete: %zu instances", epollFds_.size());
}

void HttpServer::addEvent(int fd, uint32_t events) {
    // 使用轮询方式将新连接分配给worker线程
    static std::atomic<size_t> nextWorker{0};
    size_t workerId = nextWorker.fetch_add(1) % epollFds_.size();
    int epfd = epollFds_[workerId];
    
#ifdef __linux__
    struct epoll_event ev;
    ev.events = events | EPOLL_ET;  // 边缘触发
    ev.data.fd = fd;
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) < 0) {
        CLLM_ERROR("Failed to add fd %d to epoll: %s", fd, strerror(errno));
    }
#elif defined(__APPLE__) || defined(__FreeBSD__)
    struct kevent ev;
    if (events & EPOLL_IN) {
        EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
        if (kevent(epfd, &ev, 1, nullptr, 0, nullptr) < 0) {
            CLLM_ERROR("Failed to add fd %d to kqueue (read): %s", fd, strerror(errno));
        }
    }
    if (events & EPOLL_OUT) {
        EV_SET(&ev, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, nullptr);
        if (kevent(epfd, &ev, 1, nullptr, 0, nullptr) < 0) {
            CLLM_ERROR("Failed to add fd %d to kqueue (write): %s", fd, strerror(errno));
        }
    }
#endif
}

void HttpServer::modEvent(int fd, uint32_t events) {
    // 找到fd所在的epoll实例（简化实现：尝试所有实例）
    for (int epfd : epollFds_) {
#ifdef __linux__
        struct epoll_event ev;
        ev.events = events | EPOLL_ET;
        ev.data.fd = fd;
        if (epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev) == 0) {
            return;  // 成功修改
        }
#elif defined(__APPLE__) || defined(__FreeBSD__)
        struct kevent ev;
        if (events & EPOLL_IN) {
            EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            kevent(epfd, &ev, 1, nullptr, 0, nullptr);
        }
        if (events & EPOLL_OUT) {
            EV_SET(&ev, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            kevent(epfd, &ev, 1, nullptr, 0, nullptr);
        }
        return;
#endif
    }
}

void HttpServer::delEvent(int fd) {
    // 从所有epoll实例中删除
    for (int epfd : epollFds_) {
#ifdef __linux__
        epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
#elif defined(__APPLE__) || defined(__FreeBSD__)
        struct kevent ev;
        EV_SET(&ev, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
        kevent(epfd, &ev, 1, nullptr, 0, nullptr);
        EV_SET(&ev, fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
        kevent(epfd, &ev, 1, nullptr, 0, nullptr);
#endif
    }
}

// 非阻塞读取事件处理
void HttpServer::handleReadEvent(int clientFd) {
    // 🔥 优化：先快速查找，减少锁持有时间
    ConnectionState* conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        auto it = connections_.find(clientFd);
        if (it == connections_.end()) {
            return;
        }
        conn = &it->second;
    }
    
    // 在锁外处理大部分逻辑（减少锁竞争）
    ConnectionState& connection = *conn;
    
    // 读取数据
    char buf[4096];
    ssize_t n = recv(clientFd, buf, sizeof(buf) - 1, 0);
    
    if (n <= 0) {
        if (n == 0) {
            // 连接正常关闭
            CLLM_DEBUG("Connection %d closed by peer", clientFd);
            {
                std::lock_guard<std::mutex> lock(connectionsMutex_);
                connections_.erase(clientFd);
            }
            delEvent(clientFd);
            close(clientFd);
            return;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            // 连接错误
            CLLM_WARN("Read error on connection %d: %s", clientFd, strerror(errno));
            {
                std::lock_guard<std::mutex> lock(connectionsMutex_);
                connections_.erase(clientFd);
            }
            delEvent(clientFd);
            close(clientFd);
            return;
        }
        return;  // EAGAIN，稍后重试
    }
    
    connection.readBuffer.append(buf, n);  // 🔥 优化：使用append避免临时string和\0
    
    // 解析HTTP请求（状态机）
    if (connection.state == ConnectionState::READING_HEADER) {
        // 查找请求行和头部结束
        size_t headerEnd = connection.readBuffer.find("\r\n\r\n");
        if (headerEnd == std::string::npos) {
            return;  // 头部未完整
        }
        
        std::string headerPart = connection.readBuffer.substr(0, headerEnd);
        connection.readBuffer.erase(0, headerEnd + 4);
        
        // 🔥 优化：直接解析，避免istringstream和getline开销
        size_t pos = 0;
        bool firstLine = true;
        
        while (pos < headerPart.length()) {
            size_t lineEnd = headerPart.find("\r\n", pos);
            if (lineEnd == std::string::npos) {
                lineEnd = headerPart.length();
            }
            
            if (lineEnd == pos) {
                break;  // 空行，头部结束
            }
            
            std::string line = headerPart.substr(pos, lineEnd - pos);
            pos = lineEnd + 2;
            
            if (firstLine) {
                // 解析请求行
                size_t firstSpace = line.find(' ');
                size_t secondSpace = line.find(' ', firstSpace + 1);
                if (firstSpace != std::string::npos && secondSpace != std::string::npos) {
                    std::string method = line.substr(0, firstSpace);
                    std::string path = line.substr(firstSpace + 1, secondSpace - firstSpace - 1);
                    connection.request.setMethod(std::move(method));
                    connection.request.setPath(std::move(path));
                }
                firstLine = false;
            } else {
                // 解析头部
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    std::string name = line.substr(0, colonPos);
                    std::string value = line.substr(colonPos + 1);
                    // 去除前导空格
                    size_t firstNonSpace = value.find_first_not_of(" \t");
                    if (firstNonSpace != std::string::npos) {
                        value = value.substr(firstNonSpace);
                    }
                    // 🔥 优化：原地转小写
                    for (char& c : name) {
                        if (c >= 'A' && c <= 'Z') {
                            c = c - 'A' + 'a';
                        }
                    }
                    connection.request.setHeader(std::move(name), std::move(value));
                }
            }
        }
        
        // 检查Content-Length
        std::string contentLengthStr = connection.request.getHeader("content-length");
        if (!contentLengthStr.empty()) {
            try {
                connection.contentLength = std::stoul(contentLengthStr);
                if (connection.contentLength > 0 && connection.contentLength < 10 * 1024 * 1024) {
                    connection.state = ConnectionState::READING_BODY;
                } else {
                    connection.state = ConnectionState::WRITING;  // 无body或body过大
                }
            } catch (...) {
                connection.state = ConnectionState::WRITING;
            }
        } else {
            connection.state = ConnectionState::WRITING;  // 无body
        }
        
        // 检查Keep-Alive
        std::string connHeader = connection.request.getHeader("connection");
        // 🔥 优化：原地转小写
        for (char& c : connHeader) {
            if (c >= 'A' && c <= 'Z') {
                c = c - 'A' + 'a';
            }
        }
        connection.keepAlive = (connHeader != "close");
    }
    
    if (connection.state == ConnectionState::READING_BODY) {
        // 读取请求体
        if (connection.readBuffer.length() >= connection.contentLength) {
            std::string body = connection.readBuffer.substr(0, connection.contentLength);
            connection.readBuffer.erase(0, connection.contentLength);
            connection.request.setBody(std::move(body));  // 🔥 优化：使用move
            connection.state = ConnectionState::WRITING;
        } else {
            return;  // body未完整
        }
    }
    
    if (connection.state == ConnectionState::WRITING) {
        // 处理请求
        HttpResponse response;
        try {
            if (handler_) {
                response = handler_->handleRequest(connection.request);
            } else {
                CLLM_ERROR("Handler not set for request");
                response = HttpResponse::internalError("Handler not set");
            }
        } catch (const std::exception& e) {
            CLLM_ERROR("Exception in request handler for connection %d: %s", clientFd, e.what());
            response = HttpResponse::internalError("Internal server error: " + std::string(e.what()));
        } catch (...) {
            CLLM_ERROR("Unknown exception in request handler for connection %d", clientFd);
            response = HttpResponse::internalError("Internal server error");
        }
        
        // 构建响应
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            auto it = connections_.find(clientFd);
            if (it != connections_.end()) {
                it->second.writeBuffer = buildHttpResponse(response);
            }
        }
        
        // 切换到写事件
        modEvent(clientFd, EPOLL_OUT);
    }
}

// 非阻塞写入事件处理
void HttpServer::handleWriteEvent(int clientFd) {
    // 🔥 优化：先快速查找，减少锁持有时间
    ConnectionState* conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        auto it = connections_.find(clientFd);
        if (it == connections_.end()) {
            return;
        }
        conn = &it->second;
    }
    
    ConnectionState& connection = *conn;
    
    if (connection.writeBuffer.empty()) {
        // 没有数据要写，切换回读事件
        modEvent(clientFd, EPOLL_IN);
        return;
    }
    
    // 发送数据
    ssize_t sent = send(clientFd, connection.writeBuffer.c_str(), connection.writeBuffer.length(), 0);
    
    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;  // 稍后重试
        }
        // 发送错误，关闭连接
        CLLM_WARN("Write error on connection %d: %s", clientFd, strerror(errno));
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            connections_.erase(clientFd);
        }
        delEvent(clientFd);
        close(clientFd);
        return;
    }
    
    // 🔥 优化：如果全部发送完成，直接清空；否则只删除已发送部分
    if (static_cast<size_t>(sent) >= connection.writeBuffer.length()) {
        connection.writeBuffer.clear();
    } else {
        connection.writeBuffer.erase(0, sent);
    }
    
    if (connection.writeBuffer.empty()) {
        // 响应发送完成
        bool keepAlive = connection.keepAlive;
        
        if (keepAlive) {
            // Keep-Alive：重置状态，继续读取下一个请求
            {
                std::lock_guard<std::mutex> lock(connectionsMutex_);
                auto it = connections_.find(clientFd);
                if (it != connections_.end()) {
                    it->second = ConnectionState();  // 重置状态
                }
            }
            modEvent(clientFd, EPOLL_IN);
        } else {
            // 关闭连接
            {
                std::lock_guard<std::mutex> lock(connectionsMutex_);
                connections_.erase(clientFd);
            }
            delEvent(clientFd);
            close(clientFd);
        }
    }
}

} // namespace cllm
