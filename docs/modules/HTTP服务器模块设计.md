# HTTP Server模块详细设计（基于Drogon框架）

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](../../C++编程规范.md)：定义编码风格、命名规范等
- [生成代码规范.md](../生成代码规范.md)：定义代码生成流程、设计文档一致性要求、优化同步机制等

## 0. 要生成的文件

### 0.1 头文件（include/cllm/http/）

根据[C++编程规范.md](../../C++编程规范.md)的命名规范，本模块需要生成以下头文件：

| 文件名 | 对应类/结构体 | 说明 |
|--------|--------------|------|
| `drogon_server.h` | `DrogonServer` | Drogon HTTP服务器控制器 |
| `request.h` | `HttpRequest` | HTTP请求类 |
| `response.h` | `HttpResponse` | HTTP响应类 |
| `handler.h` | `HttpHandler` | HTTP请求处理器 |
| `generate_endpoint.h` | `GenerateEndpoint` | 文本生成API端点 |
| `encode_endpoint.h` | `EncodeEndpoint` | 文本编码API端点 |
| `health_endpoint.h` | `HealthEndpoint` | 健康检查API端点 |
| `request_validator.h` | `RequestValidator` | 请求验证器 |

### 0.2 源文件（src/http/）

| 文件名 | 对应头文件 | 说明 |
|--------|-----------|------|
| `drogon_server.cpp` | `drogon_server.h` | DrogonServer类的实现 |
| `request.cpp` | `request.h` | HttpRequest类的实现 |
| `response.cpp` | `response.h` | HttpResponse类的实现 |
| `handler.cpp` | `handler.h` | HttpHandler类的实现 |
| `generate_endpoint.cpp` | `generate_endpoint.h` | GenerateEndpoint类的实现 |
| `encode_endpoint.cpp` | `encode_endpoint.h` | EncodeEndpoint类的实现 |
| `health_endpoint.cpp` | `health_endpoint.h` | HealthEndpoint类的实现 |
| `request_validator.cpp` | `request_validator.h` | RequestValidator类的实现 |

### 0.3 测试文件（tests/）

| 文件名 | 测试目标 | 说明 |
|--------|---------|------|
| `test_http_server.cpp` | DrogonServer, HttpHandler, API端点 | HTTP服务器模块的单元测试 |

### 0.4 文件命名规范说明

- **头文件名**：在http目录下，使用简洁命名（如`server.h`而非`http_server.h`）
- **源文件名**：与对应头文件名保持一致
- **目录结构**：头文件位于 `include/cllm/http/`，源文件位于 `src/http/`
- **一致性原则**：所有文件命名遵循[C++编程规范.md](../../C++编程规范.md)第1.1节

## 1. 模块概述

### 1.1 模块职责
HTTP Server模块负责处理HTTP请求，提供RESTful API接口，接收客户端请求，调用后端服务（Scheduler、Tokenizer等），并返回响应结果。

### 1.2 核心功能
- HTTP请求处理和路由
- API端点实现
- 请求参数验证
- 流式和非流式响应支持
- 错误处理和异常管理
- 请求日志记录
- 健康检查
- 服务器生命周期管理

### 1.3 设计原则
- 简单性：避免复杂的C++语法，使用RAII包装器
- 高性能：使用异步I/O和事件驱动
- 可扩展性：支持动态添加API端点
- 可维护性：清晰的代码结构和命名规范
- 安全性：输入验证和错误处理

### 1.4 技术选型
- **HTTP框架**: Drogon（现代C++异步Web框架）
- **JSON处理**: nlohmann/json（通过Drogon内置支持）
- **异步I/O**: Drogon内置异步事件循环（基于epoll/kqueue）
- **协程支持**: C++20协程（Drogon集成）
- **线程模型**: Drogon内置线程池和IO线程池

## 2. 类设计

### 2.1 HttpRequest类

```cpp
class HttpRequest {
public:
    HttpRequest();
    ~HttpRequest();
    
    std::string getMethod() const;
    std::string getPath() const;
    std::string getHeader(const std::string& name) const;
    std::string getBody() const;
    std::string getQuery(const std::string& key) const;
    
    void setMethod(const std::string& method);
    void setPath(const std::string& path);
    void setHeader(const std::string& name, const std::string& value);
    void setBody(const std::string& body);
    void setQuery(const std::string& key, const std::string& value);
    
    bool hasHeader(const std::string& name) const;
    bool hasQuery(const std::string& key) const;
    
    std::map<std::string, std::string> getAllHeaders() const;
    std::map<std::string, std::string> getAllQueries() const;
    
private:
    std::string method_;
    std::string path_;
    std::map<std::string, std::string> headers_;
    std::string body_;
    std::map<std::string, std::string> queries_;
};
```

### 2.2 HttpResponse类

```cpp
class HttpResponse {
public:
    HttpResponse();
    ~HttpResponse();
    
    void setStatusCode(int code);
    void setHeader(const std::string& name, const std::string& value);
    void setBody(const std::string& body);
    void setContentType(const std::string& contentType);
    
    int getStatusCode() const;
    std::string getHeader(const std::string& name) const;
    std::string getBody() const;
    std::string getContentType() const;
    
    std::map<std::string, std::string> getAllHeaders() const;
    
    void setJson(const nlohmann::json& json);
    void setError(int code, const std::string& message);
    
    static HttpResponse ok(const std::string& body = "");
    static HttpResponse notFound();
    static HttpResponse badRequest(const std::string& message = "");
    static HttpResponse internalError(const std::string& message = "");
    
private:
    int statusCode_;
    std::map<std::string, std::string> headers_;
    std::string body_;
};
```

### 2.3 HttpHandler类

```cpp
class HttpHandler {
public:
    typedef std::function<HttpResponse(const HttpRequest&)> HandlerFunc;
    
    HttpHandler();
    ~HttpHandler();
    
    void get(const std::string& path, HandlerFunc handler);
    void post(const std::string& path, HandlerFunc handler);
    void put(const std::string& path, HandlerFunc handler);
    void del(const std::string& path, HandlerFunc handler);
    
    HttpResponse handleRequest(const HttpRequest& request);
    
    bool hasHandler(const std::string& method, const std::string& path) const;
    
private:
    std::map<std::string, HandlerFunc> getHandlers_;
    std::map<std::string, HandlerFunc> postHandlers_;
    std::map<std::string, HandlerFunc> putHandlers_;
    std::map<std::string, HandlerFunc> deleteHandlers_;
    
    std::string normalizePath(const std::string& path) const;
    bool matchPath(const std::string& pattern, const std::string& path) const;
};
```

### 2.4 DrogonServer类

```cpp
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
    static HttpHandler* handler_;
    static std::string host_;
    static int port_;
};
```
```

### 2.5 ApiEndpoint类

```cpp
class ApiEndpoint {
public:
    ApiEndpoint(
        const std::string& name,
        const std::string& path,
        const std::string& method
    );
    ~ApiEndpoint();
    
    std::string getName() const;
    std::string getPath() const;
    std::string getMethod() const;
    
    void setHandler(HttpHandler::HandlerFunc handler);
    HttpHandler::HandlerFunc getHandler() const;
    
    void validateRequest(const HttpRequest& request);
    
    virtual HttpResponse handle(const HttpRequest& request) = 0;
    
protected:
    std::string name_;
    std::string path_;
    std::string method_;
    HttpHandler::HandlerFunc handler_;
};
```

### 2.6 HealthEndpoint类

```cpp
class HealthEndpoint : public ApiEndpoint {
public:
    HealthEndpoint();
    ~HealthEndpoint();
    
    HttpResponse handle(const HttpRequest& request) override;
};
```

### 2.7 GenerateEndpoint类

```cpp
class GenerateEndpoint : public ApiEndpoint {
public:
    GenerateEndpoint(Scheduler* scheduler, Tokenizer* tokenizer);
    ~GenerateEndpoint();
    
    HttpResponse handle(const HttpRequest& request) override;
    
    void setScheduler(Scheduler* scheduler);
    void setTokenizer(Tokenizer* tokenizer);
    
private:
    GenerateRequest parseRequest(const HttpRequest& request);
    std::string generateRequestId();
    HttpResponse handleNonStreaming(const GenerateRequest& req);
    HttpResponse handleStreaming(const GenerateRequest& req);
    
    Scheduler* scheduler_;
    Tokenizer* tokenizer_;
};
```

### 2.8 EncodeEndpoint类

```cpp
class EncodeEndpoint : public ApiEndpoint {
public:
    EncodeEndpoint(Tokenizer* tokenizer);
    ~EncodeEndpoint();
    
    HttpResponse handle(const HttpRequest& request) override;
    
    void setTokenizer(Tokenizer* tokenizer);
    
private:
    EncodeRequest parseRequest(const HttpRequest& request);
    
    Tokenizer* tokenizer_;
};
```

### 2.9 RequestValidator类

```cpp
class RequestValidator {
public:
    RequestValidator();
    ~RequestValidator();
    
    bool validateRequired(const HttpRequest& request, const std::string& field);
    bool validateType(const std::string& value, const std::string& expectedType);
    bool validateRange(int value, int min, int max);
    bool validateRange(float value, float min, float max);
    
    std::string getLastError() const;
    
private:
    std::string lastError_;
};
```

### 2.10 ResponseBuilder类

```cpp
class ResponseBuilder {
public:
    ResponseBuilder();
    ~ResponseBuilder();
    
    ResponseBuilder& setStatus(int code);
    ResponseBuilder& setHeader(const std::string& name, const std::string& value);
    ResponseBuilder& setBody(const std::string& body);
    ResponseBuilder& setJson(const nlohmann::json& json);
    
    HttpResponse build();
    
    static ResponseBuilder ok();
    static ResponseBuilder error(int code, const std::string& message);
    
private:
    int statusCode_;
    std::map<std::string, std::string> headers_;
    std::string body_;
};
```

## 3. 接口设计

### 3.1 HTTP接口

#### 3.1.1 健康检查接口
```
GET /health
```

**请求参数：** 无

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 3.1.2 生成接口
```
POST /generate
```

**请求参数：**
```json
{
  "prompt": "你好，世界",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

**响应示例（非流式）：**
```json
{
  "id": "req_1234567890_1234",
  "text": "你好！这是一个生成响应。",
  "tokens": [15496, 995, 428, 30],
  "response_time": 1.234,
  "tokens_per_second": 24.5
}
```

**响应示例（流式）：**
```
data: {"token": "你好"}
data: {"token": "！"}
data: {"token": "这是"}
data: {"token": "一个"}
data: {"token": "生成"}
data: {"token": "响应"}
data: {"token": "。", "done": true}
data: [DONE]
```

#### 3.1.3 编码接口
```
POST /encode
```

**请求参数：**
```json
{
  "text": "你好，世界"
}
```

**响应示例：**
```json
{
  "tokens": [15496, 995, 428, 30],
  "length": 4
}
```

### 3.2 内部接口

#### 3.2.1 DrogonServer接口
```cpp
static void init(const std::string& host, int port, HttpHandler* handler);
static void start();
static void stop();

// HTTP控制器方法
void health(const drogon::HttpRequestPtr& req,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback);
void generate(const drogon::HttpRequestPtr& req,
             std::function<void(const drogon::HttpResponsePtr&)>&& callback);
void generateStream(const drogon::HttpRequestPtr& req,
                   std::function<void(const drogon::HttpResponsePtr&)>&& callback);
void encode(const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback);
```

#### 3.2.2 HttpHandler接口
```cpp
// 保持原有接口，但实现方式适配Drogon框架
HttpResponse handleRequest(const HttpRequest& request);
```

#### 3.2.3 ApiEndpoint接口
```cpp
virtual HttpResponse handle(const HttpRequest& request) = 0;
void validateRequest(const HttpRequest& request);
```

## 4. 算法设计

### 4.1 HTTP请求处理流程（Drogon框架）

```
算法：Drogon请求处理
输入：drogon::HttpRequestPtr
输出：drogon::HttpResponsePtr

1. Drogon框架接收HTTP请求
2. 路由到对应的控制器方法（health/generate/encode）
3. 将Drogon请求转换为内部HttpRequest格式
4. 调用对应的HttpHandler处理请求
5. 将内部HttpResponse转换为Drogon响应格式
6. 通过回调函数返回响应
```

### 4.2 流式响应处理

```
算法：流式响应处理
输入：生成请求参数
输出：Server-Sent Events (SSE) 流

1. 设置响应头为text/event-stream
2. 创建Drogon流式响应
3. 启动异步生成任务
4. 每生成一个token，格式化为SSE事件
5. 通过流回调发送事件数据
6. 生成完成后发送结束标记
```

### 4.3 路由匹配算法

```
算法：matchPath
输入：pattern (路由模式), path (请求路径)
输出：是否匹配

1. 标准化路径（去除首尾斜杠）
2. 如果pattern和path完全相等，返回true
3. 如果pattern包含通配符（*），进行通配符匹配
4. 如果pattern包含参数占位符（如:id），进行参数匹配
5. 否则返回false
```

### 4.4 流式响应生成算法

```
算法：handleStreaming
输入：GenerateRequest对象
输出：流式HttpResponse对象

1. 创建生成请求ID
2. 向Scheduler提交生成请求
3. 设置响应头为text/event-stream
4. 创建流式响应生成器
   a. 等待生成结果
   b. 每次生成一个token
   c. 格式化为SSE格式
   d. 发送给客户端
5. 当生成完成时，发送结束标记
6. 返回流式HttpResponse对象
```

### 4.5 请求验证算法

```
算法：validateRequest
输入：HttpRequest对象
输出：验证结果

1. 检查必需参数是否存在
2. 检查参数类型是否正确
3. 检查参数值是否在有效范围内
4. 检查参数格式是否正确
5. 如果验证失败，设置错误信息
6. 返回验证结果
```

## 5. 并发设计

### 5.1 Drogon并发模型
HTTP Server采用Drogon框架的异步事件驱动模型：
- **IO线程池**：处理网络I/O操作（epoll/kqueue）
- **工作线程池**：处理CPU密集型业务逻辑
- **协程支持**：C++20协程实现异步编程

### 5.2 线程模型
```
Drogon应用框架
  ├── 主线程（事件循环管理）
  ├── IO线程池（网络I/O处理）
  │    ├── IO线程1
  │    ├── IO线程2
  │    └── IO线程N
  └── 工作线程池（业务逻辑处理）
       ├── Worker 1
       ├── Worker 2
       └── Worker M
```

### 5.3 锁策略
- **DrogonServer**：无锁，Drogon框架保证线程安全
- **HttpHandler**：无锁，通过Drogon的线程安全机制
- **请求处理**：每个请求在独立上下文中处理

### 5.4 并发安全
- Drogon框架提供线程安全的请求处理环境
- 使用异步编程模式避免阻塞IO线程
- 协程支持简化异步代码编写

### 5.5 性能优化
- Drogon内置高性能epoll/kqueue事件循环
- 零拷贝网络栈优化
- 连接复用和连接池管理
- 异步I/O和非阻塞操作

## 6. 内存管理

### 6.1 内存分配策略
- Drogon框架管理网络缓冲区和连接内存
- 使用RAII包装器管理业务逻辑对象
- 智能指针管理资源生命周期

### 6.2 Drogon内存管理
- **请求/响应对象**：由Drogon框架管理生命周期
- **业务数据**：使用共享指针确保线程安全
- **缓冲区管理**：Drogon内置高效缓冲区池

### 6.3 内存优化
- Drogon使用自研的Buffer类优化内存分配
- 零拷贝技术减少数据复制
- 使用移动语义传递大型对象
- 对象池复用频繁创建的对象

### 6.4 内存监控
- Drogon内置内存使用统计
- 监控连接内存使用
- 统计请求处理内存开销
- 集成系统内存监控

## 7. 错误处理

### 7.1 错误类型
```cpp
enum HttpError {
    HTTP_OK = 200,
    HTTP_BAD_REQUEST = 400,
    HTTP_NOT_FOUND = 404,
    HTTP_INTERNAL_ERROR = 500,
    HTTP_SERVICE_UNAVAILABLE = 503
};

class HttpException {
public:
    HttpException(int code, const std::string& message);
    
    int getCode() const;
    std::string getMessage() const;
    
private:
    int code_;
    std::string message_;
};
```

### 7.2 错误处理策略
- 参数验证失败：返回400 Bad Request
- 路由不存在：返回404 Not Found
- 服务不可用：返回503 Service Unavailable
- 内部错误：返回500 Internal Server Error
- 所有错误都记录到日志

### 7.3 错误响应格式
```json
{
  "error": {
    "code": 400,
    "message": "Invalid request parameter"
  }
}
```

### 7.4 异常传播
- 使用try-catch捕获异常
- 将异常转换为HTTP错误响应
- 记录异常堆栈到日志
- 不泄露敏感信息

## 8. 性能优化

### 8.1 I/O优化
- 使用epoll/kqueue实现高并发连接处理
- 使用非阻塞I/O提高吞吐量
- 使用TCP_NODELAY减少延迟
- 使用keep-alive复用连接

### 8.2 内存优化
- 使用mimalloc提高内存分配性能
- 使用对象池减少内存分配开销
- 使用零拷贝技术减少数据复制
- 使用内存对齐提高访问速度

### 8.3 并发优化
- 使用线程池处理并发请求
- 使用无锁数据结构减少锁竞争
- 使用事件驱动提高并发性能
- 使用线程亲和性提高缓存命中率

### 8.4 算法优化
- 使用高效的路由匹配算法
- 使用快速的JSON解析库
- 使用字符串哈希加速查找
- 使用缓存减少重复计算

### 8.5 性能监控
- 记录请求处理时间
- 统计请求吞吐量
- 监控并发连接数
- 记录错误率

## 9. 测试设计

### 9.1 单元测试
- 测试HttpRequest类的功能
- 测试HttpResponse类的功能
- 测试HttpHandler的路由功能
- 测试RequestValidator的验证功能
- 测试ResponseBuilder的构建功能

### 9.2 集成测试
- 测试HTTP服务器的启动和停止
- 测试API端点的功能
- 测试流式和非流式响应
- 测试错误处理机制
- 测试并发请求处理

### 9.3 性能测试
- 测试单请求延迟
- 测试并发请求吞吐量
- 测试内存使用情况
- 测试CPU使用情况
- 测试长时间运行稳定性

### 9.4 压力测试
- 测试最大并发连接数
- 测试最大请求速率
- 测试内存泄漏
- 测试资源耗尽情况
- 测试恢复能力

### 9.5 测试用例示例

#### 9.5.1 健康检查测试
```cpp
TEST(HttpServerTest, HealthCheck) {
    HttpServer server("127.0.0.1", 8080);
    server.start();
    
    HttpRequest request;
    request.setMethod("GET");
    request.setPath("/health");
    
    HttpResponse response = server.handleRequest(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    server.stop();
}
```

#### 9.5.2 生成接口测试
```cpp
TEST(HttpServerTest, GenerateEndpoint) {
    HttpServer server("127.0.0.1", 8080);
    server.start();
    
    nlohmann::json requestBody;
    requestBody["prompt"] = "Hello";
    requestBody["max_tokens"] = 10;
    requestBody["temperature"] = 0.7;
    requestBody["stream"] = false;
    
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/generate");
    request.setBody(requestBody.dump());
    
    HttpResponse response = server.handleRequest(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    server.stop();
}
```

#### 9.5.3 并发请求测试
```cpp
TEST(HttpServerTest, ConcurrentRequests) {
    HttpServer server("127.0.0.1", 8080);
    server.start();
    
    const int numRequests = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numRequests; i++) {
        threads.emplace_back([&server]() {
            HttpRequest request;
            request.setMethod("GET");
            request.setPath("/health");
            
            HttpResponse response = server.handleRequest(request);
            EXPECT_EQ(response.getStatusCode(), 200);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    server.stop();
}
```

## 10. 部署和配置

### 10.1 配置参数
Drogon框架提供丰富的配置选项，通过JSON配置文件管理：

```json
{
  "listeners": [
    {
      "address": "0.0.0.0",
      "port": 8080,
      "https": false
    }
  ],
  "thread_num": 16,
  "client_max_body_size": 10485760,
  "upload_path": "./uploads",
  "document_root": "./www",
  "log": {
    "log_path": "./logs",
    "log_level": "INFO"
  }
}
```

### 10.2 环境变量
- `DROGON_LISTEN_ADDR`: 监听地址
- `DROGON_LISTEN_PORT`: 监听端口
- `DROGON_THREAD_NUM`: 工作线程数
- `DROGON_LOG_LEVEL`: 日志级别
- `DROGON_DOCUMENT_ROOT`: 静态文件根目录

### 10.3 启动流程
1. 读取Drogon配置文件
2. 初始化Drogon应用框架
3. 注册HTTP控制器（DrogonServer）
4. 设置业务逻辑处理器（HttpHandler）
5. 启动Drogon事件循环
6. 处理HTTP请求
7. 优雅关闭应用

## 11. 日志和监控

### 11.1 日志记录
- 记录每个HTTP请求的基本信息
- 记录请求处理时间
- 记录错误和异常
- 记录性能指标

### 11.2 监控指标
- 请求总数
- 请求成功率
- 平均响应时间
- 并发连接数
- 错误率
- 内存使用
- CPU使用

### 11.3 健康检查
- 定期检查服务器状态
- 检查依赖服务状态
- 检查资源使用情况
- 提供健康检查接口

## 12. 安全考虑

### 12.1 Drogon安全特性
- **HTTPS支持**: 内置TLS/SSL加密传输
- **CORS支持**: 跨域资源共享配置
- **CSRF防护**: 内置CSRF令牌验证
- **输入验证**: 参数验证和过滤

### 12.2 访问控制
- 基于Drogon中间件的认证机制
- JWT令牌验证支持
- 速率限制和访问频率控制
- IP白名单/黑名单过滤

### 12.3 数据保护
- TLS加密通信
- 敏感数据脱敏处理
- 安全头部（Security Headers）自动设置
- 定期安全更新和依赖检查

## 13. 总结

HTTP Server模块基于Drogon框架重构，大幅简化了HTTP服务器实现。Drogon提供了高性能的异步事件驱动架构，内置连接池、线程池、协程支持等现代Web框架特性。

主要优势：
1. **性能提升**: Drogon框架优化了网络I/O和并发处理，支持数万并发连接
2. **开发效率**: 简洁的HttpController接口，减少样板代码
3. **功能完整**: 内置路由、中间件、模板引擎、WebSocket等完整功能
4. **生产就绪**: 成熟的Web框架，支持HTTP/1.1、HTTP/2、HTTPS等协议
5. **易于维护**: 基于标准C++20，良好的文档和社区支持

模块提供了健康检查、文本生成、文本编码等RESTful API端点，支持流式和非流式响应。通过Drogon框架的强大功能，确保了服务器的高性能、可扩展性和可维护性。

模块遵循C++编程规范，与Drogon框架的良好设计理念相结合，确保代码的质量和可持续发展。

