# 高性能HTTP服务器设计文档

## 设计目标

1. **高性能**: 最小化开销，目标达到80+ t/s
2. **轻量级**: 无外部依赖（除标准库）
3. **兼容性**: 保持与DrogonServer相同的接口
4. **功能完整**: 支持所有现有端点（/health, /generate, /generate_stream, /encode, /benchmark）

## 架构设计

### 核心组件

1. **HttpServer**: 主服务器类，管理连接和线程池
2. **HttpConnection**: 单个连接处理类，处理请求/响应
3. **HttpParser**: HTTP请求解析器（轻量级）
4. **HttpBuilder**: HTTP响应构建器（零拷贝优化）

### 技术选型

- **网络IO**: 原生socket + epoll/kqueue（Linux/macOS）
- **并发模型**: 线程池 + 事件驱动
- **连接管理**: Keep-Alive支持，连接池复用
- **内存管理**: 对象池，减少分配

## 实现方案

### 1. 线程模型

```
主线程（Acceptor）
  ↓
线程池（Worker Threads，默认16个）
  ↓
每个线程处理多个连接（事件驱动）
```

### 2. HTTP解析

- 使用状态机解析HTTP请求
- 支持HTTP/1.1 Keep-Alive
- 最小化字符串拷贝

### 3. 路由分发

- 复用现有HttpHandler
- 路径匹配优化（哈希表）

### 4. 响应构建

- 预分配缓冲区
- 零拷贝响应构建
- 支持流式响应（SSE）

## 性能优化策略

1. **零拷贝**: 使用string_view，避免不必要的拷贝
2. **对象池**: 复用HttpRequest/HttpResponse对象
3. **Keep-Alive**: 减少连接建立/关闭开销
4. **批量处理**: 批量处理多个请求
5. **无锁设计**: 使用原子操作和lock-free数据结构

## 接口兼容性

保持与DrogonServer相同的接口：

```cpp
class HttpServer {
public:
    static void init(const std::string& host, int port, HttpHandler* handler);
    static void start();
    static void stop();
};
```

## 实现计划

1. ✅ 设计文档
2. ⏳ 实现HttpParser（HTTP请求解析）
3. ⏳ 实现HttpConnection（连接处理）
4. ⏳ 实现HttpServer（主服务器）
5. ⏳ 集成到cLLM
6. ⏳ 功能测试
7. ⏳ 性能测试
