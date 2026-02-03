# cLLM C++ 重构详细设计文档

## 1. 项目概述

### 1.1 项目目标
将 xLLM Python 版本重构为 C++ 实现，以获得更高的性能和更好的资源利用率。主要目标包括：
- 将推理性能提升至 20+ tokens/s
- 降低内存占用
- 提高并发处理能力
- 保持与 Python 版本相同的功能特性

### 1.2 技术栈
- **编程语言**: C++17
- **HTTP 服务器**: 自研高性能 HTTP Server（基于 epoll/kqueue，支持流式输出）
- **深度学习推理**: LibTorch（PyTorch C++ API）
- **数值计算**: Eigen3（线性代数库）
- **JSON 处理**: nlohmann/json
- **异步框架**: Asio（standalone Asio）
- **分词器**: sentencepiece（Google分词库）
- **量化支持**: LibTorch内置量化支持
- **日志**: spdlog
- **测试**: Google Test + Google Mock

## 2. 系统架构

### 2.1 整体架构图
```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server Layer                     │
│ (RESTful API Endpoints, Request Handling, Validation)    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Request Scheduler                       │
│  (Request Management, Dynamic Batching, Execution)       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Model Executor                           │
│  (Model Loading, Inference, Quantization, Optimization)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Core Components Layer                       │
│ Tokenizer | Sampler | KV Cache | Memory Management       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

#### 2.2.1 HTTP Server 模块
- **文件**: `include/cllm/http/http_server.h`, `src/http/http_server.cpp`
- **职责**: 
  - 自研高性能 HTTP 服务器，基于 epoll (Linux) / kqueue (macOS)
  - 支持真流式输出（chunked transfer encoding）
  - 实现 RESTful API 端点（健康检查、生成、流式生成、编码）
  - 请求验证和错误处理
  - 服务器生命周期管理

#### 2.2.2 Scheduler 模块
- **文件**: `include/cllm/scheduler/scheduler.h`, `src/scheduler/scheduler.cpp`
- **职责**:
  - 请求队列管理
  - 动态批处理
  - 请求生命周期跟踪
  - 批处理执行协调

#### 2.2.3 Model Executor 模块
- **文件**: `include/cllm/model/executor.h`, `src/model/executor.cpp`
- **职责**:
  - 模型加载和管理
  - 推理执行
  - 量化支持
  - 推理优化

#### 2.2.4 Tokenizer 模块
- **文件**: `include/cllm/tokenizer/tokenizer.h`, `src/tokenizer/tokenizer.cpp`
- **职责**:
  - 文本编码/解码
  - Token ID 转换
  - 特殊 Token 处理

#### 2.2.5 Sampler 模块
- **文件**: `include/cllm/sampler.h`, `src/sampler/sampler.cpp`
- **职责**:
  - Token 采样策略
  - 批量采样优化
  - 温度、Top-K、Top-P 支持

#### 2.2.6 KV Cache 模块
- **文件**: `include/cllm/kv_cache/cache.h`, `src/kv_cache/cache.cpp`
- **职责**:
  - KV 缓存管理
  - LRU 淘汰策略
  - 内存使用监控
  - 请求上下文缓存

#### 2.2.7 Memory Management 模块
- **文件**: `include/cllm/memory/*.h`, `src/memory/*.cpp`
- **职责**:
  - 内存监控
  - 内存池管理
  - KV 缓存内存管理
  - 模型执行器内存管理

#### 2.2.8 Thread Pool 模块
- **文件**: `include/cllm/thread_pool/*.h`, `src/thread_pool/*.cpp`
- **职责**:
  - 线程池管理
  - 任务调度
  - 线程状态监控

#### 2.2.9 Batch Management 模块
- **文件**: `include/cllm/batch/*.h`, `src/batch/*.cpp`
- **职责**:
  - 批处理管理
  - 批处理构建
  - 批处理执行统计

## 3. 核心组件详细设计

### 3.1 HTTP Server

#### 3.1.1 类设计
```cpp
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/http/handler.h"
#include <string>
#include <thread>
#include <mutex>

struct ServerStats {
    long long totalRequests;        ///< 总请求数
    long long successfulRequests;   ///< 成功请求数
    long long failedRequests;       ///< 失败请求数
    float averageResponseTime;      ///< 平均响应时间
    int currentConnections;         ///< 当前连接数
    
    ServerStats() : totalRequests(0), successfulRequests(0), failedRequests(0),
                    averageResponseTime(0.0f), currentConnections(0) {}
};

class HttpServer {
public:
    HttpServer(const std::string& host, int port);
    ~HttpServer();
    
    void start();
    void stop();
    
    void setHandler(HttpHandler* handler);
    bool isRunning() const;
    std::string getAddress() const;
    int getPort() const;
    ServerStats getStats() const;
    
private:
    // 内部实现细节
};
```

#### 3.1.2 API 端点
- `POST /generate`: 非流式生成
- `POST /generate_stream`: 流式生成
- `GET /health`: 健康检查
- `GET /stats`: 统计信息
- `POST /encode`: 文本编码

#### 3.1.3 关键组件
- **请求处理**: 通过 HttpHandler 接口处理不同类型的请求
- **响应构建**: 使用 ResponseBuilder 构建标准化响应
- **请求验证**: RequestValidator 确保请求参数合法
- **端点管理**: 支持多种 API 端点，包括生成、编码和健康检查

### 3.2 Scheduler

#### 3.2.1 类设计
```cpp
#include "cllm/scheduler/config.h"
#include "cllm/scheduler/stats.h"
#include "cllm/scheduler/tracker.h"
#include "cllm/scheduler/batch_processor.h"
#include "cllm/common/queue.h"
#include "cllm/common/request_state.h"
#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/memory/monitor.h"

enum class SchedulerError {
    SCHEDULER_NOT_RUNNING,       ///< 调度器未运行
    REQUEST_NOT_FOUND,           ///< 请求未找到
    REQUEST_TIMEOUT,             ///< 请求超时
    BATCH_PROCESSING_FAILED,     ///< 批处理失败
    INVALID_REQUEST              ///< 无效请求
};

class SchedulerException : public std::runtime_error {
public:
    SchedulerException(SchedulerError error, const std::string& message)
        : std::runtime_error(message), error_(error) {}
    
    SchedulerError getError() const { return error_; }
    
private:
    SchedulerError error_;  ///< 错误类型
};

class Scheduler {
public:
    Scheduler(
        const std::string& modelPath,
        const std::string& quantization = "",
        size_t maxBatchSize = 8,
        size_t maxContextLength = 2048
    );
    ~Scheduler();
    
    void start();
    void stop();
    
    // 其他公共方法
    
private:
    void schedulerLoop();
    void processBatch();
    void updateRequests();
    
    SchedulerConfig config_;
    SchedulerStats stats_;
    RequestTracker tracker_;
    BatchProcessor batchProcessor_;
    
    std::unique_ptr<ModelExecutor> modelExecutor_;
    std::unique_ptr<KVCache> kvCache_;
    std::unique_ptr<BatchManager> batchManager_;
    
    RequestQueue requestQueue_;
    std::map<std::string, RequestState> requestStates_;
    
    std::thread schedulerThread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
};
```

#### 3.2.2 调度器组件
- **配置管理**: SchedulerConfig 管理调度器参数
- **统计信息**: SchedulerStats 收集调度器运行统计
- **请求跟踪**: RequestTracker 跟踪请求状态和生命周期
- **批处理处理器**: BatchProcessor 负责批处理的构建和执行
- **请求队列**: 管理待处理的请求

#### 3.2.3 核心流程
1. 请求入队: 接收请求并加入请求队列
2. 请求调度: 按策略选择请求进行批处理
3. 批处理构建: 将多个请求合并为一个批处理
4. 模型执行: 调用 ModelExecutor 执行推理
5. 结果处理: 处理推理结果并更新请求状态
6. 请求完成: 返回结果并清理资源
```

### 3.3 Model Executor

#### 3.3.1 类设计
```cpp
#include "cllm/model/config.h"
#include "cllm/model/stats.h"
#include "cllm/model/exceptions.h"
#include "cllm/model/quantization_manager.h"
#include "cllm/model/inference_optimizer.h"
#include "cllm/model/batch_processor.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/memory/float_array.h"
#include <string>
#include <vector>
#include <mutex>
#include <memory>

namespace cllm {

class Sampler;
class KVCache;

class ModelExecutor {
public:
    ModelExecutor(
        const std::string& modelPath,
        const std::string& quantization = "",
        bool enableSIMD = true
    );
    ~ModelExecutor();
    
    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;
    
    BatchOutput forward(const BatchInput& input);
    
    std::vector<int> generate(
        const std::vector<int>& inputIds,
        size_t maxNewTokens = 100,
        float temperature = 0.7f
    );
    
    int sampleToken(const std::vector<int>& inputIds, float temperature = 0.7f);
    
    void loadModel();
    void unloadModel();
    
    ModelStats getStats() const;
    void resetStats();
    
    Sampler* getSampler() const;
    KVCache* getKVCache() const;
    const ModelConfig& getConfig() const;
    
private:
    void _loadFullPrecisionModel();
    void _loadInt8QuantizedModel();
    void _loadInt4QuantizedModel();
    
    void _applyInferenceOptimizations();
    void _warmupModel();
    
    FloatArray _prepareInput(const std::vector<int>& inputIds);
    void _processOutput(FloatArray& logits, size_t batchSize, size_t vocabSize);
    
    FloatArray _executeModelInference(const FloatArray& inputTensor, size_t batchSize, size_t maxSeqLength);
    
    void _optimizeMemoryUsage();
    void _enableMemoryCompression();
    
    std::string modelPath_;
    std::string quantization_;
    bool enableSIMD_;
    
    void* modelHandle_;
    void* modelWeights_;
    size_t modelSize_;
    
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<KVCache> kvCache_;
    std::unique_ptr<QuantizationManager> quantizationManager_;
    std::unique_ptr<InferenceOptimizer> inferenceOptimizer_;
    std::unique_ptr<BatchProcessor> batchProcessor_;
    
    ModelConfig config_;
    ModelStats stats_;
    
    mutable std::mutex modelMutex_;
    
    bool isModelLoaded_;
    
    FloatArray inferenceBuffer_;
    FloatArray inputBuffer_;
};

} // namespace cllm
```

#### 3.3.2 关键功能
- **模型加载**: 支持从指定路径加载模型文件
- **前向传播**: 接收 BatchInput 并返回 BatchOutput 结果
- **文本生成**: 根据输入 Token ID 生成新的 Token 序列
- **量化支持**: 支持 int8 和 int4 量化
- **推理优化**: 支持 SIMD 优化

#### 3.3.3 批处理支持
- 使用 BatchProcessor 处理批量请求
- 支持动态批处理大小
- 优化批处理执行效率

### 3.4 Tokenizer

#### 3.4.1 类设计
```cpp
#include <string>
#include <vector>

namespace cllm {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& modelPath);
    ~Tokenizer();
    
    std::vector<int> encode(const std::string& text, bool addSpecialTokens = false);
    std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true);
    
    int getVocabSize() const;
    std::string getTokenText(int tokenId) const;
    bool isSpecialToken(int tokenId) const;
    
    void setPadToken(int tokenId);
    void setEosToken(int tokenId);
    void setBosToken(int tokenId);
    
    int getPadToken() const;
    int getEosToken() const;
    int getBosToken() const;
    
    void loadModel(const std::string& modelPath);
    void unloadModel();
    bool isLoaded() const;
    
private:
    void* tokenizerHandle_;      ///< Tokenizer句柄
    std::string modelPath_;      ///< 模型路径
    
    int padTokenId_;             ///< 填充token ID
    int eosTokenId_;             ///< 结束token ID
    int bosTokenId_;             ///< 开始token ID
    
    bool loaded_;                ///< 加载状态
};

} // namespace cllm
```

### 3.5 Sampler

#### 3.5.1 类设计
```cpp
#include "cllm/memory/float_array.h"
#include <vector>

namespace cllm {

class Sampler {
public:
    /**
     * @brief 构造函数
     */
    Sampler();
    
    /**
     * @brief 析构函数
     */
    ~Sampler();
    
    /**
     * @brief 从logits分布中采样token
     * @param logits 模型输出的logits分布
     * @param temperature 温度参数，默认1.0
     * @return 采样得到的token ID
     */
    int sample(const FloatArray& logits, float temperature = 1.0f);
    
private:
    /**
     * @brief 贪心采样（选择概率最高的token）
     * @param logits logits分布
     * @return token ID
     */
    int sampleGreedy(const FloatArray& logits);
    
    /**
     * @brief 温度采样
     * @param logits logits分布
     * @param temperature 温度参数
     * @return token ID
     */
    int sampleTemperature(const FloatArray& logits, float temperature);
    
    unsigned int seed_;  ///< 随机数种子
};

} // namespace cllm
```

#### 3.5.2 关键功能
- **单Token采样**: 从logits中采样单个Token
- **贪心采样**: 选择概率最高的Token
- **温度采样**: 调整采样的随机性

### 3.6 KV Cache

#### 3.6.1 类设计
```cpp
#include "cllm/kv_cache/entry.h"
#include "cllm/kv_cache/stats.h"
#include "cllm/memory/float_array.h"
#include <map>
#include <list>
#include <mutex>
#include <cstddef>

namespace cllm {

class KVCache {
public:
    /**
     * @brief 构造函数
     * @param maxSize 最大缓存数量，默认10
     * @param maxMemoryMB 最大内存限制(MB)，0表示不限制
     */
    explicit KVCache(size_t maxSize = 10, size_t maxMemoryMB = 0);
    
    /**
     * @brief 析构函数
     */
    ~KVCache();
    
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    /**
     * @brief 获取缓存条目
     * @param sequenceId 序列ID
     * @param entry 输出参数，缓存条目
     * @return true 如果找到，false 否则
     */
    bool get(size_t sequenceId, KVCacheEntry& entry);
    
    /**
     * @brief 存储缓存条目
     * @param sequenceId 序列ID
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     */
    void put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache);
    
    /**
     * @brief 移除缓存条目
     * @param sequenceId 序列ID
     */
    void remove(size_t sequenceId);
    
    /**
     * @brief 清空缓存
     */
    void clear();
    
    /**
     * @brief 获取缓存大小
     * @return 缓存条目数
     */
    size_t size() const;
    
    /**
     * @brief 判断是否包含指定序列
     * @param sequenceId 序列ID
     * @return true 如果包含，false 否则
     */
    bool contains(size_t sequenceId) const;
    
    /**
     * @brief 获取统计信息
     * @return 缓存统计信息
     */
    CacheStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
    /**
     * @brief 获取内存使用量
     * @return 内存使用量(MB)
     */
    float getMemoryUsageMB() const;
    
    /**
     * @brief 获取最大缓存数量
     * @return 最大缓存数量
     */
    size_t getMaxSize() const;
    
    /**
     * @brief 获取最大内存限制
     * @return 最大内存限制(MB)
     */
    size_t getMaxMemoryMB() const;
    
    /**
     * @brief 增量更新缓存
     * @param sequenceId 序列ID
     * @param newKeyPart 新的键部分
     * @param newValuePart 新的值部分
     */
    void updateIncremental(
        size_t sequenceId,
        const FloatArray& newKeyPart,
        const FloatArray& newValuePart
    );
    
private:
    void evictOldest();  ///< 淘汰最旧条目
    void ensureMemoryLimit();  ///< 确保内存不超限
    float calculateMemoryUsage(const FloatArray& keyCache, const FloatArray& valueCache);  ///< 计算内存使用量
    
    std::map<size_t, KVCacheEntry> cache_;  ///< 缓存映射表
    std::list<size_t> accessList_;          ///< LRU访问列表
    
    size_t maxSize_;          ///< 最大缓存数量
    size_t maxMemoryMB_;      ///< 最大内存限制(MB)
    float memoryUsage_;       ///< 当前内存使用量(MB)
    
    mutable std::mutex cacheMutex_;  ///< 缓存互斥锁
    
    CacheStats stats_;  ///< 统计信息
};

} // namespace cllm
```

#### 3.6.2 缓存条目
```cpp
#include "cllm/memory/float_array.h"
#include <cstddef>

namespace cllm {

struct KVCacheEntry {
    FloatArray keyCache;       ///< 键缓存数据
    FloatArray valueCache;     ///< 值缓存数据
    size_t sequenceId;         ///< 序列ID
    size_t lastAccessTime;     ///< 最后访问时间
    size_t hitCount;           ///< 命中次数
    size_t createdTime;        ///< 创建时间
    size_t lastUpdateTime;     ///< 最后更新时间
    float memoryUsage;         ///< 内存使用量(MB)
    size_t accessCount;        ///< 访问总次数
    
    /**
     * @brief 默认构造函数
     */
    KVCacheEntry()
        : sequenceId(0)
        , lastAccessTime(0)
        , hitCount(0)
        , createdTime(0)
        , lastUpdateTime(0)
        , memoryUsage(0.0f)
        , accessCount(0) {}
    
    /**
     * @brief 构造函数
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     * @param sequenceId 序列ID
     */
    KVCacheEntry(
        const FloatArray& keyCache,
        const FloatArray& valueCache,
        size_t sequenceId
    ) : keyCache(keyCache)
        , valueCache(valueCache)
        , sequenceId(sequenceId)
        , lastAccessTime(0)
        , hitCount(0)
        , createdTime(0)
        , lastUpdateTime(0)
        , memoryUsage(0.0f)
        , accessCount(0) {
        updateMemoryUsage();
    }
    
    /**
     * @brief 更新访问时间和计数器
     */
    void updateAccess() {
        lastAccessTime = getCurrentTime();
        hitCount++;
        accessCount++;
    }
    
    /**
     * @brief 更新内存使用量统计
     */
    void updateMemoryUsage() {
        memoryUsage = calculateMemoryUsage();
    }
    
private:
    size_t getCurrentTime() const {
        static size_t counter = 0;
        return ++counter;
    }
    
    float calculateMemoryUsage() const {
        size_t totalSize = keyCache.size() + valueCache.size();
        return static_cast<float>(totalSize * sizeof(float)) / (1024.0f * 1024.0f);
    }
};

} // namespace cllm
```

#### 3.6.3 关键功能
- **LRU缓存策略**: 使用最近最少使用算法管理缓存条目
- **内存限制**: 支持设置最大内存限制，自动驱逐超出限制的条目
- **线程安全**: 支持并发访问
- **统计功能**: 记录缓存命中率、访问次数等统计信息
- **自动驱逐**: 根据大小和内存限制自动驱逐旧条目

## 4. 性能优化策略

### 4.1 内存管理

#### 4.1.1 成熟的第三方内存管理库调研

经过调研，我们推荐使用以下成熟的第三方内存管理库：

| 库名称 | 开发者 | 特点 | 适用场景 |
|--------|--------|------|----------|
| **mimalloc** | 微软 | 高性能、跨平台、线程安全 | **推荐使用**，性能最优 |
| **TCMalloc** | Google | 优化小块内存分配、低碎片 | 高并发场景 |
| **Jemalloc** | Facebook | 可扩展、低延迟 | 多核服务器环境 |

**推荐选择：mimalloc**

选择理由：
- 性能显著优于系统分配器和tcmalloc（20-30%吞吐量提升）
- 跨平台支持（Linux、macOS、Windows）
- 线程安全，适合高并发场景
- 微软开源，维护活跃
- 集成简单，只需链接库即可
- 自动内存对齐，支持SIMD优化

#### 4.1.2 内存管理设计原则

- 使用mimalloc替代系统malloc/free
- 使用RAII包装器管理资源生命周期，避免内存泄漏
- 实现简单的内存监控和限制机制
- 内存对齐以支持SIMD优化（mimalloc自动处理）
- 避免频繁的内存分配和释放

#### 4.1.3 RAII包装器设计

##### RAII概念

**RAII (Resource Acquisition Is Initialization)** 是一种C++编程技术，核心思想是：
- **资源的获取**在对象的构造函数中进行
- **资源的释放**在对象的析构函数中进行
- 利用C++对象的生命周期**自动管理资源**

##### RAII的优势

1. **自动资源释放**：对象离开作用域时自动调用析构函数
2. **异常安全**：即使发生异常，资源也能被正确释放
3. **代码简洁**：不需要手动调用释放函数
4. **避免内存泄漏**：不会忘记释放资源

##### 数组RAII包装器 (float_array.h)
```cpp
#pragma once

#include <cstddef>

class FloatArray {
public:
    explicit FloatArray(size_t size = 0);
    ~FloatArray();
    
    FloatArray(const FloatArray& other);
    FloatArray& operator=(const FloatArray& other);
    
    FloatArray(FloatArray&& other) noexcept;
    FloatArray& operator=(FloatArray&& other) noexcept;
    
    void resize(size_t newSize);
    
    float* data();
    const float* data() const;
    
    size_t size() const;
    bool empty() const;
    
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    
private:
    float* data_;
    size_t size_;
};
```

##### 数组RAII包装器实现 (float_array.cpp)
```cpp
#include "float_array.h"
#include <stdexcept>
#include <algorithm>

FloatArray::FloatArray(size_t size) : data_(nullptr), size_(size) {
    if (size > 0) {
        data_ = new float[size];
        if (!data_) {
            throw std::runtime_error("Failed to allocate FloatArray");
        }
    }
}

FloatArray::~FloatArray() {
    delete[] data_;
}

FloatArray::FloatArray(const FloatArray& other) : data_(nullptr), size_(other.size_) {
    if (size_ > 0) {
        data_ = new float[size_];
        if (!data_) {
            throw std::runtime_error("Failed to allocate FloatArray");
        }
        std::copy(other.data_, other.data_ + size_, data_);
    }
}

FloatArray& FloatArray::operator=(const FloatArray& other) {
    if (this != &other) {
        delete[] data_;
        data_ = nullptr;
        size_ = other.size_;
        
        if (size_ > 0) {
            data_ = new float[size_];
            if (!data_) {
                throw std::runtime_error("Failed to allocate FloatArray");
            }
            std::copy(other.data_, other.data_ + size_, data_);
        }
    }
    return *this;
}

FloatArray::FloatArray(FloatArray&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

FloatArray& FloatArray::operator=(FloatArray&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void FloatArray::resize(size_t newSize) {
    if (newSize == size_) {
        return;
    }
    
    float* newData = nullptr;
    if (newSize > 0) {
        newData = new float[newSize];
        if (!newData) {
            throw std::runtime_error("Failed to allocate FloatArray");
        }
        if (data_ != nullptr) {
            std::copy(data_, data_ + std::min(size_, newSize), newData);
        }
    }
    
    delete[] data_;
    data_ = newData;
    size_ = newSize;
}

float* FloatArray::data() {
    return data_;
}

const float* FloatArray::data() const {
    return data_;
}

size_t FloatArray::size() const {
    return size_;
}

bool FloatArray::empty() const {
    return size_ == 0;
}

float& FloatArray::operator[](size_t index) {
    if (index >= size_) {
        throw std::out_of_range("FloatArray index out of range");
    }
    return data_[index];
}

const float& FloatArray::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("FloatArray index out of range");
    }
    return data_[index];
}
```

##### 智能指针使用示例
```cpp
#include <memory>
#include "cllm/memory/float_array.h"
#include "cllm/model/executor.h"

void process_data() {
    FloatArray array(1000);
    
    for (size_t i = 0; i < array.size(); ++i) {
        array[i] = i * 1.0f;
    }
    
    // 函数结束时自动释放内存，无需手动delete
}

void process_with_smart_ptr() {
    std::unique_ptr<ModelExecutor> executor = std::make_unique<ModelExecutor>("path/to/model.gguf");
    
    // 执行推理
    std::vector<int> input = {1, 2, 3};
    std::vector<int> output = executor->generate(input, 10);
    
    // 函数结束时自动释放executor，无需手动delete
}

void safe_operation() {
    FloatArray array(100);
    
    if (some_error_condition) {
        return;  // 安全，array的析构函数会自动释放内存
    }
    
    // 正常处理
}
```

##### 组件资源管理示例
```cpp
class ModelExecutor {
private:
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<KVCache> kvCache_;
    std::unique_ptr<QuantizationManager> quantizationManager_;
    std::unique_ptr<InferenceOptimizer> inferenceOptimizer_;
    
public:
    ModelExecutor() {
        sampler_ = std::make_unique<Sampler>();
        kvCache_ = std::make_unique<KVCache>(100, 4096);
        quantizationManager_ = std::make_unique<QuantizationManager>();
        inferenceOptimizer_ = std::make_unique<InferenceOptimizer>();
    }
    
    // 析构函数自动释放所有资源，无需手动清理
    ~ModelExecutor() = default;
    
    // 禁止拷贝，防止资源所有权混乱
    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;
    
    // 允许移动
    ModelExecutor(ModelExecutor&&) noexcept = default;
    ModelExecutor& operator=(ModelExecutor&&) noexcept = default;
};
```

#### 4.1.4 mimalloc集成配置

##### CMakeLists.txt配置
```cmake
# 查找mimalloc库
find_package(mimalloc REQUIRED)

# 链接mimalloc
target_link_libraries(cLLM PRIVATE mimalloc)

# 或者使用FetchContent自动下载
include(FetchContent)
FetchContent_Declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG v2.1.2
)
FetchContent_MakeAvailable(mimalloc)
```

##### 使用示例
```cpp
#include <mimalloc.h>

// 初始化mimalloc（可选，通常自动初始化）
void init_memory_allocator() {
    mi_stats_reset();  // 重置统计信息
}

// 获取内存使用统计
void print_memory_stats() {
    mi_stats_print(NULL);  // 打印内存统计信息
}
```

#### 4.1.5 内存监控和限制机制

##### 内存监控器 (memory_monitor.h)
```cpp
#pragma once
#include <atomic>
#include <functional>

class MemoryMonitor {
public:
    typedef std::function<void(size_t used, size_t limit)> MemoryLimitCallback;
    
    static MemoryMonitor& instance();
    
    void set_limit(size_t limit_bytes);
    size_t get_limit() const;
    
    void allocate(size_t bytes);
    void deallocate(size_t bytes);
    
    size_t get_used() const;
    size_t get_peak() const;
    
    void set_limit_callback(MemoryLimitCallback callback);
    void reset_peak();
    
private:
    MemoryMonitor() {
        used_memory_ = 0;
        peak_memory_ = 0;
        memory_limit_ = 0;
    }
    
    std::atomic<size_t> used_memory_;
    std::atomic<size_t> peak_memory_;
    std::atomic<size_t> memory_limit_;
    MemoryLimitCallback limit_callback_;
};
```

##### 内存监控器实现 (memory_monitor.cpp)
```cpp
#include "memory_monitor.h"
#include <stdexcept>

MemoryMonitor& MemoryMonitor::instance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::set_limit(size_t limit_bytes) {
    memory_limit_.store(limit_bytes);
}

size_t MemoryMonitor::get_limit() const {
    return memory_limit_.load();
}

void MemoryMonitor::allocate(size_t bytes) {
    size_t limit = memory_limit_.load();
    if (limit > 0) {
        size_t used = used_memory_.load();
        if (used + bytes > limit) {
            if (limit_callback_) {
                limit_callback_(used, limit);
            }
            throw std::runtime_error("Memory limit exceeded");
        }
    }
    
    used_memory_.fetch_add(bytes);
    
    size_t current = used_memory_.load();
    size_t peak = peak_memory_.load();
    while (current > peak) {
        if (peak_memory_.compare_exchange_weak(peak, current)) {
            break;
        }
    }
}

void MemoryMonitor::deallocate(size_t bytes) {
    used_memory_.fetch_sub(bytes);
}

size_t MemoryMonitor::get_used() const {
    return used_memory_.load();
}

size_t MemoryMonitor::get_peak() const {
    return peak_memory_.load();
}

void MemoryMonitor::set_limit_callback(MemoryLimitCallback callback) {
    limit_callback_ = callback;
}

void MemoryMonitor::reset_peak() {
    peak_memory_.store(0);
}
```

#### 4.1.6 KV Cache内存管理

##### KV Cache内存管理器 (kv_cache_memory.h)
```cpp
#pragma once
#include "memory_monitor.h"
#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <string>
#include <chrono>

struct KVCacheEntry {
    std::string request_id;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    size_t sequence_length;
    std::chrono::steady_clock::time_point last_access;
    size_t memory_usage;
};

class KVCacheMemoryManager {
public:
    KVCacheMemoryManager(size_t max_memory_mb);
    
    bool insert(const std::string& request_id, 
                const std::vector<float>& key_cache,
                const std::vector<float>& value_cache);
    
    bool get(const std::string& request_id,
             std::vector<float>& key_cache,
             std::vector<float>& value_cache);
    
    void evict(const std::string& request_id);
    
    size_t get_used_memory() const;
    size_t get_total_memory() const;
    
    void set_eviction_callback(std::function<void(const std::string&)> callback);
    
private:
    void evict_oldest();
    size_t calculate_memory_usage(const std::vector<float>& key_cache,
                                  const std::vector<float>& value_cache);
    
    std::unordered_map<std::string, std::list<KVCacheEntry>::iterator> cache_map_;
    std::list<KVCacheEntry> cache_list_;
    mutable std::mutex mutex_;
    size_t max_memory_bytes_;
    std::function<void(const std::string&)> eviction_callback_;
};
```

##### KV Cache内存管理器实现 (kv_cache_memory.cpp)
```cpp
#include "kv_cache_memory.h"

KVCacheMemoryManager::KVCacheMemoryManager(size_t max_memory_mb)
    : max_memory_bytes_(max_memory_mb * 1024 * 1024) {}

bool KVCacheMemoryManager::insert(const std::string& request_id,
                                  const std::vector<float>& key_cache,
                                  const std::vector<float>& value_cache) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t memory_usage = calculate_memory_usage(key_cache, value_cache);
    
    // 检查是否需要淘汰旧缓存
    size_t current_usage = get_used_memory();
    while (current_usage + memory_usage > max_memory_bytes_ && !cache_list_.empty()) {
        evict_oldest();
        current_usage = get_used_memory();
    }
    
    if (current_usage + memory_usage > max_memory_bytes_) {
        return false;
    }
    
    KVCacheEntry entry;
    entry.request_id = request_id;
    entry.key_cache = key_cache;
    entry.value_cache = value_cache;
    entry.sequence_length = key_cache.size();
    entry.last_access = std::chrono::steady_clock::now();
    entry.memory_usage = memory_usage;
    
    cache_list_.push_front(entry);
    cache_map_[request_id] = cache_list_.begin();
    
    return true;
}

bool KVCacheMemoryManager::get(const std::string& request_id,
                                std::vector<float>& key_cache,
                                std::vector<float>& value_cache) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(request_id);
    if (it == cache_map_.end()) {
        return false;
    }
    
    key_cache = it->second->key_cache;
    value_cache = it->second->value_cache;
    it->second->last_access = std::chrono::steady_clock::now();
    
    return true;
}

void KVCacheMemoryManager::evict(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(request_id);
    if (it != cache_map_.end()) {
        if (eviction_callback_) {
            eviction_callback_(request_id);
        }
        cache_list_.erase(it->second);
        cache_map_.erase(it);
    }
}

size_t KVCacheMemoryManager::get_used_memory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& entry : cache_list_) {
        total += entry.memory_usage;
    }
    return total;
}

size_t KVCacheMemoryManager::get_total_memory() const {
    return max_memory_bytes_;
}

void KVCacheMemoryManager::set_eviction_callback(
    std::function<void(const std::string&)> callback) {
    eviction_callback_ = callback;
}

void KVCacheMemoryManager::evict_oldest() {
    if (cache_list_.empty()) {
        return;
    }
    
    auto& oldest = cache_list_.back();
    if (eviction_callback_) {
        eviction_callback_(oldest.request_id);
    }
    
    cache_map_.erase(oldest.request_id);
    cache_list_.pop_back();
}

size_t KVCacheMemoryManager::calculate_memory_usage(
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache) {
    return (key_cache.size() + value_cache.size()) * sizeof(float);
}
```

#### 4.1.7 Model Executor内存管理

##### Model Executor内存管理器 (executor_memory.h)
```cpp
#pragma once
#include "memory_monitor.h"
#include <vector>
#include <memory>

struct BufferInfo {
    void* ptr;
    size_t size;
    bool is_temp;
};

class ModelExecutorMemoryManager {
public:
    ModelExecutorMemoryManager(size_t max_memory_mb);
    
    void* allocate_temp_buffer(size_t size);
    void deallocate_temp_buffer(void* ptr);
    
    void* allocate_weights_cache(size_t size);
    void deallocate_weights_cache(void* ptr);
    
    size_t get_temp_memory_used() const;
    size_t get_weights_memory_used() const;
    size_t get_total_memory_used() const;
    
    void clear_all();
    
private:
    std::vector<BufferInfo> buffers_;
    size_t max_memory_bytes_;
    size_t temp_memory_used_;
    size_t weights_memory_used_;
};

class TensorBuffer {
public:
    TensorBuffer(size_t size);
    ~TensorBuffer();
    
    float* data();
    const float* data() const;
    size_t size() const;
    
private:
    float* data_;
    size_t size_;
};
```

##### Model Executor内存管理器实现 (executor_memory.cpp)
```cpp
#include "executor_memory.h"
#include <stdexcept>

ModelExecutorMemoryManager::ModelExecutorMemoryManager(size_t max_memory_mb)
    : max_memory_bytes_(max_memory_mb * 1024 * 1024),
      temp_memory_used_(0),
      weights_memory_used_(0) {}

void* ModelExecutorMemoryManager::allocate_temp_buffer(size_t size) {
    if (temp_memory_used_ + size > max_memory_bytes_) {
        throw std::runtime_error("Temp buffer memory limit exceeded");
    }
    
    void* ptr = malloc(size);
    if (!ptr) {
        throw std::runtime_error("Failed to allocate temp buffer");
    }
    
    BufferInfo info;
    info.ptr = ptr;
    info.size = size;
    info.is_temp = true;
    buffers_.push_back(info);
    temp_memory_used_ += size;
    
    return ptr;
}

void ModelExecutorMemoryManager::deallocate_temp_buffer(void* ptr) {
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
        if (it->ptr == ptr && it->is_temp) {
            temp_memory_used_ -= it->size;
            free(it->ptr);
            buffers_.erase(it);
            return;
        }
    }
}

void* ModelExecutorMemoryManager::allocate_weights_cache(size_t size) {
    if (weights_memory_used_ + size > max_memory_bytes_) {
        throw std::runtime_error("Weights cache memory limit exceeded");
    }
    
    void* ptr = malloc(size);
    if (!ptr) {
        throw std::runtime_error("Failed to allocate weights cache");
    }
    
    BufferInfo info;
    info.ptr = ptr;
    info.size = size;
    info.is_temp = false;
    buffers_.push_back(info);
    weights_memory_used_ += size;
    
    return ptr;
}

void ModelExecutorMemoryManager::deallocate_weights_cache(void* ptr) {
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
        if (it->ptr == ptr && !it->is_temp) {
            weights_memory_used_ -= it->size;
            free(it->ptr);
            buffers_.erase(it);
            return;
        }
    }
}

size_t ModelExecutorMemoryManager::get_temp_memory_used() const {
    return temp_memory_used_;
}

size_t ModelExecutorMemoryManager::get_weights_memory_used() const {
    return weights_memory_used_;
}

size_t ModelExecutorMemoryManager::get_total_memory_used() const {
    return temp_memory_used_ + weights_memory_used_;
}

void ModelExecutorMemoryManager::clear_all() {
    for (auto& buffer : buffers_) {
        free(buffer.ptr);
    }
    buffers_.clear();
    temp_memory_used_ = 0;
    weights_memory_used_ = 0;
}

TensorBuffer::TensorBuffer(size_t size)
    : data_(nullptr), size_(size) {
    data_ = (float*)malloc(size * sizeof(float));
    if (!data_) {
        throw std::runtime_error("Failed to allocate tensor buffer");
    }
}

TensorBuffer::~TensorBuffer() {
    if (data_) {
        free(data_);
    }
}

float* TensorBuffer::data() {
    return data_;
}

const float* TensorBuffer::data() const {
    return data_;
}

size_t TensorBuffer::size() const {
    return size_;
}
```

#### 4.1.8 内存管理最佳实践

1. **使用mimalloc替代系统分配器**:
   - 自动优化内存分配性能
   - 自动处理内存对齐
   - 线程安全，适合高并发

2. **使用RAII包装器管理资源**:
   - FloatArray用于管理浮点数组
   - RAIIWrapper用于管理单个对象
   - 避免手动delete，防止内存泄漏

3. **内存监控**:
   - 设置合理的内存限制
   - 实现内存超限回调
   - 定期检查内存使用情况

4. **内存管理配置**:
   - 使用KV Cache管理器管理缓存
   - 使用TensorBuffer管理张量内存
   - 定期清理未使用的内存

5. **性能优化**:
   - 减少内存分配次数
   - 复用已分配的内存
   - 避免频繁的内存拷贝
   - 使用mimalloc的统计功能监控内存使用

### 4.2 并发优化
- 使用无锁数据结构（如 lock-free queue）
- 细粒度锁减少锁竞争
- 读写锁分离读写操作
- CPU 亲和性绑定线程到特定核心

### 4.3 推理优化
- 批处理优化：动态调整批次大小
- 量化支持：int8、fp16 量化
- 算子融合：减少内存访问
- 内存复用：减少内存拷贝

### 4.4 缓存优化
- KV 缓存复用
- Tokenizer 结果缓存
- 模型权重缓存
- 预计算常用模式

### 4.5 SIMD 优化
- **向量化计算**: 使用SIMD指令并行处理多个数据元素
- **矩阵运算优化**: 在矩阵乘法、点积等操作中使用SIMD加速
- **激活函数优化**: 向量化实现ReLU、GELU等激活函数
- **归一化优化**: SIMD加速LayerNorm、BatchNorm等归一化操作
- **采样优化**: 在logits计算和softmax中使用SIMD指令
- **内存对齐**: 确保数据内存对齐以获得最佳SIMD性能
- **自动向量化**: 利用编译器的自动向量化优化

## 4.6 SIMD 优化实现详解

### 4.6.1 SIMD 工具类设计

#### SIMD 工具类 (simd_utils.h)
```cpp
#pragma once
```

## 5. 主程序设计

### 5.1 主程序概述

主程序是cLLM系统的入口点，负责协调各个组件的初始化、配置和运行。它提供了命令行参数解析、组件生命周期管理、错误处理和优雅关闭机制。

### 5.2 主程序架构

主程序采用`Application`类封装所有功能，实现了以下核心职责：

1. **命令行参数解析**：使用cxxopts库解析用户输入的配置参数
2. **组件初始化**：按依赖顺序初始化各核心组件
3. **组件协调**：管理组件间的依赖关系和交互
4. **生命周期管理**：提供启动、运行和优雅关闭机制
5. **错误处理**：统一的错误处理和日志记录
6. **信号处理**：支持优雅关闭（SIGINT/SIGTERM）

### 5.3 主程序类设计

```cpp
class Application {
public:
    Application();
    ~Application();

    bool initialize(int argc, char** argv);
    void start();
    void stop();
    void waitForExit();

private:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8080;
        std::string model_path;
        std::string tokenizer_path;
        int num_threads = 4;
        int max_batch_size = 32;
        int queue_size = 1000;
        int kv_cache_size_mb = 1024;
        std::string log_level = "info";
        bool enable_quantization = false;
    } config_;

    // Application components
    std::unique_ptr<HttpServer> http_server_;
    std::unique_ptr<HttpHandler> http_handler_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelExecutor> model_executor_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<Sampler> sampler_;
    bool running_;

    bool parseArguments(int argc, char** argv);
    bool initializeTokenizer();
    bool initializeKVCache();
    bool initializeSampler();
    bool initializeModelExecutor();
    bool initializeScheduler();
    bool initializeHTTPComponents();
    void registerEndpoints();
};
```

### 5.4 初始化流程

主程序的初始化流程按照组件依赖关系顺序进行：

1. **命令行参数解析**：解析并验证用户输入的配置参数
2. **日志初始化**：设置日志级别和格式
3. **Tokenizer初始化**：加载分词器模型
4. **KV Cache初始化**：创建KV缓存管理器
5. **Sampler初始化**：创建采样器实例
6. **Model Executor初始化**：加载模型并创建执行器
7. **Scheduler初始化**：创建请求调度器
8. **HTTP组件初始化**：创建HTTP服务器和处理器
9. **API端点注册**：注册RESTful API端点

### 5.5 组件依赖关系

```
Application
├─ Config (命令行参数)
├─ Logger
├─ Tokenizer
├─ KVCache
├─ Sampler
├─ ModelExecutor
│  └─ ThreadPool
├─ Scheduler
│  └─ ModelExecutor
└─ HttpServer
   ├─ HttpHandler
   ├─ HealthEndpoint
   ├─ GenerateEndpoint
   │  ├─ Scheduler
   │  └─ Tokenizer
   └─ EncodeEndpoint
      └─ Tokenizer
```

### 5.6 命令行参数

主程序支持以下命令行参数：

| 参数 | 缩写 | 类型 | 默认值 | 描述 |
|------|------|------|--------|------|
| --host | -h | string | 0.0.0.0 | 服务器绑定地址 |
| --port | -p | int | 8080 | 服务器监听端口 |
| --model | -m | string | 必填 | 模型文件路径 |
| --tokenizer | -t | string | 必填 | 分词器模型路径 |
| --num_threads | -n | int | 4 | 工作线程数量 |
| --max_batch_size | -b | int | 32 | 最大批处理大小 |
| --queue_size | -q | int | 1000 | 请求队列大小 |
| --kv_cache_size | -c | int | 1024 | KV缓存大小(MB) |
| --log_level | -l | string | info | 日志级别(debug/info/warn/error) |
| --quantize | -q | bool | false | 启用模型量化 |
| --help | | | | 显示帮助信息 |

### 5.7 启动流程

1. **创建Application实例**：创建主程序实例
2. **初始化Application**：调用initialize()方法解析参数并初始化组件
3. **启动组件**：按顺序启动Scheduler和HttpServer
4. **进入等待状态**：等待用户信号或错误发生

### 5.8 优雅关闭

主程序支持通过SIGINT（Ctrl+C）和SIGTERM信号进行优雅关闭：

1. **捕获信号**：注册信号处理器
2. **停止组件**：按相反顺序停止各组件
3. **清理资源**：释放各组件占用的资源
4. **退出程序**：返回退出码

### 5.9 错误处理

主程序采用统一的错误处理机制：

1. **异常捕获**：捕获所有标准异常
2. **日志记录**：使用spdlog记录详细错误信息
3. **用户反馈**：向用户输出简洁的错误信息
4. **优雅恢复**：在可能的情况下继续运行，否则关闭程序

### 5.10 内存管理

主程序使用RAII原则管理所有资源：

1. **智能指针**：使用std::unique_ptr管理组件生命周期
2. **自动释放**：组件在Application析构时自动释放
3. **资源限制**：通过配置参数限制各组件的资源使用
4. **内存监控**：集成MemoryMonitor进行内存使用监控

### 5.11 性能考虑

1. **组件初始化顺序**：按依赖关系顺序初始化，避免不必要的等待
2. **线程池大小**：根据CPU核心数和工作负载动态调整
3. **批处理优化**：通过max_batch_size参数控制批处理大小
4. **缓存管理**：合理设置KV缓存大小，平衡性能和内存使用

### 5.12 代码结构

主程序代码位于`src/main.cpp`，包含以下主要部分：

1. **Application类定义**：主程序核心逻辑
2. **信号处理函数**：处理优雅关闭信号
3. **main函数**：程序入口点

### 5.13 使用示例

```bash
# 基本使用
./cllm --model ./models/llama2-7b.gguf --tokenizer ./models/tokenizer.model

# 自定义配置
./cllm --model ./models/llama2-7b.gguf --tokenizer ./models/tokenizer.model --port 9000 --num_threads 8 --kv_cache_size 2048

# 启用量化
./cllm --model ./models/llama2-7b.gguf --tokenizer ./models/tokenizer.model --quantize

# 查看帮助
./cllm --help
```

## 6. 部署与运行

### 6.1 构建

```bash
cd /path/to/xllm/cpp/cLLM
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 6.2 运行

```bash
./cllm --model /path/to/model.gguf --tokenizer /path/to/tokenizer.model
```

### 6.3 测试

```bash
# 健康检查
curl http://localhost:8080/health

# 文本生成
curl -X POST http://localhost:8080/generate -H "Content-Type: application/json" -d '{"prompt":"Hello, world","max_new_tokens":50}'

# 文本编码
curl -X POST http://localhost:8080/encode -H "Content-Type: application/json" -d '{"text":"Hello, world"}'
```

### 6.4 性能监控

主程序提供了以下监控方式：

1. **日志输出**：实时记录系统状态和错误信息
2. **健康检查端点**：提供系统状态检查
3. **内存监控**：集成MemoryMonitor记录内存使用情况
4. **组件统计**：各组件内置统计功能
#include <immintrin.h>
#include <cstdint>
#include <memory>

namespace simd {

// SIMD对齐分配器
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    
    AlignedAllocator() noexcept {}
    template<typename U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    
    pointer allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* p = nullptr;
        if (posix_memalign(&p, 64, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }
    
    void deallocate(pointer p, size_t) noexcept {
        free(p);
    }
};

// SIMD对齐向量
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// CPU特性检测
class CPUFeatures {
public:
    static bool hasAVX2();
    static bool hasAVX512F();
    static bool hasSSE42();
    static const char* getCPUName();
};

} // namespace simd
```

#### SIMD 操作类 (simd_ops.h)
```cpp
#pragma once
#include "simd_utils.h"
#include <Eigen/Dense>

namespace simd {

// SIMD优化的softmax
class SoftmaxSIMD {
public:
    static void compute_avx2(const float* input, float* output, size_t size);
    static void compute_avx512(const float* input, float* output, size_t size);
    static void compute_scalar(const float* input, float* output, size_t size);
    
    static void compute(const Eigen::VectorXf& input, Eigen::VectorXf& output);
};

// SIMD优化的矩阵乘法
class MatMulSIMD {
public:
    static void compute_avx2(const float* A, const float* B, float* C, 
                            size_t M, size_t N, size_t K);
    static void compute_avx512(const float* A, const float* B, float* C, 
                              size_t M, size_t N, size_t K);
};

// SIMD优化的激活函数
class ActivationSIMD {
public:
    static void relu_avx2(const float* input, float* output, size_t size);
    static void relu_avx512(const float* input, float* output, size_t size);
    
    static void gelu_avx2(const float* input, float* output, size_t size);
    static void gelu_avx512(const float* input, float* output, size_t size);
};

// SIMD优化的LayerNorm
class LayerNormSIMD {
public:
    static void compute_avx2(const float* input, const float* gamma, const float* beta,
                            float* output, size_t size, float eps);
    static void compute_avx512(const float* input, const float* gamma, const float* beta,
                              float* output, size_t size, float eps);
};

} // namespace simd
```

#### AVX-512 专用优化 (avx512_ops.h)
```cpp
#pragma once
#include <immintrin.h>
#include <cmath>

namespace simd {

// AVX-512优化的点积
inline float dot_product_avx512(const float* a, const float* b, size_t size) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    
    // 处理512位对齐的数据（16个float）
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    // 处理剩余数据
    float result = _mm512_reduce_add_ps(sum);
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// AVX-512优化的向量加法
inline void vector_add_avx512(const float* a, const float* b, float* result, size_t size) {
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        _mm512_store_ps(result + i, _mm512_add_ps(va, vb));
    }
    
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

// AVX-512优化的指数函数（用于softmax）
inline void exp_avx512(const float* input, float* output, size_t size) {
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_load_ps(input + i);
        __m512 exp_v = _mm512_exp_ps(v);
        _mm512_store_ps(output + i, exp_v);
    }
    
    for (; i < size; ++i) {
        output[i] = std::exp(input[i]);
    }
}

} // namespace simd
```

### 4.6.2 SIMD 优化应用场景

#### 1. Sampler 中的 SIMD 优化
```cpp
Eigen::VectorXf Sampler::softmax_simd(const Eigen::VectorXf& logits) {
    Eigen::VectorXf result(logits.size());
    
    if (enable_simd_ && CPUFeatures::hasAVX512F()) {
        simd::SoftmaxSIMD::compute(logits, result);
    } else if (enable_simd_ && CPUFeatures::hasAVX2()) {
        simd::SoftmaxSIMD::compute(logits, result);
    } else {
        // 回退到标量实现
        float max_val = logits.maxCoeff();
        Eigen::VectorXf exp_logits = (logits.array() - max_val).exp();
        result = exp_logits / exp_logits.sum();
    }
    
    return result;
}
```

#### 2. Model Executor 中的 SIMD 优化
```cpp
class ModelExecutor {
private:
    void apply_simd_optimizations() {
        if (CPUFeatures::hasAVX512F()) {
            logger_->info("Enabling AVX-512 optimizations");
            use_avx512_ = true;
        } else if (CPUFeatures::hasAVX2()) {
            logger_->info("Enabling AVX2 optimizations");
            use_avx2_ = true;
        }
    }
    
    bool use_avx512_ = false;
    bool use_avx2_ = false;
};
```

### 4.6.3 SIMD 性能优化策略

1. **内存对齐**: 使用64字节对齐（AVX-512缓存行大小）
2. **数据布局**: 使用SoA（Structure of Arrays）而非AoS（Array of Structures）
3. **循环展开**: 手动展开循环以减少分支预测开销
4. **预取**: 使用预取指令减少缓存未命中
5. **编译器优化**: 启用编译器自动向量化

### 4.6.4 编译器优化标志

```cmake
# CMakeLists.txt 中的优化选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /fp:fast")
endif()
```

### 4.6.5 SIMD 性能测试

```cpp
// 性能测试工具类
class SIMDPerformanceBenchmark {
public:
    static void benchmark_softmax(size_t size, int iterations = 1000);
    static void benchmark_matmul(size_t M, size_t N, size_t K, int iterations = 100);
    static void benchmark_activation(size_t size, int iterations = 1000);
    
    static void compare_scalar_vs_simd();
};
```

## 5. 并发设计

### 5.1 并发架构概述

基于Python版本的异步并发实现，C++版本将采用多线程+任务队列的并发架构，实现高效的请求处理和批处理优化。

#### 5.1.1 并发设计原则

1. **最大化CPU利用率**: 充分利用多核CPU，实现并行推理
2. **最小化锁竞争**: 使用无锁数据结构和细粒度锁，减少线程阻塞
3. **批处理优化**: 动态批处理策略，提高推理吞吐量
4. **优先级调度**: 基于等待时间和请求长度的智能调度
5. **资源隔离**: 推理线程与I/O线程分离，避免相互影响

#### 5.1.2 并发架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP Server Layer                           │
│              (自研 HTTP Server - epoll/kqueue)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Request 1   │  │  Request 2   │  │  Request N   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Request Scheduler                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Priority Queue (Thread-Safe)                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │  Req 1  │  │  Req 2  │  │  Req 3  │  │  Req N  │     │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │   │
│  │       │            │            │            │           │   │
│  └───────┼────────────┼────────────┼────────────┼───────────┘   │
│          │            │            │            │               │
│  ┌───────▼────────────▼────────────▼────────────▼───────────┐   │
│  │         Batch Formation (Dynamic Batching)              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Batch 1    │  │   Batch 2    │  │   Batch N    │   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │   │
│  └─────────┼─────────────────┼─────────────────┼───────────┘   │
└────────────┼─────────────────┼─────────────────┼───────────────┘
             │                 │                 │
┌────────────▼─────────────────▼─────────────────▼───────────────┐
│                 Thread Pool (Inference)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Worker 1    │  │  Worker 2    │  │  Worker N    │           │
│  │  (Inference) │  │  (Inference) │  │  (Inference) │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Model Executor                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Shared Model & KV Cache (Thread-Safe)          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Model      │  │  KV Cache    │  │  Sampler     │   │   │
│  │  │  (Read-Only) │  │  (R/W Lock) │  │  (Stateless) │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 线程管理

#### 5.2.1 线程池设计

基于Python版本的ThreadPoolExecutor实现，C++版本将使用Intel TBB实现高性能线程池。

##### 线程池接口 (thread_pool.h)
```cpp
#pragma once
#include <functional>
#include <future>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    void shutdown();
    size_t get_queue_size() const;
    size_t get_active_threads() const;
    
private:
    void worker_thread();
    
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_threads_;
};

template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_) {
            throw std::runtime_error("submit on stopped ThreadPool");
        }
        
        tasks_.emplace([task]() { (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}
```

##### 线程池实现 (thread_pool.cpp)
```cpp
#include "thread_pool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_threads_(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { this->worker_thread(); });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        
        active_threads_++;
        task();
        active_threads_--;
    }
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers_.clear();
}

size_t ThreadPool::get_queue_size() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

size_t ThreadPool::get_active_threads() const {
    return active_threads_.load();
}
```

#### 5.2.2 线程配置策略

基于Python版本的线程池配置（CPU核心数的2-4倍），C++版本采用以下策略：

```cpp
struct ThreadPoolConfig {
    size_t inference_threads;      // 推理线程数：CPU核心数 * 2
    size_t io_threads;             // I/O线程数：CPU核心数 / 2
    size_t scheduler_threads;      // 调度线程数：1-2个
    
    static ThreadPoolConfig from_cpu_cores(size_t cpu_cores) {
        ThreadPoolConfig config;
        config.inference_threads = cpu_cores * 2;
        config.io_threads = std::max(size_t(1), cpu_cores / 2);
        config.scheduler_threads = 2;
        return config;
    }
};
```

### 5.3 任务调度

#### 5.3.1 请求队列

基于Python版本的优先队列实现，C++版本将使用线程安全的优先队列。

##### 请求队列接口 (request_queue.h)
```cpp
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "request_state.h"

class RequestQueue {
public:
    RequestQueue();
    
    void push(const RequestState& request);
    bool pop(RequestState& request, int timeout_ms = 0);
    bool try_pop(RequestState& request);
    
    size_t size() const;
    bool empty() const;
    void clear();
    
private:
    struct PriorityComparator {
        bool operator()(const RequestState& a, const RequestState& b) const {
            return a.priority > b.priority;
        }
    };
    
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::priority_queue<
        RequestState,
        std::vector<RequestState>,
        PriorityComparator
    > queue_;
};
```

##### 请求队列实现 (request_queue.cpp)
```cpp
#include "request_queue.h"
#include <chrono>

RequestQueue::RequestQueue() {}

void RequestQueue::push(const RequestState& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(request);
    condition_.notify_one();
}

bool RequestQueue::pop(RequestState& request, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (timeout_ms > 0) {
        condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return !queue_.empty();
        });
    } else {
        condition_.wait(lock, [this] { return !queue_.empty(); });
    }
    
    if (queue_.empty()) {
        return false;
    }
    
    request = queue_.top();
    queue_.pop();
    return true;
}

bool RequestQueue::try_pop(RequestState& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    request = queue_.top();
    queue_.pop();
    return true;
}

size_t RequestQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool RequestQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void RequestQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}
```

#### 5.3.2 优先级计算

基于Python版本的优先级策略，C++版本实现相同的优先级计算逻辑：

```cpp
class PriorityCalculator {
public:
    static double calculate_priority(
        const RequestState& request,
        const std::chrono::steady_clock::time_point& current_time
    ) {
        auto wait_time = std::chrono::duration<double>(
            current_time - request.start_time
        ).count();
        
        size_t prompt_length = request.tokenized_prompt.size();
        
        if (prompt_length <= 50) {
            return -(wait_time + std::min(prompt_length / 10.0, 5.0));
        } else {
            return -(wait_time + std::min(prompt_length / 50.0, 20.0));
        }
    }
};
```

### 5.4 批处理管理

#### 5.4.1 动态批处理

基于Python版本的动态批处理实现，C++版本将实现智能批处理策略。

##### 批处理管理器接口 (batch_manager.h)
```cpp
#pragma once
#include <vector>
#include "request_state.h"

struct BatchConfig {
    size_t max_batch_size;
    size_t max_context_length;
    double context_usage_threshold;
    
    BatchConfig()
        : max_batch_size(8)
        , max_context_length(2048)
        , context_usage_threshold(0.75) {}
};

class BatchManager {
public:
    explicit BatchManager(const BatchConfig& config);
    
    std::vector<std::vector<RequestState>> form_batches(
        std::vector<RequestState>& requests
    );
    
    BatchConfig get_config() const;
    void set_config(const BatchConfig& config);
    
private:
    size_t calculate_batch_size(const std::vector<RequestState>& requests);
    double calculate_avg_request_length(const std::vector<RequestState>& requests);
    
    BatchConfig config_;
};
```

##### 批处理管理器实现 (batch_manager.cpp)
```cpp
#include "batch_manager.h"
#include <algorithm>
#include <numeric>

BatchManager::BatchManager(const BatchConfig& config) : config_(config) {}

std::vector<std::vector<RequestState>> BatchManager::form_batches(
    std::vector<RequestState>& requests
) {
    std::vector<std::vector<RequestState>> batches;
    
    if (requests.empty()) {
        return batches;
    }
    
    size_t running_requests_length = 0;
    for (const auto& req : requests) {
        running_requests_length += req.tokenized_prompt.size();
    }
    
    if (running_requests_length > config_.max_context_length * config_.context_usage_threshold) {
        return batches;
    }
    
    size_t batch_size = calculate_batch_size(requests);
    
    for (size_t i = 0; i < requests.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, requests.size());
        batches.emplace_back(requests.begin() + i, requests.begin() + end);
    }
    
    return batches;
}

size_t BatchManager::calculate_batch_size(const std::vector<RequestState>& requests) {
    if (requests.empty()) {
        return 0;
    }
    
    double avg_length = calculate_avg_request_length(requests);
    
    if (avg_length > 500) {
        return std::min(size_t(2), config_.max_batch_size);
    } else if (avg_length > 200) {
        return std::min(size_t(4), config_.max_batch_size);
    } else {
        return config_.max_batch_size;
    }
}

double BatchManager::calculate_avg_request_length(const std::vector<RequestState>& requests) {
    if (requests.empty()) {
        return 0.0;
    }
    
    double total_length = std::accumulate(
        requests.begin(),
        requests.end(),
        0.0,
        [](double sum, const RequestState& req) {
            return sum + req.tokenized_prompt.size();
        }
    );
    
    return total_length / requests.size();
}

BatchConfig BatchManager::get_config() const {
    return config_;
}

void BatchManager::set_config(const BatchConfig& config) {
    config_ = config;
}
```

### 5.5 同步机制

#### 5.5.1 读写锁

用于保护共享资源（如KV缓存）的读写锁：

```cpp
#include <shared_mutex>

class ThreadSafeKVCache {
public:
    std::vector<float> get_keys(const std::string& request_id) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(request_id);
        if (it != cache_map_.end()) {
            return it->second.keys;
        }
        return {};
    }
    
    void set(const std::string& request_id, 
             const std::vector<float>& keys,
             const std::vector<float>& values) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_map_[request_id] = {keys, values};
    }
    
private:
    struct CacheEntry {
        std::vector<float> keys;
        std::vector<float> values;
    };
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, CacheEntry> cache_map_;
};
```

#### 5.5.2 无锁队列

用于高性能任务队列的无锁队列实现：

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue() : head_(new Node), tail_(head_.load()) {}
    
    void enqueue(T value) {
        Node* new_node = new Node(std::move(value));
        
        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
        prev_tail->data.reset(new T(std::move(value)));
    }
    
    bool dequeue(T& result) {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return false;
        }
        
        result = std::move(*next->data);
        head_.store(next);
        delete head;
        
        return true;
    }
    
private:
    struct Node {
        std::unique_ptr<T> data;
        std::atomic<Node*> next;
        
        Node() : next(nullptr) {}
        Node(T value) : data(new T(std::move(value))), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
};
```

### 5.6 组件关系图

#### 5.6.1 并发组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        cLLM 并发架构                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HTTP Server (自研 epoll/kqueue)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  /generate   │  │  /stream     │  │  /health     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Request Handler                                                 │
│  - 解析请求                                                       │
│  - 创建RequestState                                              │
│  - 提交到Scheduler                                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Scheduler                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RequestQueue (Priority Queue)                            │   │
│  │  - push(request)                                          │   │
│  │  - pop(request)                                           │   │
│  │  - size()                                                 │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │  BatchManager                                             │   │
│  │  - form_batches(requests)                                 │   │
│  │  - calculate_batch_size()                                 │   │
│  │  - calculate_avg_request_length()                          │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │  SchedulerLoop                                            │   │
│  │  - _scheduler_loop()                                      │   │
│  │  - _process_requests()                                    │   │
│  │  - _process_batch()                                       │   │
│  └────────────────────┬─────────────────────────────────────┘   │
└───────────────────────┼───────────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────────┐
│  ThreadPool (Inference)                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Worker 1    │  │  Worker 2    │  │  Worker N    │           │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │           │
│  │  │Task    │  │  │  │Task    │  │  │  │Task    │  │           │
│  │  │Queue   │  │  │  │Queue   │  │  │  │Queue   │  │           │
│  │  └───┬────┘  │  │  └───┬────┘  │  │  └───┬────┘  │           │
│  │      │       │  │      │       │  │      │       │           │
│  │  ┌───▼────┐  │  │  ┌───▼────┐  │  │  ┌───▼────┐  │           │
│  │  │Inference│  │  │  │Inference│  │  │  │Inference│  │           │
│  │  │Task    │  │  │  │Task    │  │  │  │Task    │  │           │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  ModelExecutor                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ThreadSafeKVCache                                         │   │
│  │  - get_keys(request_id) [shared_lock]                     │   │
│  │  - set(request_id, keys, values) [unique_lock]            │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │  Model (Read-Only, Thread-Safe)                          │   │
│  │  - forward(input)                                         │   │
│  │  - generate(input, max_tokens)                            │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │  Sampler (Stateless, Thread-Safe)                        │   │
│  │  - sample(logits, temperature)                            │   │
│  │  - sample_top_k(logits, k)                                │   │
│  │  - sample_top_p(logits, p)                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.6.2 数据流图

```
Request Flow:
1. HTTP Request → Request Handler
2. Request Handler → RequestQueue (push)
3. SchedulerLoop → RequestQueue (pop)
4. SchedulerLoop → BatchManager (form_batches)
5. BatchManager → ThreadPool (submit batch tasks)
6. ThreadPool Workers → ModelExecutor (inference)
7. ModelExecutor → ThreadSafeKVCache (get/set)
8. ModelExecutor → Sampler (sample)
9. ThreadPool Workers → SchedulerLoop (return results)
10. SchedulerLoop → Request Handler (send response)
11. Request Handler → HTTP Response
```

### 5.7 并发性能优化

#### 5.7.1 线程亲和性

```cpp
#include <pthread.h>
#include <unistd.h>

class ThreadAffinity {
public:
    static void set_thread_affinity(std::thread& thread, int core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        
        pthread_setaffinity_np(
            thread.native_handle(),
            sizeof(cpu_set_t),
            &cpuset
        );
    }
    
    static void set_thread_affinity_range(
        std::thread& thread,
        int start_core,
        int end_core
    ) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        for (int i = start_core; i <= end_core; ++i) {
            CPU_SET(i, &cpuset);
        }
        
        pthread_setaffinity_np(
            thread.native_handle(),
            sizeof(cpu_set_t),
            &cpuset
        );
    }
};
```

#### 5.7.2 批处理并行化

```cpp
class ParallelBatchProcessor {
public:
    std::vector<BatchResult> process_batches_parallel(
        const std::vector<Batch>& batches,
        ThreadPool& thread_pool
    ) {
        std::vector<std::future<BatchResult>> futures;
        
        for (const auto& batch : batches) {
            futures.push_back(
                thread_pool.submit([this, batch]() {
                    return process_batch(batch);
                })
            );
        }
        
        std::vector<BatchResult> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    }
    
private:
    BatchResult process_batch(const Batch& batch);
};
```

### 5.8 并发测试策略

#### 5.8.1 并发测试用例

```cpp
class ConcurrencyTest {
public:
    static void test_concurrent_requests(int num_requests, int num_threads);
    static void test_batch_processing(int batch_size, int num_batches);
    static void test_priority_queue(int num_requests);
    static void test_kv_cache_concurrency(int num_readers, int num_writers);
    static void test_thread_pool_performance(int num_tasks, int num_threads);
};
```

## 6. 依赖库选择

### 6.0 技术栈对比分析

基于Python版本的技术栈分析，我们选择以下C++技术栈：

| Python版本 | C++版本 | 说明 |
|-----------|---------|------|
| FastAPI + Uvicorn | 自研 HTTP Server | 基于 epoll/kqueue，支持真流式输出 |
| PyTorch (torch) | LibTorch | PyTorch官方C++ API，完全兼容 |
| NumPy | Eigen3 | 高性能数值计算库，内置SIMD优化 |
| asyncio | Asio | 异步I/O库 |
| ThreadPoolExecutor | Intel TBB | 并行计算和任务调度 |
| transformers (AutoTokenizer) | sentencepiece | 分词库，支持BPE等算法 |
| logging | spdlog | 日志库 |
| unittest | Google Test | 测试框架 |
| json | nlohmann/json | JSON处理库 |
| - | x86 SIMD Intrinsics | 手动SIMD优化（AVX2/AVX-512） |
| - | oneDNN | Intel深度学习优化库（可选） |

### 5.1 HTTP 服务器
**选择**: 自研高性能 HTTP Server
- 基于 epoll (Linux) / kqueue (macOS) 的事件驱动架构
- 支持真流式输出（chunked transfer encoding）
- 每生成一个 token 立即发送，TTFB < 0.1s
- 轻量级，无外部依赖
- 完整的 RESTful API 支持
- 支持 SSE (Server-Sent Events) 格式

**设计特点**:
- 非阻塞 I/O，高并发支持
- 流式回调机制，避免轮询延迟
- Pistache: 现代HTTP库，RESTful支持

### 5.2 深度学习推理引擎
**选择**: LibTorch (PyTorch C++ API)
- PyTorch官方C++接口，与Python版本完全兼容
- 支持模型加载、推理、量化
- 提供torch::jit::compile等优化功能
- 成熟稳定，文档完善
- 可以直接复用Python版本的模型和优化技术

**备选方案**:
- ONNX Runtime: 跨平台推理引擎，支持多种模型格式
- TensorRT: NVIDIA GPU优化推理（GPU场景）

### 5.3 数值计算库
**选择**: Eigen3
- 高性能C++模板库，用于线性代数
- 支持矩阵运算、向量运算
- 头文件库，易于集成
- 与NumPy功能对应

### 5.4 异步框架
**选择**: 原生系统调用 (epoll/kqueue)
- 直接使用操作系统提供的高性能 I/O 多路复用
- Linux: epoll, macOS/BSD: kqueue
- 零依赖，最大性能
- 与 Python asyncio 功能对应

### 5.5 并行计算库
**选择**: Intel TBB (Threading Building Blocks)
- 高性能并行计算库
- 提供任务调度器、并发容器
- 支持细粒度并行
- 与Python ThreadPoolExecutor功能对应

### 5.6 JSON 处理
**选择**: nlohmann/json
- 现代C++ JSON库
- 类似Python字典的API设计
- 高性能，易于使用
- 广泛使用，社区活跃

### 5.7 日志
**选择**: spdlog
- 高性能C++日志库
- 支持异步日志
- 多种sink支持
- 与Python logging功能对应

### 5.8 Tokenizer
**选择**: sentencepiece
- Google开源的分词库
- 支持BPE、Unigram等算法
- 与HuggingFace transformers兼容
- C++原生支持，性能优秀

### 5.9 测试框架
**选择**: Google Test + Google Mock
- 功能完善的测试框架
- 支持单元测试和集成测试
- Mock支持完善
- 与Python unittest功能对应

### 5.10 SIMD 优化库
**选择**: x86 SIMD Intrinsics + Eigen3 SIMD
- **x86 SIMD Intrinsics**: 手动SIMD优化，支持AVX2、AVX-512指令集
- **Eigen3 SIMD**: Eigen3内置SIMD优化，自动向量化支持
- **oneDNN (可选)**: Intel深度学习优化库，提供高度优化的算子实现

**SIMD优化策略**:
- 在关键计算路径中使用SIMD指令
- 利用Eigen3的自动向量化能力
- 针对特定CPU架构手动优化（AVX2/AVX-512）
- 内存对齐确保SIMD指令的最佳性能
- 编译器优化标志：-march=native -O3

## 7. 项目结构

```
cpp/cLLM/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── 设计文档.md
│   ├── API文档.md
│   └── 性能优化指南.md
├── include/
│   ├── server/
│   │   └── http_server.h
│   ├── scheduler/
│   │   └── scheduler.h
│   ├── executor/
│   │   └── model_executor.h
│   ├── tokenizer/
│   │   └── tokenizer.h
│   ├── sampler/
│   │   └── sampler.h
│   ├── cache/
│   │   └── kv_cache.h
│   ├── simd/
│   │   ├── simd_utils.h
│   │   ├── simd_ops.h
│   │   └── avx512_ops.h
│   └── common/
│       ├── types.h
│       ├── config.h
│       └── utils.h
├── src/
│   ├── server/
│   │   └── http_server.cpp
│   ├── scheduler/
│   │   └── scheduler.cpp
│   ├── executor/
│   │   └── model_executor.cpp
│   ├── tokenizer/
│   │   └── tokenizer.cpp
│   ├── sampler/
│   │   └── sampler.cpp
│   ├── cache/
│   │   └── kv_cache.cpp
│   ├── simd/
│   │   ├── simd_utils.cpp
│   │   ├── simd_ops.cpp
│   │   └── avx512_ops.cpp
│   └── main.cpp
├── tests/
│   ├── unit/
│   │   ├── test_tokenizer.cpp
│   │   ├── test_sampler.cpp
│   │   ├── test_kv_cache.cpp
│   │   └── test_scheduler.cpp
│   └── integration/
│       └── test_api.cpp
├── examples/
│   ├── simple_client.cpp
│   └── benchmark_client.cpp
├── tools/
│   ├── benchmark.cpp
│   └── profiler.cpp
└── third_party/
    ├── llama.cpp/          # LLM 推理引擎
    ├── tokenizers-cpp/     # Hugging Face tokenizers
    ├── spdlog/
    └── nlohmann_json/
```

## 8. 构建系统

### 8.1 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.16)
project(cLLM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖项
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
# llama.cpp 和 tokenizers-cpp 作为子模块构建

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 可执行文件
add_executable(cllm_server ${SOURCES})

# 链接库
target_link_libraries(cllm_server
    PRIVATE
    llama              # llama.cpp 推理引擎
    tokenizers_cpp     # Hugging Face tokenizers
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# 设置C++标准
target_compile_features(cllm_server PUBLIC cxx_std_20)
```

## 9. 开发路线图
（省略，按照实际情况）

## 10. 性能指标

### 10.1 目标指标
- **推理速度**: >20 tokens/s
- **并发请求**: 支持 100+ 并发
- **内存占用**: <2GB（不含模型）
- **响应延迟**: P99 <100ms
- **吞吐量**: >1000 req/s

### 10.2 基准测试
- 单请求生成性能
- 批处理性能
- 并发请求性能
- 内存使用效率
- CPU 使用率

## 11. 兼容性

### 11.1 API 兼容性
保持与 Python 版本相同的 API 接口：
- 请求格式
- 响应格式
- 错误码
- 流式响应格式

### 11.2 模型兼容性
支持与 Python 版本相同的模型格式：
- HuggingFace 模型
- 自定义模型格式
- 量化模型

## 12. 风险和挑战

### 12.1 技术风险
- C++ 实现复杂度高
- 性能优化难度大
- 内存管理复杂
- 并发控制复杂

### 12.2 缓解措施
- 分阶段开发和测试
- 充分的单元测试和集成测试
- 性能分析和优化工具
- 代码审查和最佳实践

## 13. 参考文档

- Python 版本代码: `/Users/dannypan/PycharmProjects/xllm/python/`
- 设计文档: `/Users/dannypan/PycharmProjects/xllm/docs/design_document.md`
- 并行实现文档: `/Users/dannypan/PycharmProjects/xllm/docs/xllm的并行实现.md`

## 14. 总结

本设计文档详细规划了 cLLM C++ 重构的技术方案、架构设计和实施路线。通过采用现代 C++ 技术和性能优化策略，预期可以显著提升系统性能，同时保持与 Python 版本的功能兼容性。项目将分阶段实施，确保每个阶段的质量和稳定性。