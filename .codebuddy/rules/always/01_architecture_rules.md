# 🏗️ cLLM 架构设计约束

> **优先级**: HIGH | 保证系统架构完整性和模块解耦

---

## 🎯 架构原则

### 1. 模块化设计

```
cLLM 采用分层架构:

┌─────────────────────────────────────────┐
│         HTTP Server Layer               │  ← 对外API
│  (自定义HTTP Server - 基于Asio)          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      Request Processing Layer           │
│  (Validator, Handler, Response)         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     TokenizerManager Layer              │  ← 核心业务
│  (Tokenizer, Generator)                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     Model Executor Layer                │
│  (Inference Engine, Sampler)            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    Backend Layer (LibTorch/Kylin)       │  ← 推理后端
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Infrastructure Layer                  │
│ (Logger, ThreadPool, KVCache, Memory)   │
└─────────────────────────────────────────┘
```

### 2. 依赖规则

**允许的依赖方向** (上层依赖下层):

```
HTTP → TokenizerManager → ModelExecutor → Backend → Infrastructure
  ↓                           ↓
Request                    KVCache
                              ↓
                        Infrastructure
```

**禁止的依赖**:

- ❌ Infrastructure 依赖上层模块
- ❌ Backend 依赖 TokenizerManager
- ❌ 任何循环依赖

---

## 📦 核心模块说明

### 1. Tokenizer 模块

**位置**: `include/cllm/tokenizer/`, `src/tokenizer/`, `src/CTokenizer/`

**职责**:
- 文本编码/解码
- 支持多种分词器格式 (HF, SentencePiece, Native)
- Token生成流式输出

**接口定义**:

```cpp
namespace cllm {

// 基础接口 (所有Tokenizer必须实现)
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    
    virtual bool load(const std::string& modelPath) = 0;
    virtual std::vector<int> encode(const std::string& text, bool addSpecialTokens) = 0;
    virtual std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) = 0;
    
    virtual int getVocabSize() const = 0;
    virtual int getBosId() const = 0;
    virtual int getEosId() const = 0;
    virtual int getPadId() const = 0;
    virtual int getUnkId() const = 0;
};

// 实现类
class HFTokenizer : public ITokenizer { /* ... */ };
class NativeTokenizer : public ITokenizer { /* ... */ };
class UnifiedTokenizer : public ITokenizer { /* ... */ };

} // namespace cllm
```

**依赖规则**:
- ✅ 可依赖: `common/logger`, `common/utils`
- ❌ 禁止依赖: `model/executor`, `http/`
- ⚠️  条件依赖: `tokenizers-cpp` (条件编译)

---

### 2. ModelExecutor 模块

**位置**: `include/cllm/model/`, `src/model/`

**职责**:
- 模型推理
- Batch处理
- 采样策略
- KVCache管理

**接口定义**:

```cpp
namespace cllm {

class ModelExecutor {
public:
    // 单次推理
    torch::Tensor forward(
        const torch::Tensor& inputIds,
        const torch::Tensor& attentionMask,
        std::optional<torch::Tensor> pastKeyValues = std::nullopt
    );
    
    // 生成 (完整)
    std::vector<int> generate(
        const std::vector<int>& inputIds,
        int maxTokens,
        float temperature = 1.0f
    );
    
    // 流式生成
    int generateNext(
        const torch::Tensor& inputIds,
        const torch::Tensor& attentionMask,
        float temperature = 1.0f
    );
};

} // namespace cllm
```

**依赖规则**:
- ✅ 可依赖: `kv_cache/cache`, `sampler/sampler`, `common/*`
- ❌ 禁止依赖: `tokenizer/manager`, `http/`
- ✅ 可被依赖: `tokenizer/manager`

---

### 3. KVCache 模块

**位置**: `include/cllm/kv_cache/`, `src/kv_cache/`

**职责**:
- 缓存 Key-Value states
- 内存管理
- 缓存淘汰策略

**接口定义**:

```cpp
namespace cllm {

class KVCache {
public:
    void insert(const std::string& key, const torch::Tensor& kv);
    std::optional<torch::Tensor> get(const std::string& key);
    void evict(const std::string& key);
    void clear();
    
    size_t size() const;
    size_t memoryUsage() const;
};

} // namespace cllm
```

**依赖规则**:
- ✅ 可依赖: `common/logger`, `common/memory_utils`
- ❌ 禁止依赖: 任何业务模块
- ✅ 可被依赖: `model/executor`

---

### 4. HTTP Server 模块

**位置**: `include/cllm/http/`, `src/http/`

**职责**:
- HTTP请求处理
- OpenAI API兼容
- 请求验证
- 响应构建

**接口定义**:

```cpp
namespace cllm {

class HttpServer {
public:
    void start(const std::string& host, int port);
    void stop();
    
    void registerEndpoint(const std::string& path, EndpointHandler handler);
};

// Endpoint handlers
void handleGenerate(const HttpRequest& req, HttpResponse& resp);
void handleEncode(const HttpRequest& req, HttpResponse& resp);
void handleHealth(const HttpRequest& req, HttpResponse& resp);

} // namespace cllm
```

**依赖规则**:
- ✅ 可依赖: `tokenizer/manager`, `model/executor`, `common/*`
- ❌ 禁止依赖: 底层Backend

---

### 5. Scheduler 模块

**位置**: `include/cllm/scheduler/`, `src/scheduler/`

**职责**:
- 请求调度
- 批处理优化
- 优先级管理

**接口定义**:

```cpp
namespace cllm {

class Scheduler {
public:
    void submit(std::shared_ptr<Request> request);
    std::vector<std::shared_ptr<Request>> schedule();
    
    void setPriority(const std::string& requestId, int priority);
    void cancel(const std::string& requestId);
};

} // namespace cllm
```

---

## 🔧 模块集成规范

### 1. TokenizerManager 集成

```cpp
// ✅ 正确的初始化顺序
auto modelExecutor = std::make_unique<ModelExecutor>(config);
auto kvCache = std::make_shared<KVCache>(cacheConfig);

modelExecutor->setKVCache(kvCache.get());

auto tokenizerManager = std::make_unique<TokenizerManager>(
    modelPath,
    modelExecutor.get(),
    TokenizerImpl::AUTO  // 自动检测
);
```

### 2. HTTP Server 集成

```cpp
// ✅ 正确的服务启动流程
HttpServer server;

// 设置依赖
server.setTokenizerManager(tokenizerManager.get());
server.setModelExecutor(modelExecutor.get());

// 注册端点
server.registerEndpoint("/v1/chat/completions", handleGenerate);
server.registerEndpoint("/v1/embeddings", handleEncode);
server.registerEndpoint("/health", handleHealth);

// 启动服务
server.start("0.0.0.0", 8080);
```

---

## 📝 模块修改规范

### 修改前检查清单

在修改任何模块前,必须检查:

1. **依赖影响分析**
   ```bash
   # 搜索所有依赖该模块的代码
   search_content("include.*<cllm/模块名/", "include,src")
   ```

2. **接口兼容性**
   - 是否改变了公共接口?
   - 是否需要更新依赖模块?
   - 是否需要更新单元测试?

3. **头文件修改同步**
   ```
   修改 include/cllm/tokenizer/hf_tokenizer.h
   ↓ 必须同步检查
   src/tokenizer/hf_tokenizer.cpp
   ```

4. **CMakeLists.txt 更新**
   - 新增源文件需添加到 `target_sources`
   - 新增依赖需添加到 `target_link_libraries`

---

## 🏭 设计模式应用

### 1. Factory 模式 (Tokenizer创建)

```cpp
// ✅ 使用Factory统一创建
class TokenizerFactory {
public:
    static std::unique_ptr<ITokenizer> create(
        const std::string& modelPath,
        TokenizerImpl impl = TokenizerImpl::AUTO
    );
};

// 使用
auto tokenizer = TokenizerFactory::create(modelPath);
```

### 2. Strategy 模式 (采样策略)

```cpp
class SamplerStrategy {
public:
    virtual int sample(const torch::Tensor& logits) = 0;
};

class GreedySampler : public SamplerStrategy { /* ... */ };
class TopKSampler : public SamplerStrategy { /* ... */ };
class TopPSampler : public SamplerStrategy { /* ... */ };
```

### 3. Observer 模式 (流式生成)

```cpp
class GenerationObserver {
public:
    virtual void onTokenGenerated(int tokenId, const std::string& text) = 0;
    virtual void onComplete() = 0;
};

// StreamGenerator 通知观察者
for (auto observer : observers_) {
    observer->onTokenGenerated(tokenId, text);
}
```

### 4. Singleton 模式 (Logger)

```cpp
// ✅ 使用局部静态变量实现线程安全单例
class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }
    
private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};
```

---

## 🔒 线程安全规范

### 1. 共享资源保护

```cpp
class KVCache {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, torch::Tensor> cache_;
    
public:
    void insert(const std::string& key, const torch::Tensor& kv) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = kv;
    }
    
    std::optional<torch::Tensor> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
};
```

### 2. 线程池使用

```cpp
#include <BS_thread_pool.hpp>

// ✅ 推荐: 使用BS::thread_pool
BS::thread_pool pool(std::thread::hardware_concurrency());

// 提交任务
auto future = pool.submit_task([](int x) {
    return x * x;
}, 42);

int result = future.get();
```

---

## 📊 性能监控埋点

### 1. 关键路径计时

```cpp
#include "cllm/common/timer.h"

void processRequest() {
    Timer timer("processRequest");
    
    // 业务逻辑
    
    CLLM_INFO("Request processed in %.2f ms", timer.elapsed());
}
```

### 2. 统计信息收集

```cpp
class TokenizerStats {
public:
    void incrementEncodeCount() { ++encodeCount_; }
    void addEncodeTime(float time) { totalEncodeTime_ += time; }
    
    float getAvgEncodeTime() const {
        return totalEncodeTime_ / encodeCount_;
    }
    
private:
    std::atomic<size_t> encodeCount_{0};
    std::atomic<float> totalEncodeTime_{0.0f};
};
```

---

## 🚨 架构变更审批

以下变更需特别谨慎:

1. **修改核心接口** (ITokenizer, ModelExecutor)
   - 影响范围: 所有实现类
   - 需要: 完整的迁移计划

2. **添加新的模块依赖**
   - 影响范围: 编译系统
   - 需要: 更新CMakeLists.txt, 文档

3. **修改线程模型**
   - 影响范围: 整体性能
   - 需要: 性能测试验证

4. **修改数据流向**
   - 影响范围: 架构完整性
   - 需要: 架构图更新

---

## 📚 相关设计文档

- **整体架构**: `docs/cLLM详细设计.md`
- **Tokenizer设计**: `docs/modules/分词器设计.md`
- **调度器设计**: `docs/modules/调度器模块设计.md`
- **组件交互**: `docs/组件交互设计.md`

---

**最后更新**: 2026-01-11  
**维护者**: cLLM Architecture Team
