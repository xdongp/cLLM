# KV Cache 后端抽象设计方案

**设计日期**: 2026-01-22  
**问题背景**: `cllm::KVCache` 对于 llama.cpp 后端不必要，需要灵活处理

---

## 一、问题分析

### 1.1 当前架构问题

```
Scheduler
    │
    ├─ 创建 KVCache (cllm::KVCache)        ← 对 llama.cpp 后端冗余
    │
    └─ 创建 BatchProcessor
           │
           └─ 接收 KVCache* cache 参数    ← 从未实际使用
```

### 1.2 后端差异

| 后端 | KV Cache 管理方式 | 需要 cllm::KVCache |
|------|------------------|-------------------|
| llama.cpp | llama.cpp 内部管理 + inference::KVCacheManager 统计 | ❌ 不需要 |
| Kylin | 可能需要外部 KV Cache | ✅ 需要 |
| LibTorch | 可能需要外部 KV Cache | ✅ 需要 |

---

## 二、推荐方案：混合方案（条件编译 + 运行时配置）

### 2.1 设计原则

1. **编译期**: 使用 `#ifdef CLLM_USE_LLAMA_CPP` 排除不需要的代码
2. **运行时**: 使用 `backendType` 配置按需创建 KVCache
3. **兼容性**: 保留对非 llama.cpp 后端的支持

### 2.2 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         混合方案架构                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scheduler                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  init(backendType) {                                                │   │
│  │      #ifndef CLLM_LLAMA_CPP_ONLY  // 不是纯 llama.cpp 构建           │   │
│  │          if (backendType != "llama_cpp") {                          │   │
│  │              kvCache_ = new KVCache(...);  // 按需创建               │   │
│  │          }                                                          │   │
│  │      #endif                                                         │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  BatchProcessor                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  // KVCache 参数可选                                                 │   │
│  │  BatchProcessor(scheduler, executor, batchManager, kvCache = nullptr)│   │
│  │                                                                      │   │
│  │  processIteration() {                                               │   │
│  │      if (cache_ != nullptr) {                                       │   │
│  │          // 使用 KVCache（非 llama.cpp 后端）                         │   │
│  │      }                                                              │   │
│  │      // llama.cpp 后端: cache_ == nullptr，跳过                      │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、实现方案

### 3.1 方案A: 最小改动（推荐）

保持接口不变，运行时按需创建 KVCache。

#### 修改 1: Scheduler 构造函数

```cpp
// src/scheduler/scheduler.cpp

Scheduler::Scheduler(ModelExecutor* executor) 
    : modelExecutor_(executor)
    , ownsModelExecutor_(false)
{
    // ... 其他初始化 ...
    
    // 🔧 修改: 根据后端类型决定是否创建 KVCache
    std::string backendType = modelExecutor_->getBackendType();
    
    if (needsExternalKVCache(backendType)) {
        kvCache_ = new KVCache(
            Config::instance().serverKvCacheMaxSize(),
            Config::instance().serverKvCacheMaxMemoryMb()
        );
        CLLM_INFO("[Scheduler] Created KVCache for backend: %s", backendType.c_str());
    } else {
        kvCache_ = nullptr;
        CLLM_INFO("[Scheduler] KVCache not needed for backend: %s (managed internally)", 
                  backendType.c_str());
    }
}

// 辅助函数：判断是否需要外部 KVCache
bool Scheduler::needsExternalKVCache(const std::string& backendType) const {
    // llama.cpp 后端内部管理 KV Cache，不需要外部 KVCache
    if (backendType == "llama_cpp" || backendType == "llama.cpp" || backendType == "LlamaCpp") {
        return false;
    }
    // Kylin 和 LibTorch 可能需要外部 KVCache
    return true;
}
```

#### 修改 2: BatchProcessor 处理 nullptr

```cpp
// src/scheduler/batch_processor.cpp

void SchedulerBatchProcessor::processIteration(...) {
    // ... 现有逻辑 ...
    
    // 🔧 修改: 检查 KVCache 是否可用
    if (cache_ != nullptr) {
        // 非 llama.cpp 后端: 使用外部 KVCache
        // ... KVCache 相关操作 ...
    }
    // llama.cpp 后端: cache_ == nullptr，KV Cache 由内部管理
}
```

#### 修改 3: 析构函数安全删除

```cpp
// src/scheduler/scheduler.cpp

Scheduler::~Scheduler() {
    stop();
    
    // 🔧 修改: 安全删除
    if (kvCache_ != nullptr) {
        delete kvCache_;
        kvCache_ = nullptr;
    }
    
    // ...
}
```

### 3.2 方案B: 条件编译（最大性能）

仅在非 llama.cpp 构建时包含 KVCache 代码。

#### CMakeLists.txt 添加选项

```cmake
option(CLLM_LLAMA_CPP_ONLY "Build for llama.cpp backend only" OFF)

if(CLLM_LLAMA_CPP_ONLY)
    add_definitions(-DCLLM_LLAMA_CPP_ONLY)
endif()
```

#### 头文件条件编译

```cpp
// include/cllm/scheduler/scheduler.h

class Scheduler {
private:
#ifndef CLLM_LLAMA_CPP_ONLY
    KVCache* kvCache_ = nullptr;  // 仅非 llama.cpp 构建包含
#endif
};
```

---

## 四、方案对比

| 维度 | 方案A (运行时) | 方案B (条件编译) |
|------|---------------|-----------------|
| 代码改动量 | 小 | 中 |
| 运行时开销 | 极低 (nullptr 检查) | 零 |
| 灵活性 | 高 | 低 |
| 构建复杂度 | 不变 | 增加选项 |
| 推荐场景 | 通用场景 | 纯 llama.cpp 部署 |

---

## 五、推荐实施步骤

### 第一阶段: 方案A（运行时配置）

1. 修改 `Scheduler` 按 `backendType` 决定是否创建 `KVCache`
2. 修改 `BatchProcessor` 安全处理 `nullptr`
3. 测试验证

### 第二阶段: 可选 - 方案B（条件编译）

如果需要极致性能优化，可以添加 `CLLM_LLAMA_CPP_ONLY` 编译选项。

---

## 六、配置示例

```yaml
# config/config.yaml

# 后端类型: llama_cpp / kylin / libtorch
backend:
  type: llama_cpp  # 使用 llama.cpp 后端，不创建外部 KVCache
  
  llama_cpp:
    n_batch: 512
    n_threads: 0
    n_gpu_layers: 0
    n_seq_max: 8
```

---

**设计完成时间**: 2026-01-22
