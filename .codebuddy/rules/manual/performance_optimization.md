# ⚡ 性能优化专项规则

> **触发条件**: 用户提到"优化"、"加速"、"性能"、"慢"时使用本规则

---

## 🎯 优化目标

- 提升推理吞吐量 (tokens/s)
- 降低延迟 (首token时间)
- 减少内存占用
- 提高并发处理能力

---

## 🔍 性能分析流程

### 1. Profiling (必须先执行)

```bash
# CPU Profiling
perf record -g ./bin/cllm_server
perf report

# 或使用gprof
g++ -pg -o cllm_server ...
./cllm_server
gprof cllm_server gmon.out > analysis.txt

# 内存Profiling
valgrind --tool=massif ./bin/cllm_server
ms_print massif.out.*
```

### 2. 热点识别

```markdown
关注以下热点:
1. ✅ Tokenizer encode/decode
2. ✅ Model forward pass
3. ✅ KVCache操作
4. ✅ 数据拷贝 (CPU↔GPU)
5. ✅ 线程同步开销
```

---

## 📋 优化检查清单

### CPU优化

- [ ] **避免不必要的拷贝**
  ```cpp
  // ❌ 值传递
  void process(std::vector<int> data);
  
  // ✅ 引用传递
  void process(const std::vector<int>& data);
  
  // ✅ 移动语义
  void setData(std::vector<int>&& data) {
      data_ = std::move(data);
  }
  ```

- [ ] **预分配内存**
  ```cpp
  std::vector<int> tokens;
  tokens.reserve(estimatedSize);  // ✅ 避免多次realloc
  
  for (int i = 0; i < n; ++i) {
      tokens.push_back(i);  // 不会触发realloc
  }
  ```

- [ ] **使用并行算法**
  ```cpp
  #include <BS_thread_pool.hpp>
  
  BS::thread_pool pool(std::thread::hardware_concurrency());
  
  // 并行处理batch
  pool.parallelize_loop(0, batchSize, 
      [&](int start, int end) {
          for (int i = start; i < end; ++i) {
              processBatch(batches[i]);
          }
      }
  );
  pool.wait();
  ```

- [ ] **减少虚函数调用**
  ```cpp
  // ❌ 频繁虚函数调用
  for (auto& item : items) {
      item->virtualMethod();  // 每次查虚表
  }
  
  // ✅ 批量处理
  batchProcess(items);  // 一次虚函数调用
  ```

- [ ] **内联小函数**
  ```cpp
  // ✅ 使用inline避免函数调用开销
  inline int add(int a, int b) {
      return a + b;
  }
  ```

### 内存优化

- [ ] **对象池复用**
  ```cpp
  class ObjectPool {
  public:
      torch::Tensor acquire() {
          if (!pool_.empty()) {
              auto tensor = pool_.back();
              pool_.pop_back();
              return tensor;
          }
          return torch::empty({1024});
      }
      
      void release(torch::Tensor tensor) {
          pool_.push_back(tensor);
      }
      
  private:
      std::vector<torch::Tensor> pool_;
  };
  ```

- [ ] **减少临时对象**
  ```cpp
  // ❌ 创建临时对象
  std::string result = getPrefix() + getSuffix();
  
  // ✅ 直接构造
  std::string result;
  result.reserve(estimatedSize);
  result.append(getPrefix());
  result.append(getSuffix());
  ```

- [ ] **智能指针性能**
  ```cpp
  // ✅ 优先使用unique_ptr (无引用计数开销)
  std::unique_ptr<Tokenizer> tokenizer;
  
  // ⚠️  shared_ptr有原子操作开销
  std::shared_ptr<Tokenizer> tokenizer;
  
  // ✅ 在需要共享时再用shared_ptr
  ```

### I/O优化

- [ ] **减少磁盘I/O**
  ```cpp
  // ✅ 缓存tokenizer
  static std::unordered_map<std::string, Tokenizer> tokenizerCache;
  
  // ✅ mmap大文件
  int fd = open(path.c_str(), O_RDONLY);
  void* data = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  ```

- [ ] **异步I/O**
  ```cpp
  #include <asio.hpp>
  
  asio::io_context io;
  asio::post(io, []() {
      // 异步加载模型
      loadModel();
  });
  ```

### 并发优化

- [ ] **减少锁竞争**
  ```cpp
  // ❌ 粗粒度锁
  std::lock_guard<std::mutex> lock(globalMutex_);
  // ... 长时间操作 ...
  
  // ✅ 细粒度锁
  {
      std::lock_guard<std::mutex> lock(cacheMutex_);
      auto item = cache_.get(key);
  }
  // 释放锁后再处理
  process(item);
  ```

- [ ] **使用无锁数据结构**
  ```cpp
  // ✅ 原子操作
  std::atomic<size_t> counter{0};
  counter.fetch_add(1, std::memory_order_relaxed);
  
  // ✅ 线程局部存储
  thread_local std::vector<int> localCache;
  ```

- [ ] **批处理减少同步**
  ```cpp
  // ❌ 每个请求都同步
  for (auto& req : requests) {
      mutex_.lock();
      process(req);
      mutex_.unlock();
  }
  
  // ✅ 批量处理
  mutex_.lock();
  for (auto& req : requests) {
      process(req);
  }
  mutex_.unlock();
  ```

---

## 🚀 cLLM特定优化

### Tokenizer优化

```cpp
// 1. 缓存encode结果
class TokenCache {
    std::unordered_map<std::string, std::vector<int>> cache_;
    size_t maxSize_ = 10000;
    
public:
    std::optional<std::vector<int>> get(const std::string& text) {
        auto it = cache_.find(text);
        return it != cache_.end() ? std::make_optional(it->second) : std::nullopt;
    }
    
    void put(const std::string& text, std::vector<int> ids) {
        if (cache_.size() < maxSize_) {
            cache_[text] = std::move(ids);
        }
    }
};

// 2. 批量encode
std::vector<std::vector<int>> batchEncode(
    const std::vector<std::string>& texts
) {
    BS::thread_pool pool;
    std::vector<std::future<std::vector<int>>> futures;
    
    for (const auto& text : texts) {
        futures.push_back(pool.submit_task([&, text]() {
            return tokenizer_->encode(text);
        }));
    }
    
    std::vector<std::vector<int>> results;
    for (auto& f : futures) {
        results.push_back(f.get());
    }
    return results;
}
```

### KVCache优化

```cpp
// 1. 预分配cache
class KVCache {
    std::vector<torch::Tensor> preallocated_;
    
public:
    KVCache(size_t capacity) {
        preallocated_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            preallocated_.push_back(torch::empty({...}));
        }
    }
    
    torch::Tensor acquire() {
        if (!preallocated_.empty()) {
            auto tensor = preallocated_.back();
            preallocated_.pop_back();
            return tensor;
        }
        return torch::empty({...});
    }
};

// 2. 分块管理
class ChunkedKVCache {
    static constexpr size_t CHUNK_SIZE = 64;
    std::vector<std::unique_ptr<CacheChunk>> chunks_;
    
    torch::Tensor get(size_t index) {
        size_t chunkIdx = index / CHUNK_SIZE;
        size_t offset = index % CHUNK_SIZE;
        return chunks_[chunkIdx]->get(offset);
    }
};
```

### Model Executor优化

```cpp
// 1. Batch推理
class BatchedExecutor {
    std::vector<Request> buffer_;
    size_t batchSize_ = 32;
    
public:
    void submit(Request req) {
        buffer_.push_back(std::move(req));
        if (buffer_.size() >= batchSize_) {
            processBatch(buffer_);
            buffer_.clear();
        }
    }
    
private:
    void processBatch(const std::vector<Request>& batch) {
        // 打包输入
        auto inputIds = packInputs(batch);
        
        // 批量推理
        auto outputs = model_->forward(inputIds);
        
        // 分发结果
        distributeOutputs(batch, outputs);
    }
};

// 2. 流水线并行
class PipelineExecutor {
    BS::thread_pool prefetchPool_;
    BS::thread_pool inferencePool_;
    
public:
    void execute(Request req) {
        // Stage 1: 预处理 (CPU)
        prefetchPool_.submit_task([&, req]() {
            auto tokens = tokenizer_->encode(req.text);
            
            // Stage 2: 推理 (GPU)
            inferencePool_.submit_task([&, tokens]() {
                auto output = model_->forward(tokens);
                
                // Stage 3: 后处理 (CPU)
                prefetchPool_.submit_task([&, output]() {
                    auto text = tokenizer_->decode(output);
                    req.callback(text);
                });
            });
        });
    }
};
```

---

## 📊 性能监控

### 1. 添加计时器

```cpp
#include "cllm/common/timer.h"

void TokenizerManager::encode(const std::string& text) {
    Timer timer("encode");
    
    auto result = tokenizer_->encode(text);
    
    float elapsed = timer.elapsed();
    CLLM_DEBUG("Encode took %.2f ms", elapsed);
    
    // 更新统计
    stats_.addEncodeTime(elapsed);
    
    return result;
}
```

### 2. 统计信息收集

```cpp
class PerformanceStats {
public:
    void recordLatency(float ms) {
        latencies_.push_back(ms);
        totalLatency_ += ms;
        ++count_;
    }
    
    float getAvgLatency() const {
        return count_ > 0 ? totalLatency_ / count_ : 0.0f;
    }
    
    float getP50Latency() const {
        if (latencies_.empty()) return 0.0f;
        auto sorted = latencies_;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2];
    }
    
    float getP99Latency() const {
        if (latencies_.empty()) return 0.0f;
        auto sorted = latencies_;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() * 99 / 100];
    }
    
private:
    std::vector<float> latencies_;
    float totalLatency_ = 0.0f;
    size_t count_ = 0;
};
```

### 3. 实时监控

```cpp
class PerformanceMonitor {
    std::atomic<size_t> requestsProcessed_{0};
    std::atomic<size_t> tokensGenerated_{0};
    std::chrono::steady_clock::time_point startTime_;
    
public:
    PerformanceMonitor() : startTime_(std::chrono::steady_clock::now()) {}
    
    void recordRequest() {
        requestsProcessed_.fetch_add(1);
    }
    
    void recordTokens(size_t count) {
        tokensGenerated_.fetch_add(count);
    }
    
    void printStats() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(now - startTime_).count();
        
        float reqPerSec = requestsProcessed_.load() / elapsed;
        float tokensPerSec = tokensGenerated_.load() / elapsed;
        
        CLLM_INFO("Performance:");
        CLLM_INFO("  Requests/s: %.2f", reqPerSec);
        CLLM_INFO("  Tokens/s: %.2f", tokensPerSec);
    }
};
```

---

## 🎯 优化目标基准

### 当前基线 (需测量)

```markdown
- Tokenizer encode: ? ms/request
- Tokenizer decode: ? ms/request
- Model forward: ? ms/token
- End-to-end latency: ? ms
- Throughput: ? tokens/s
- Memory usage: ? MB
```

### 优化目标

```markdown
- Tokenizer: 提升 3-5x
- Model inference: 提升 2x (通过batch)
- 并发能力: 支持 100+ 并发请求
- 内存: 减少 20-30%
```

---

## 🔍 Profiling工具使用

### perf (Linux)

```bash
# 记录性能数据
perf record -g -F 99 ./bin/cllm_server

# 生成报告
perf report

# 火焰图
git clone https://github.com/brendangregg/FlameGraph
perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > flame.svg
```

### Instruments (macOS)

```bash
# Time Profiler
instruments -t "Time Profiler" -D trace.trace ./bin/cllm_server

# Allocations
instruments -t "Allocations" -D trace.trace ./bin/cllm_server
```

### Valgrind

```bash
# 内存泄漏检测
valgrind --leak-check=full ./bin/cllm_server

# 内存profiling
valgrind --tool=massif ./bin/cllm_server
ms_print massif.out.12345
```

---

## 📚 参考资料

- **C++性能优化**: [Optimized C++](https://www.oreilly.com/library/view/optimized-c/9781491922057/)
- **并行编程**: [C++ Concurrency in Action](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)
- **性能分析**: [Systems Performance](http://www.brendangregg.com/systems-performance-2nd-edition-book.html)

---

**最后更新**: 2026-01-11
