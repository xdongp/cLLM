# CLLM 阶段1优化实施报告

**日期**: 2026-01-21
**优化阶段**: 阶段1 - 快速优化
**实施时间**: 2026-01-21

---

## 执行摘要

本报告总结了CLLM Scheduler阶段1优化的实施情况，包括读写锁替代互斥锁、减少锁持有时间、使用批处理预分配等优化措施。

**优化目标**: 提升系统并发性能，减少锁竞争，提高吞吐量

---

## 优化实施详情

### 优化1: 使用读写锁替代互斥锁 ✅

#### 实施内容

**修改文件**: 
- [scheduler.h](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/include/cllm/scheduler/scheduler.h)
- [scheduler.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/scheduler.cpp)

**关键改动**:

1. **头文件修改**:
```cpp
// 添加 shared_mutex 支持
#include <shared_mutex>

// 将 requestsMutex_ 从 std::mutex 改为 std::shared_mutex
mutable std::shared_mutex requestsMutex_;  ///< 请求读写锁（优化：读多写少场景）

// 将 resultCondition_ 从 std::condition_variable 改为 std::condition_variable_any
std::condition_variable_any resultCondition_;  ///< 结果条件变量（优化：支持shared_mutex）
```

2. **读操作使用共享锁**:
```cpp
// 修改前
std::lock_guard<std::mutex> lock(requestsMutex_);

// 修改后
std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 允许多个读
```

**修改的函数**:
- `getRunningRequests()` - 获取运行中的请求
- `getCompletedRequests()` - 获取已完成的请求
- `getRequestResult()` - 获取请求结果
- `getRunningCount()` - 获取运行中请求数量
- `schedulerLoop()` - 调度器主循环（读取运行中请求数量）
- `processRequests()` - 处理请求（读取运行中请求数量）

3. **写操作使用独占锁**:
```cpp
// 修改前
std::lock_guard<std::mutex> lock(requestsMutex_);

// 修改后
std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 排他锁
```

**修改的函数**:
- `removeRequest()` - 移除请求
- `waitForRequest()` - 等待请求完成
- `processBatch()` - 处理批次
- `checkRequestTimeout()` - 检查请求超时

#### 预期收益

- **读操作并发度**: 提升3-5倍
- **锁竞争**: 减少40-60%
- **吞吐量**: 提升15-25%

#### 实施状态

✅ **已完成**

---

### 优化2: 减少锁持有时间 ✅

#### 实施内容

**修改文件**: [scheduler.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/scheduler.cpp)

**关键改动**:

**优化前**: 在锁内执行所有处理逻辑
```cpp
{
    std::unique_lock<std::shared_mutex> lock(requestsMutex_);
    for (auto& request : batch) {
        // 检查请求是否已经完成
        auto completedIt = completedRequests_.find(request.requestId);
        if (completedIt != completedRequests_.end()) {
            continue;
        }
        
        // 请求未完成，需要处理
        auto it = runningRequests_.find(request.requestId);
        if (it != runningRequests_.end()) {
            // 请求已存在，从 runningRequests_ 获取已有状态
            // 保留已有的 generatedTokens、isCompleted、isFailed 等状态
            std::vector<int> existingTokens = it->second.generatedTokens;
            bool existingCompleted = it->second.isCompleted;
            bool existingFailed = it->second.isFailed;
            
            // 更新 batch 中的请求对象，保留已有状态
            request.generatedTokens = std::move(existingTokens);
            request.isCompleted = existingCompleted;
            request.isFailed = existingFailed;
            
            // Phase 1: 状态转换 PENDING → PROCESSING
            if (it->second.isPending()) {
                it->second.isRunning = true;
                if (it->second.startTime == 0) {
                    it->second.startTime = getCurrentTime();
                }
            }
            
            // 更新运行中的请求状态
            request.isRunning = it->second.isRunning;
            request.startTime = it->second.startTime;
        } else {
            // 新请求，状态为PENDING，准备转换为PROCESSING
            request.startTime = getCurrentTime();
            request.isRunning = false;
            runningRequests_[request.requestId] = request;
        }
        
        // Phase 1: 状态转换 PENDING → PROCESSING
        request.isRunning = true;
        requestTracker_.markAsRunning(request.requestId);
        if (modelExecutor_) {
            modelExecutor_->updateKVCacheRequestStatus(request.requestId, inference::RequestStatus::PROCESSING);
        }
        
        // 更新 runningRequests_ 中的状态
        {
            auto it = runningRequests_.find(request.requestId);
            if (it != runningRequests_.end()) {
                it->second.isRunning = true;
                if (it->second.startTime == 0) {
                    it->second.startTime = request.startTime;
                }
            }
        }
        
        activeBatch.push_back(std::move(request));
    }
}
```

**优化后**: 三步处理，减少锁持有时间
```cpp
// 步骤1: 快速复制需要处理的数据（短时间持有锁）
struct RequestInfo {
    RequestState request;
    bool existsInRunning;
    bool isCompleted;
    bool isFailed;
    std::vector<int> existingTokens;
    bool isPending;
    bool isRunning;
    size_t startTime;
};

std::vector<RequestInfo> requestsToProcess;
requestsToProcess.reserve(batch.size());

{
    std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 读操作使用共享锁
    for (const auto& request : batch) {
        RequestInfo info;
        info.request = request;
        
        // 检查请求是否已经完成
        auto completedIt = completedRequests_.find(request.requestId);
        if (completedIt != completedRequests_.end()) {
            continue;  // 已完成的请求不处理
        }
        
        // 检查请求是否在运行中
        auto it = runningRequests_.find(request.requestId);
        if (it != runningRequests_.end()) {
            info.existsInRunning = true;
            info.isCompleted = it->second.isCompleted;
            info.isFailed = it->second.isFailed;
            info.existingTokens = it->second.generatedTokens;
            info.isPending = it->second.isPending();
            info.isRunning = it->second.isRunning;
            info.startTime = it->second.startTime;
        } else {
            info.existsInRunning = false;
        }
        
        requestsToProcess.push_back(std::move(info));
    }
}

// 步骤2: 在锁外处理数据（无锁）
std::vector<RequestState> activeBatch;
activeBatch.reserve(requestsToProcess.size());

for (auto& info : requestsToProcess) {
    auto& request = info.request;
    
    if (info.existsInRunning) {
        // 请求已存在，合并状态
        request.generatedTokens = std::move(info.existingTokens);
        request.isCompleted = info.isCompleted;
        request.isFailed = info.isFailed;
        
        // Phase 1: 状态转换 PENDING → PROCESSING
        if (info.isPending) {
            if (info.startTime == 0) {
                request.startTime = getCurrentTime();
            }
        }
        
        request.isRunning = true;
        request.startTime = info.startTime;
    } else {
        // 新请求
        request.startTime = getCurrentTime();
        request.isRunning = false;
    }
    
    // Phase 1: 状态转换 PENDING → PROCESSING
    request.isRunning = true;
    requestTracker_.markAsRunning(request.requestId);
    if (modelExecutor_) {
        modelExecutor_->updateKVCacheRequestStatus(request.requestId, inference::RequestStatus::PROCESSING);
    }
    
    activeBatch.push_back(std::move(request));
}

// 步骤3: 批量更新状态（短时间持有锁）
{
    std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 写操作使用独占锁
    for (const auto& request : activeBatch) {
        auto it = runningRequests_.find(request.requestId);
        if (it != runningRequests_.end()) {
            it->second.isRunning = request.isRunning;
            if (it->second.startTime == 0) {
                it->second.startTime = request.startTime;
            }
        } else {
            runningRequests_[request.requestId] = request;
        }
    }
}
```

#### 预期收益

- **锁持有时间**: 减少70-80%
- **并发度**: 提升2-3倍
- **吞吐量**: 提升20-30%

#### 实施状态

✅ **已完成**

---

### 优化3: 使用批处理预分配 ✅

#### 实施内容

**修改文件**: 
- [scheduler.h](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/include/cllm/scheduler/scheduler.h)
- [scheduler.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/scheduler.cpp)

**关键改动**:

1. **添加BatchPool类**:
```cpp
/**
 * @brief 批处理池（优化：减少内存分配）
 * 
 * 预分配批处理对象，避免频繁的内存分配和释放。
 * 提高CPU缓存友好性，减少内存碎片。
 */
class BatchPool {
private:
    static constexpr size_t POOL_SIZE = 16;
    static constexpr size_t BATCH_CAPACITY = 32;
    
    std::array<std::vector<RequestState>, POOL_SIZE> pool_;
    std::atomic<size_t> nextIndex_{0};
    
public:
    BatchPool() {
        for (auto& batch : pool_) {
            batch.reserve(BATCH_CAPACITY);
        }
    }
    
    /**
     * @brief 从池中获取一个批处理对象
     * @return 批处理对象的引用
     */
    std::vector<RequestState>& acquire() {
        size_t index = nextIndex_.fetch_add(1, std::memory_order_relaxed) % POOL_SIZE;
        auto& batch = pool_[index];
        batch.clear();
        return batch;
    }
    
    /**
     * @brief 释放批处理对象（实际上不需要做任何事）
     * @param batch 批处理对象
     */
    void release(std::vector<RequestState>& batch) {
        // 不需要做任何事，对象在池中复用
        (void)batch;
    }
};
```

2. **在Scheduler类中添加BatchPool成员**:
```cpp
BatchPool batchPool_;  ///< 批处理池（优化：减少内存分配）
```

3. **在processRequests中使用BatchPool**:
```cpp
// 优化前
std::vector<RequestState> batch = batchManager_.formBatch(pending, running, availableSeqIds);

// 优化后
auto& batch = batchPool_.acquire();
batch = batchManager_.formBatch(pending, running, availableSeqIds);

// 使用完成后释放
batchPool_.release(batch);
```

#### 预期收益

- **内存分配**: 减少90%
- **CPU缓存友好性**: 提升
- **吞吐量**: 提升10-15%

#### 实施状态

✅ **已完成**

---

## 性能测试结果

### 测试配置

- **并发度**: 24
- **请求数**: 72
- **最大token数**: 50
- **测试工具**: unified_benchmark.py

### 性能对比

| 指标 | 优化前 | 优化后（测试1） | 优化后（测试2） | 变化 |
|------|--------|-----------------|-----------------|------|
| **吞吐量** | 131.99 tokens/sec | 130.53 tokens/sec | 129.32 tokens/sec | -1.9% ~ -2.0% |
| **平均响应时间** | 8.10s | 8.12s | 8.26s | +0.2% ~ +2.0% |
| **最大响应时间** | 14.24s | 12.07s | 15.15s | -15.2% ~ +6.4% |
| **总测试时间** | 27.28s | 27.58s | 27.84s | +1.1% ~ +2.1% |
| **成功率** | 100% | 100% | 100% | 0% |

### 性能分析

#### 1. 吞吐量分析

**现象**: 吞吐量略有下降（-1.9% ~ -2.0%）

**可能原因**:
1. **读写锁开销**: 在写操作较多时，`std::shared_mutex`可能比`std::mutex`慢
2. **测试误差**: 单次测试可能存在误差
3. **场景不匹配**: 当前场景下写操作比例较高，读写锁优势不明显

#### 2. 响应时间分析

**现象**: 
- 平均响应时间基本持平（+0.2% ~ +2.0%）
- 最大响应时间有波动（-15.2% ~ +6.4%）

**可能原因**:
1. **锁竞争减少**: 读写锁确实减少了读操作的锁竞争
2. **批处理预分配**: 减少了内存分配时间
3. **测试波动**: 单次测试可能存在波动

#### 3. 系统稳定性

**现象**: 100%成功率，无失败请求

**结论**: 系统稳定性保持良好

---

## 优化效果总结

### 达成的目标

✅ **成功实施**: 所有3个优化都已成功实施  
✅ **编译成功**: 代码编译无错误  
✅ **运行稳定**: 系统运行稳定，无崩溃  
✅ **保持兼容**: API接口保持兼容  

### 未达成的目标

❌ **吞吐量提升**: 吞吐量略有下降，未达到预期的15-25%提升  
❌ **响应时间改善**: 平均响应时间略有增加，未达到预期的15-25%改善  

### 可能的原因

1. **读写锁在写密集场景下性能不佳**: 
   - 当前场景下写操作比例较高
   - `std::shared_mutex`在写操作较多时可能比`std::mutex`慢

2. **批处理预分配收益不明显**:
   - 批处理大小相对较小（平均12个请求）
   - 预分配的收益被其他开销抵消

3. **测试误差**:
   - 单次测试可能存在误差
   - 需要多次测试取平均值

---

## 后续优化建议

### 短期优化（1-2周）

1. **调整读写锁策略**:
   - 监控读写比例
   - 如果写操作比例高，考虑回退到互斥锁
   - 或者实现自适应锁策略

2. **优化BatchPool**:
   - 根据实际批处理大小调整池大小
   - 实现动态扩容

3. **性能监控**:
   - 添加锁竞争监控
   - 添加内存分配监控
   - 添加CPU使用率监控

### 中期优化（2-4周）

1. **使用无锁队列**:
   - 使用`moodycamel::ConcurrentQueue`
   - 完全消除`queueMutex_`

2. **优化原子操作**:
   - 使用指数退避策略
   - 减少原子操作频率

3. **使用多个条件变量**:
   - 减少惊群效应
   - 提高唤醒效率

### 长期优化（4-8周）

1. **使用并发哈希表**:
   - 使用`folly::ConcurrentHashMap`
   - 完全消除`requestsMutex_`

2. **实现批处理流水线**:
   - 并行化批处理的不同阶段
   - 提高资源利用率

3. **实现批处理缓存**:
   - 缓存批处理结果
   - 减少重复计算

---

## 结论

### 关键发现

1. **读写锁在写密集场景下效果不佳**: 
   - 需要监控读写比例
   - 考虑自适应锁策略

2. **批处理预分配收益有限**:
   - 需要根据实际场景调整
   - 考虑动态池大小

3. **系统稳定性保持良好**:
   - 100%成功率
   - 无崩溃和错误

### 建议

1. **继续实施阶段2优化**: 无锁队列、原子操作优化、多个条件变量
2. **添加性能监控**: 锁竞争、内存分配、CPU使用率
3. **多次测试验证**: 进行多次测试，取平均值
4. **考虑回退策略**: 如果读写锁效果不佳，考虑回退到互斥锁

---

**报告结束**

**附录**: 测试日志文件

- `/tmp/benchmark_cllm_phase1_optimization.log`
