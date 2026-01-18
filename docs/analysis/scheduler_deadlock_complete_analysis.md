# 调度器卡住问题完整分析 - 11个请求后100%卡住

## 问题现象

从日志文件 `/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/build/a.log` 分析：

1. **请求1-11正常处理完成**
2. **请求12开始处理，但卡住**：
   - 最后一条日志：`[LlamaCppBackend::forwardBatch] Sequence 0 (llama_seq_id=3): prefill, 4 tokens (from 0 to 4), position set to 0`
   - 之后没有 "Calling llama_decode" 的日志
   - 没有 "Batch processing completed" 的日志

## 根本原因分析

### 关键发现

#### 1. llama.cpp 序列数限制

```
n_seq_max = 8  // llama.cpp 最多支持8个并行序列
```

#### 2. 序列ID分配模式（从日志提取）

```
请求1: llama_seq_id=0, nextSeqId=1
请求2: llama_seq_id=1, nextSeqId=2
请求3: llama_seq_id=2, nextSeqId=3
请求4: llama_seq_id=3, nextSeqId=4  ← 第一次使用序列ID 3
...
请求8: llama_seq_id=7, nextSeqId=0  ← 循环回到0
请求9: llama_seq_id=0, nextSeqId=1  ← 重用序列ID 0
请求10: llama_seq_id=1, nextSeqId=2  ← 重用序列ID 1
请求11: llama_seq_id=2, nextSeqId=3  ← 重用序列ID 2
请求12: llama_seq_id=3, nextSeqId=4  ← **第二次使用序列ID 3，在这里卡住**
```

#### 3. 日志分析

从日志看，请求12的最后一条日志是：
```
[2026-01-18 12:13:03.001] [info] [LlamaCppBackend::forwardBatch] Sequence 0 (llama_seq_id=3): prefill, 4 tokens (from 0 to 4), position set to 0
```

这对应代码第692行。之后应该还有第697行的日志：
```cpp
CLLM_INFO("[LlamaCppBackend::forwardBatch] Calling llama_decode with %d tokens...", ...);
```

但这条日志**没有出现**！

### 问题定位

#### 可能原因1：`llama_decode` 调用前卡住

代码流程：
1. 第692行：输出序列信息日志（**已输出**）
2. 第693行：循环结束 `}`
3. 第695行：`batch.n_tokens = static_cast<int32_t>(actualTokenCount);`
4. 第697行：输出 "Calling llama_decode" 日志（**未输出**）
5. 第701行：调用 `llama_decode(ctx_, batch)`

**如果第697行的日志没输出，说明卡在第693-697行之间。**

#### 可能原因2：`llama_decode` 阻塞

如果日志被缓冲，第697行的日志可能还没写入。但更可能的是：
- `llama_decode` 被调用，但**阻塞了**，没有返回
- 导致后续日志都无法输出

#### 可能原因3：序列ID重用冲突

**关键观察**：请求4和请求12都使用序列ID 3！

- **请求4**：使用序列ID 3，完成后清理KV cache
- **请求12**：使用序列ID 3，清理KV cache，然后...卡住

**可能的问题**：
1. 请求4的KV cache清理不完整
2. llama.cpp 内部对序列ID 3的引用还没释放
3. 请求12尝试使用序列ID 3时，llama.cpp检测到冲突，阻塞或死锁

### 为什么是11个请求后？

- **前8个请求**：使用序列ID 0-7，全部用完
- **第9-11个请求**：循环回到序列ID 0-2（第一次重用）
- **第12个请求**：使用序列ID 3（**第二次重用**）

**假设**：序列ID的第一次重用（请求9-11）可能触发了某些内部状态变化，导致第二次重用（请求12）时出现问题。

### 更深层的问题

从代码看，序列ID的分配是简单的循环使用：
```cpp
nextSeqId_ = (nextSeqId_ + 1) % 8;
```

但这**不保证序列ID在使用前已经完全清理**。可能存在竞态条件：
1. 请求4完成，开始清理序列ID 3的KV cache
2. 请求12到达，分配序列ID 3
3. 请求12开始清理序列ID 3的KV cache（但可能还没完全清理）
4. 请求12调用 `llama_decode` 时，llama.cpp检测到序列ID 3的状态不一致，阻塞

## 解决方案

### 方案1：序列ID生命周期管理（推荐）

不要循环使用序列ID，而是维护一个序列ID池，确保序列ID在使用前完全清理：

```cpp
class LlamaCppBackend {
private:
    std::vector<int32_t> availableSeqIds_;  // 可用序列ID池 [0, 1, ..., 7]
    std::map<size_t requestId, int32_t> requestToSeqId_;  // 请求ID到序列ID的映射
    mutable std::mutex seqIdPoolMutex_;
    
public:
    int32_t allocateSeqId(size_t requestId) {
        std::lock_guard<std::mutex> lock(seqIdPoolMutex_);
        if (availableSeqIds_.empty()) {
            // 没有可用序列ID，必须等待
            CLLM_WARN("No available sequence ID, request %zu must wait", requestId);
            return -1;  // 表示需要等待
        }
        int32_t seqId = availableSeqIds_.front();
        availableSeqIds_.erase(availableSeqIds_.begin());
        requestToSeqId_[requestId] = seqId;
        CLLM_DEBUG("Allocated seq_id=%d for request %zu", seqId, requestId);
        return seqId;
    }
    
    void releaseSeqId(size_t requestId) {
        std::lock_guard<std::mutex> lock(seqIdPoolMutex_);
        auto it = requestToSeqId_.find(requestId);
        if (it != requestToSeqId_.end()) {
            int32_t seqId = it->second;
            // 清理KV cache和状态
            clearKVCacheForSequence(seqId);
            resetSeqPosition(seqId);
            // 回收序列ID
            availableSeqIds_.push_back(seqId);
            requestToSeqId_.erase(it);
            CLLM_DEBUG("Released seq_id=%d for request %zu", seqId, requestId);
        }
    }
};
```

### 方案2：限制并发请求数

在调度器中限制同时处理的请求数不超过 `n_seq_max`：

```cpp
void Scheduler::processRequests() {
    // 限制并发请求数，避免序列ID冲突
    size_t runningCount;
    {
        std::lock_guard<std::mutex> reqLock(requestsMutex_);
        runningCount = runningRequests_.size();
    }
    
    // n_seq_max = 8，限制同时处理的请求数
    if (runningCount >= 8) {
        CLLM_DEBUG("Max concurrent requests reached (%zu), waiting...", runningCount);
        return;  // 等待，直到有请求完成
    }
    
    // ... 继续处理 ...
}
```

### 方案3：请求完成时立即释放序列ID

在 `processBatch` 完成后，立即释放序列ID：

```cpp
// 在 scheduler.cpp 的 processBatch 中
for (auto& request : batch) {
    if (request.isCompleted || request.isFailed) {
        // 释放序列ID
        // 注意：需要在 LlamaCppBackend 中添加 releaseSeqId 接口
        llamaBackend->releaseSeqId(request.requestId);
    }
}
```

## 验证方法

1. **添加调试日志**：
   - 在序列ID分配和释放处添加日志
   - 在 `llama_decode` 调用前后添加时间戳

2. **使用 gdb 调试**：
   ```bash
   gdb -p <pid>
   (gdb) thread apply all bt
   ```
   查看所有线程的调用栈，定位死锁位置

3. **测试序列ID重用**：
   - 修改代码，让序列ID不循环使用
   - 观察是否还会卡住
