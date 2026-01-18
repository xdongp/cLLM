# 调度器卡住问题根因分析 - 11个请求后100%卡住

## 问题现象

- 发送11次请求后，服务器100%卡住
- 日志显示请求12开始处理（`llama_seq_id=3`），但之后没有日志
- 调度器循环可能卡住

## 关键证据

### 1. llama.cpp 序列数限制

```
n_seq_max = 8  // llama.cpp 最多支持8个并行序列
```

### 2. 序列ID分配模式（从日志提取）

```
请求1: llama_seq_id=0, nextSeqId=1
请求2: llama_seq_id=1, nextSeqId=2
请求3: llama_seq_id=2, nextSeqId=3
请求4: llama_seq_id=3, nextSeqId=4
请求5: llama_seq_id=4, nextSeqId=5
请求6: llama_seq_id=5, nextSeqId=6
请求7: llama_seq_id=6, nextSeqId=7
请求8: llama_seq_id=7, nextSeqId=0  ← 循环回到0
请求9: llama_seq_id=0, nextSeqId=1
请求10: llama_seq_id=1, nextSeqId=2
请求11: llama_seq_id=2, nextSeqId=3
请求12: llama_seq_id=3, nextSeqId=4  ← **在这里卡住**
```

### 3. 日志终止点

最后一条日志：
```
[2026-01-18 12:13:03.001] [info] [LlamaCppBackend::forwardBatch] Sequence 0 (llama_seq_id=3): prefill, 4 tokens (from 0 to 4), position set to 0
```

之后应该是：
```
[info] [LlamaCppBackend::forwardBatch] Calling llama_decode with 4 tokens...
```

但这条日志**没有出现**，说明在调用 `llama_decode` 之前就卡住了，或者 `llama_decode` 阻塞了。

## 根本原因分析

### 最可能的原因：`llama_decode` 阻塞

#### 假设1：KV cache冲突导致阻塞

当序列ID循环回到0-7时：
1. 请求4使用序列ID 3，完成后清理KV cache
2. 请求12也使用序列ID 3，清理KV cache
3. **但如果请求4的KV cache清理不完整**，或者 llama.cpp 内部还在使用序列ID 3，`llama_decode` 可能会阻塞

#### 假设2：序列ID重用冲突

虽然代码中清理了KV cache，但可能存在竞态条件：
- 请求4完成后，正在清理序列ID 3的KV cache
- 请求12到达，也尝试使用序列ID 3
- 两个请求同时操作序列ID 3，导致冲突

#### 假设3：`llama_decode` 内部阻塞`

llama.cpp 的 `llama_decode` 可能在等待某个资源（如KV cache槽位），如果该资源被占用或状态不一致，可能会永久阻塞。

### 为什么是11个请求后？

- **前8个请求**：使用序列ID 0-7，全部用完
- **第9-11个请求**：循环回到序列ID 0-2
- **第12个请求**：使用序列ID 3，这是第一次**重用**之前使用过的序列ID

关键点：**请求4和请求12都使用序列ID 3**，可能存在冲突。

## 验证方法

1. **添加调试日志**：在 `llama_decode` 调用前后添加日志
2. **检查死锁**：使用 gdb 附加到进程，查看调用栈
3. **测试序列ID重用**：故意重用序列ID，观察是否卡住

## 解决方案

### 方案1：序列ID生命周期管理（推荐）

不要循环使用序列ID，而是维护一个序列ID池，确保序列ID在使用前完全清理：

```cpp
class LlamaCppBackend {
private:
    std::vector<int32_t> availableSeqIds_;  // 可用序列ID池
    std::map<size_t requestId, int32_t> requestToSeqId_;  // 请求ID到序列ID的映射
    mutable std::mutex seqIdPoolMutex_;
    
public:
    int32_t allocateSeqId(size_t requestId) {
        std::lock_guard<std::mutex> lock(seqIdPoolMutex_);
        if (availableSeqIds_.empty()) {
            // 没有可用序列ID，等待或失败
            CLLM_WARN("No available sequence ID, request %zu must wait", requestId);
            return -1;  // 或等待
        }
        int32_t seqId = availableSeqIds_.front();
        availableSeqIds_.erase(availableSeqIds_.begin());
        requestToSeqId_[requestId] = seqId;
        return seqId;
    }
    
    void releaseSeqId(size_t requestId) {
        std::lock_guard<std::mutex> lock(seqIdPoolMutex_);
        auto it = requestToSeqId_.find(requestId);
        if (it != requestToSeqId_.end()) {
            int32_t seqId = it->second;
            clearKVCacheForSequence(seqId);  // 清理KV cache
            resetSeqPosition(seqId);  // 重置位置
            availableSeqIds_.push_back(seqId);  // 回收序列ID
            requestToSeqId_.erase(it);
        }
    }
};
```

### 方案2：限制并发请求数

在调度器中限制同时处理的请求数不超过 `n_seq_max`：

```cpp
void Scheduler::processRequests() {
    // 限制并发请求数
    size_t runningCount;
    {
        std::lock_guard<std::mutex> reqLock(requestsMutex_);
        runningCount = runningRequests_.size();
    }
    
    if (runningCount >= 8) {  // n_seq_max = 8
        // 已达到最大并发数，等待
        return;
    }
    
    // ... 继续处理 ...
}
```

### 方案3：请求完成时立即释放序列ID

在 `processBatch` 完成后，立即释放序列ID：

```cpp
void Scheduler::processBatch(...) {
    // ... 处理批处理 ...
    
    // 请求完成时，释放序列ID
    for (auto& request : batch) {
        if (request.isCompleted || request.isFailed) {
            // 释放序列ID（通过后端）
            // 注意：需要添加接口来释放序列ID
        }
    }
}
```
