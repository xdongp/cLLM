# 调度器卡住问题完整分析 - 11个请求后100%卡住

## 问题描述

在发送11次请求后，服务器100%卡住，不再处理后续请求。日志显示：
- 请求1-11正常处理完成
- 请求12开始处理，但之后没有日志输出
- 调度器循环可能卡住

## 关键发现

### 1. llama.cpp 序列数限制

从日志看到：
```
n_seq_max = 8
```

llama.cpp 最多支持8个并行序列。序列ID从0开始，循环使用0-7。

### 2. 序列ID分配模式

从日志看序列ID的分配：
- 请求1: `llama_seq_id=0, nextSeqId=1`
- 请求2: `llama_seq_id=1, nextSeqId=2`
- ...
- 请求8: `llama_seq_id=7, nextSeqId=0`
- 请求9: `llama_seq_id=0, nextSeqId=1` (循环回到0)
- 请求10: `llama_seq_id=1, nextSeqId=2`
- 请求11: `llama_seq_id=2, nextSeqId=3`
- 请求12: `llama_seq_id=3, nextSeqId=4` ← **在这里卡住**

### 3. 关键代码逻辑

在 `LlamaCppBackend::forwardBatch` 中：

```cpp
// 序列ID分配（循环使用）
uint32_t contextNSeqMax = contextParams_ ? contextParams_->n_seq_max : 8;
int32_t newSeqId;
{
    std::lock_guard<std::mutex> lock(seqPositionsMutex_);
    newSeqId = nextSeqId_;
    nextSeqId_ = (nextSeqId_ + 1) % static_cast<int32_t>(contextNSeqMax);
}
llamaSeqId = static_cast<llama_seq_id>(newSeqId);
clearKVCacheForSequence(newSeqId);  // 清理KV cache
```

### 4. 死锁可能的原因

#### 假设1：KV cache清理阻塞

当序列ID循环回到0-7时，如果某个序列的KV cache还在使用中（前一个请求还没完成），`clearKVCacheForSequence` 可能会阻塞或失败。

#### 假设2：`llama_decode` 阻塞

如果 `llama_decode` 在内部等待某个资源（如KV cache槽位），而该资源被其他序列占用，可能会导致死锁。

#### 假设3：调度器条件变量问题

从日志看，请求12开始处理后，调度器循环可能等待某个条件变量，但条件永远不会满足。

### 5. 为什么是11个请求后？

可能的原因：
1. **序列ID耗尽**：虽然序列ID循环使用，但在处理第12个请求时，所有8个序列ID可能都在使用中（虽然前11个请求已完成，但它们的KV cache可能还没完全清理）
2. **KV cache槽位冲突**：llama.cpp 的KV cache可能因为序列ID重用而出现冲突
3. **死锁形成**：第12个请求尝试使用序列ID 3，但这个ID可能还被某个地方引用

## 根本原因推测

### 最可能的原因：序列ID和KV cache管理的竞态条件

当 `nextSeqId_` 循环回到0-7时：

1. 前8个请求使用序列ID 0-7，完成后清理KV cache
2. 但清理操作可能和序列ID分配存在竞态条件
3. 第9个请求使用序列ID 0时，序列0的KV cache可能还没完全清理
4. 到第12个请求时，某些序列ID的KV cache状态不一致，导致死锁

### 验证方法

1. 添加日志，记录每次序列ID分配和KV cache清理的时间戳
2. 检查 `llama_decode` 是否在某个序列ID上阻塞
3. 检查调度器循环的状态（是否在等待条件变量）

## 解决方案建议

### 方案1：序列ID生命周期管理

不要简单地循环使用序列ID，而是维护一个序列ID池：

```cpp
// 维护可用序列ID池
std::vector<int32_t> availableSeqIds;  // 初始化为 [0, 1, 2, ..., 7]
std::map<size_t requestId, int32_t seqId> requestToSeqId;  // 请求ID到序列ID的映射

// 分配序列ID
int32_t allocateSeqId(size_t requestId) {
    if (availableSeqIds.empty()) {
        // 没有可用序列ID，等待或失败
        return -1;
    }
    int32_t seqId = availableSeqIds.front();
    availableSeqIds.erase(availableSeqIds.begin());
    requestToSeqId[requestId] = seqId;
    return seqId;
}

// 释放序列ID
void releaseSeqId(size_t requestId) {
    auto it = requestToSeqId.find(requestId);
    if (it != requestToSeqId.end()) {
        int32_t seqId = it->second;
        clearKVCacheForSequence(seqId);  // 清理KV cache
        availableSeqIds.push_back(seqId);  // 回收序列ID
        requestToSeqId.erase(it);
    }
}
```

### 方案2：限制并发请求数

在调度器中限制同时处理的请求数不超过 `n_seq_max`（8）：

```cpp
// 在 processRequests 中
if (runningRequests_.size() >= n_seq_max) {
    // 等待，直到有请求完成
    return;
}
```

### 方案3：请求完成时立即释放序列ID

在 `processBatch` 完成后，立即释放序列ID：

```cpp
// 在 processBatch 完成后
for (auto& request : batch) {
    if (request.isCompleted || request.isFailed) {
        // 释放序列ID
        llamaBackend->releaseSeqId(request.requestId);
    }
}
```

## 下一步行动

1. **添加调试日志**：在序列ID分配、KV cache清理、llama_decode调用处添加详细日志
2. **检查死锁**：使用gdb或lldb附加到进程，查看调用栈
3. **验证假设**：测试序列ID生命周期管理方案
