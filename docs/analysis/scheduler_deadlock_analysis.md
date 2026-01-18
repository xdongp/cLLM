# 调度器卡住问题分析

## 问题描述

在发送11次请求后，服务器卡住，不再处理后续请求。

## 代码流程分析

### 1. 调度器主循环 (`schedulerLoop`)

```cpp
void Scheduler::schedulerLoop() {
    while (running_) {
        processRequests();  // 步骤1：处理请求
        
        std::unique_lock<std::mutex> lock(queueMutex_);  // 步骤2：获取锁
        size_t queueSize = requestQueue_.getQueueSize();  // 步骤3：检查队列大小
        size_t runningCount = runningRequests_.size();   // 步骤4：检查运行中的请求
        
        if (queueSize == 0 && runningCount == 0) {
            // 等待通知
            queueCondition_.wait_for(...);
        } else {
            lock.unlock();
            std::this_thread::sleep_for(...);  // 步骤5：睡眠
        }
    }
}
```

### 2. 请求处理 (`processRequests`)

```cpp
void Scheduler::processRequests() {
    // 获取队列大小
    size_t queueSize;
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        queueSize = requestQueue_.getQueueSize();
    }
    
    // 获取待处理请求
    std::vector<RequestState> pending = requestQueue_.getPendingRequests();  // 这里会重建队列！
    
    // 形成批处理
    std::vector<RequestState> batch = batchManager_.formBatch(pending, running);
    
    if (batch.empty() && queueSize > 0) {
        return;  // 队列中有请求但无法形成批处理，返回
    }
    
    if (batch.empty()) {
        return;
    }
    
    processBatch(batch);
}
```

### 3. 批处理完成后的通知

```cpp
void Scheduler::processBatch(...) {
    // ... 处理批处理 ...
    
    // 检查是否还有待处理的请求
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        if (requestQueue_.getQueueSize() > 0) {
            queueCondition_.notify_one();  // 通知调度器继续处理
        }
    }
}
```

## 潜在问题

### 问题1: `getPendingRequests()` 重建队列的时序问题

`getPendingRequests()` 会：
1. 清空队列
2. 复制请求到临时向量
3. 重新填充队列

在 `schedulerLoop()` 中：
- 步骤1：`processRequests()` 调用 `getPendingRequests()`，清空并重建队列
- 步骤2-3：在 `processRequests()` 返回后，才获取锁并检查 `queueSize`

**时序问题**：
```
T1: processRequests() 调用 getPendingRequests()，队列被清空
T2: processRequests() 返回
T3: schedulerLoop() 获取 queueMutex_，检查 queueSize
```

如果在 T1 和 T3 之间，队列中加入了新请求，但在 T3 时检查的 `queueSize` 可能不准确。

### 问题2：条件变量通知的竞态

`processBatch()` 完成后，如果队列中有请求，会调用 `queueCondition_.notify_one()`。

但是在 `schedulerLoop()` 中：
- 如果在 `notify_one()` 之后，但 `processRequests()` 返回之前，检查 `queueSize`
- 或者在 `notify_one()` 之后，但锁还没有被释放，导致通知丢失

### 问题3: `formBatch` 返回空 batch 的死循环

如果 `formBatch` 因为资源限制（如上下文长度）返回空 batch，但队列中还有请求：

```cpp
// processRequests() 中
if (batch.empty() && queueSize > 0) {
    return;  // 返回，但不处理请求
}

// schedulerLoop() 中
if (queueSize == 0 && runningCount == 0) {
    wait_for(...);  // 等待通知
} else {
    sleep_for(...);  // 睡眠后继续循环
}
```

如果 `formBatch` 一直返回空 batch，`processRequests()` 会立即返回，而 `schedulerLoop()` 会继续循环，但不会处理请求，导致**无限循环但无实际处理**。

### 问题4: 锁的持有时间过长

在 `schedulerLoop()` 中，获取 `queueMutex_` 后，检查 `queueSize` 和 `runningCount`，然后才决定是否等待。这个过程可能太长，导致新请求无法及时加入队列。

## 根本原因分析 - 11个请求后卡住的死锁

### 关键发现：`getRunningRequests()` 未过滤已完成的请求

在 `processRequests()` 中：

```cpp
std::vector<RequestState> running = getRunningRequests();  // 获取所有运行中的请求
std::vector<RequestState> batch = batchManager_.formBatch(pending, running);
```

而 `getRunningRequests()` 的实现：
```cpp
std::vector<RequestState> Scheduler::getRunningRequests() const {
    std::lock_guard<std::mutex> lock(requestsMutex_);
    std::vector<RequestState> requests;
    for (const auto& pair : runningRequests_) {
        requests.push_back(pair.second);  // 直接添加，不检查 isCompleted！
    }
    return requests;
}
```

### 死锁场景分析

**为什么是11个请求后卡住？**

1. **前10个请求处理完成后**：
   - 这10个请求从 `runningRequests_` 移到 `completedRequests_`
   - 但可能存在竞态条件：在 `processBatch` 完成后，但在清理 `runningRequests_` 之前

2. **第11个请求到达时**：
   - `processRequests()` 调用 `getRunningRequests()` 
   - 如果 `runningRequests_` 中还有前10个已完成的请求（虽然不应该有），它们会被包含
   - `formBatch` 计算 `runningLength` 时，会包括这些已完成的请求的 token 长度
   - 如果前10个请求的总长度 > `maxContextLength_ * contextUsageThreshold_`，`formBatch` 返回空 batch

3. **死锁形成**：
   ```cpp
   if (batch.empty() && queueSize > 0) {
       return;  // 返回，但不处理请求
   }
   ```
   - `processRequests()` 返回，第11个请求不被处理
   - `schedulerLoop()` 继续循环，但 `formBatch` 一直返回空 batch
   - 形成**无限循环但不处理请求**

### 为什么是11而不是其他数字？

- **前10个请求完成后**：`runningLength` 计算时可能包含这些已完成的请求
- **第11个请求**：`formBatch` 因为 `runningLength` 过大返回空 batch
- **后续请求**：都被阻塞，无法处理

### 根本问题

1. `getRunningRequests()` **不检查 `isCompleted`**，可能返回已完成的请求
2. `formBatch` 使用这些请求计算 `runningLength`，导致高估
3. 如果 `runningLength > threshold`，返回空 batch，导致请求无法处理

具体场景：
1. 第11个请求完成，`runningRequests_` 变为空
2. 队列中有第12-20个请求等待处理
3. `processRequests()` 调用 `getPendingRequests()`，获取了所有待处理请求
4. `formBatch` 可能因为某种原因（资源限制）返回空 batch
5. `processRequests()` 返回，但不处理请求
6. `schedulerLoop()` 检查 `queueSize` 和 `runningCount`
7. 如果 `runningCount == 0`，但 `queueSize > 0`，会进入 `else` 分支，睡眠后继续循环
8. 但如果 `formBatch` 一直返回空 batch，就会形成**无限循环但不处理请求**

## 解决方案

1. **改进 `processRequests()` 的检查逻辑**：在调用 `formBatch` 之前，先检查是否可以形成批处理
2. **修复 `getPendingRequests()` 的队列重建问题**：避免重建队列，使用只读方式获取请求
3. **改进条件变量通知逻辑**：确保通知在正确的时机发送
4. **添加超时机制**：如果 `formBatch` 一直返回空，应该等待一段时间再重试，而不是立即返回
