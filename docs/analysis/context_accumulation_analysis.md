# 上下文叠加问题分析

## 问题假设

用户提出：性能退化可能不是内存碎片化的问题，而是**上下文叠加**的问题。

**假设**：
- 每次新请求时，虽然我们调用了 `llama_memory_seq_rm` 清理 KV cache
- 但 llama.cpp 内部可能仍然保留了位置信息
- 当我们设置 `batch.pos = 0` 时，llama.cpp 可能认为这是位置回退或位置不连续
- 导致上下文叠加，性能越来越慢

## llama.cpp 的位置连续性要求

根据文档和代码分析：

### 关键要求

对于同一个 `seq_id`，llama.cpp 要求：
- **位置必须连续**：新批次的位置必须是 `last_pos + 1`
- **不能有间隙**：不能跳过位置
- **不能回退**：位置不能小于上次的位置

### 错误示例

如果违反连续性要求，llama.cpp 会报错：
```
sequence positions are not continuous
or
tokens of sequence s in the input batch have inconsistent sequence positions:
  last position in KV cache = X
  starting position in batch = Y
  require Y = X + 1
```

## 当前实现分析

### 当前代码逻辑

```cpp
// 新请求时
if (isNewRequest) {
    clearKVCacheForSequence(seqIdKey);  // 调用 llama_memory_seq_rm
    resetSeqPosition(seqIdKey);         // 重置我们的位置记录为 0
    // ...
    batch.pos[tokenIdx] = static_cast<llama_pos>(0);  // 设置位置为 0
}
```

### 潜在问题

1. **`llama_memory_seq_rm` 可能不重置位置跟踪**
   - 它可能只标记 KV cache 为未使用
   - 但 llama.cpp 内部的位置记录可能仍然存在

2. **位置不连续**
   - 如果 llama.cpp 内部记录上次位置是 19
   - 我们设置 `batch.pos = 0`
   - llama.cpp 可能认为这是位置回退，导致：
     - 错误（如果严格检查）
     - 或上下文叠加（如果允许但不正确处理）

3. **上下文叠加**
   - 如果 llama.cpp 允许位置回退，它可能：
     - 保留旧的 KV cache（虽然标记为未使用）
     - 在新的位置 0 开始，但旧的 KV cache 仍然存在
     - 导致上下文叠加，性能越来越慢

## 测试方案

### 测试1：检查 llama.cpp 内部位置记录

**方法**：
- 在调用 `llama_memory_seq_rm` 后，检查 llama.cpp 内部的位置记录
- 看是否真的被重置

**问题**：llama.cpp 可能没有公开 API 来查询位置记录

### 测试2：使用不同的 seq_id

**方法**：
- 每次新请求使用不同的 `seq_id`（而不是重用 `seq_id = 0`）
- 这样可以避免位置连续性问题

**预期**：
- 如果性能稳定，说明是位置连续性问题
- 如果性能仍然退化，说明是其他问题

### 测试3：检查位置连续性错误

**方法**：
- 启用详细日志，检查是否有位置连续性错误
- 检查 llama.cpp 是否报告位置不连续

### 测试4：强制位置从上次位置继续

**方法**：
- 不重置位置，而是从上次位置继续
- 但先清理 KV cache

**预期**：
- 如果性能改善，说明是位置连续性问题

## 可能的解决方案

### 方案1：使用不同的 seq_id（推荐测试）

**方法**：
- 为每个新请求分配新的 `seq_id`
- 使用请求 ID 或时间戳作为 `seq_id`

**优点**：
- 避免位置连续性问题
- 每个请求完全独立

**挑战**：
- `seq_id` 必须 < `n_seq_max`
- 需要管理 `seq_id` 的分配和回收

### 方案2：确保位置连续性

**方法**：
- 在清理 KV cache 后，检查 llama.cpp 内部的位置记录
- 如果位置记录仍然存在，从该位置继续，而不是从 0 开始

**问题**：
- llama.cpp 可能没有公开 API 来查询位置记录
- 即使清理了 KV cache，位置记录可能仍然存在

### 方案3：完全重建 Context

**方法**：
- 定期销毁并重建 `llama_context`
- 完全清理所有状态

**缺点**：
- 性能开销大
- 可能更慢

## 下一步行动

1. **测试使用不同的 seq_id**：验证是否是位置连续性问题
2. **检查 llama.cpp 日志**：看是否有位置连续性错误
3. **分析位置设置**：确保位置连续性
