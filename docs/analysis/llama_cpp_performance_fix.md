# LlamaCppBackend 性能退化问题修复

## 问题描述

用户报告：第一次请求很快（0.9秒），但后续请求越来越慢（从2.1秒到6秒），性能持续退化。

## 根本原因分析

### 1. KV Cache 清理不彻底

**问题**：
- `clearKVCacheForSequence()` 只在位置记录 > 0 时才清理 KV cache
- 如果位置记录被重置为 0，就不会清理 KV cache
- 导致之前的 KV cache 数据残留

**代码问题**：
```cpp
// 修复前
if (it != seqPositions_.end() && it->second > 0) {
    llama_memory_seq_rm(...);  // 只在位置 > 0 时清理
}
```

### 2. seq_id 重用问题

**问题**：
- 每次批处理都使用 seq_id = 0, 1, 2...（批处理索引）
- 不同请求会重用相同的 seq_id
- 即使检测到新请求，如果位置记录为 0，KV cache 不会被清理
- 导致每次新请求都要处理之前请求残留的 KV cache

**影响**：
- 第一次请求：KV cache 为空，处理速度快
- 第二次请求：KV cache 中有第一次请求的数据，需要处理更多 tokens
- 第三次请求：KV cache 中有前两次请求的数据，需要处理更多 tokens
- ... 性能持续退化

## 修复方案

### 1. 修改 `clearKVCacheForSequence()` 总是清理

```cpp
// 修复后
void LlamaCppBackend::clearKVCacheForSequence(int32_t seqId) {
    if (!ctx_) {
        return;
    }
    
    // 总是清理 KV cache，不管位置记录是什么
    // 因为 seq_id 可能被重用，之前的 KV cache 必须清理
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem) {
        llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seqId), -1, -1);
        // ... 记录日志
    }
}
```

### 2. 在批处理开始时清理所有 seq_id

```cpp
// 修复后：在 forwardBatch() 中
// 首先，清理所有可能被重用的 seq_id 的 KV cache
for (size_t seqId = 0; seqId < batchSize; ++seqId) {
    int32_t seqIdKey = static_cast<int32_t>(seqId);
    clearKVCacheForSequence(seqIdKey);  // 总是清理，确保没有残留
}
```

## 修复效果

### 修复前
- 第一次请求：0.9秒
- 第二次请求：2.1秒（性能下降 133%）
- 第三次请求：2.8秒（性能下降 211%）
- 第四次请求：4.6秒（性能下降 411%）
- 第五次请求：5.8秒（性能下降 544%）
- 第六次请求：6.0秒（性能下降 567%）

### 修复后（预期）
- 所有请求应该保持稳定的性能
- 每次请求都清理 KV cache，确保没有残留数据
- 性能应该接近第一次请求的水平

## 技术细节

### seq_id 重用机制

在 llama.cpp 中：
- `seq_id` 是批处理中的序列标识符
- 范围是 `[0, n_seq_max)`
- 每次批处理都使用 0, 1, 2... 作为 seq_id
- 不同请求会重用相同的 seq_id

### KV Cache 管理

- KV cache 存储在 llama.cpp 的上下文中
- 每个 seq_id 有独立的 KV cache
- 必须显式清理，否则会累积
- 清理使用 `llama_memory_seq_rm(mem, seq_id, -1, -1)`

### 位置管理

- `seqPositions_` 映射用于跟踪每个 seq_id 的当前位置
- 用于区分新请求和增量推理
- 但位置记录不能完全依赖，因为 seq_id 会被重用

## 相关文件

- `src/inference/llama_cpp_backend.cpp` - 主要修复文件
  - `clearKVCacheForSequence()` - 修改为总是清理
  - `forwardBatch()` - 在开始时清理所有 seq_id

## 测试建议

1. **性能测试**：
   ```bash
   for i in {1..10}; do
       time curl -X POST http://localhost:18081/generate \
           -H "Content-Type: application/json" \
           -d '{"prompt": "1+1=", "max_tokens": 16}'
       sleep 1
   done
   ```

2. **验证要点**：
   - 所有请求的响应时间应该接近
   - 不应该有性能持续退化
   - KV cache 应该被正确清理

3. **日志检查**：
   - 检查是否有 "Cleared KV cache" 日志
   - 确认每次批处理都清理了 KV cache
