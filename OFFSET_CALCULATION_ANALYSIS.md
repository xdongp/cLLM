# getLogitsForRequest Offset 计算分析

## Logits 数组结构

从 `ModelExecutor::_executeModelInference` 可以看到：
- Logits 数组大小：`totalTokens * config_.vocabSize`
- Logits 数组布局：`[token_0_logits, token_1_logits, ..., token_N_logits]`
- 每个 token 对应 `vocabSize` 个 logits
- Token i 的 logits 在数组中的位置：`[i * vocabSize, (i + 1) * vocabSize)`

## requestPositions 的含义

从 `BatchManager::prepareBatchInput` 可以看到：
- `requestPositions[i] = {currentPos, currentPos + inputIds.size()}`
- `{start, end}` 表示请求的 token 范围：`[start, end)`（左闭右开区间）
- `start` 是请求的起始 token 位置
- `end` 是请求的结束 token 位置（不包含）

## getLogitsForRequest 的 Offset 计算

```cpp
auto [start, end] = requestPositions[requestIndex];
size_t lastTokenPos = end - 1;  // 最后一个 token 的位置（因为 end 是不包含的）
size_t logitsOffset = lastTokenPos * vocabSize;  // logits 数组中的偏移量
```

### 分析

1. **lastTokenPos = end - 1**：
   - 正确！因为 `end` 是不包含的，所以最后一个 token 的位置是 `end - 1`

2. **logitsOffset = lastTokenPos * vocabSize**：
   - 正确！Token `lastTokenPos` 的 logits 在数组中的偏移量是 `lastTokenPos * vocabSize`

3. **提取 logits**：
   ```cpp
   for (size_t i = 0; i < vocabSize; ++i) {
       result[i] = logits[logitsOffset + i];
   }
   ```
   - 正确！从 `logitsOffset` 开始提取 `vocabSize` 个 logits

## 边界检查

代码中有边界检查：
```cpp
if (logitsOffset + vocabSize > logits.size()) {
    // 使用 fallback 逻辑
}
```

但是 fallback 逻辑有问题：
```cpp
size_t availableSize = (logits.size() > start) ? std::min(vocabSize, logits.size() - start) : 0;
for (size_t i = 0; i < availableSize; ++i) {
    result[i] = logits[start + i];  // ❌ 错误！应该使用 logitsOffset，而不是 start
}
```

**问题**：fallback 逻辑使用了 `start` 而不是 `logitsOffset`，这是错误的！

## 结论

1. **主要的 offset 计算是正确的**：`logitsOffset = lastTokenPos * vocabSize`
2. **Fallback 逻辑有 bug**：应该使用 `logitsOffset` 而不是 `start`
3. **日志格式化有问题**：`CLLM_DEBUG` 使用 printf 风格，但代码中使用了 `{}` 占位符（fmt 风格）
