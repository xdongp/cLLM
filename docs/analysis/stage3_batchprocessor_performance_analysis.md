# Stage 3 (BatchProcessor) 性能分析报告

## 执行时间
2026-01-20

## 1. 问题定位

用户指出调用路径应该在 `BatchProcessor::processBatch()` 中：

```cpp
BatchOutput BatchProcessor::processBatch(const BatchInput& input) {
    if (input.requestPositions.empty()) {
        throw std::invalid_argument("Batch size cannot be zero");
    }
    
    prepareBatchInput(input);  // ⚠️ 潜在性能瓶颈
    
    BatchOutput output = executor_->forward(input);
    
    processBatchOutput(output);  // ⚠️ 潜在性能瓶颈
    
    return output;
}
```

## 2. 性能瓶颈分析

### 2.1 prepareBatchInput() 的问题

```cpp
void BatchProcessor::prepareBatchInput(const BatchInput& input) {
    // 准备批处理输入
    // 简化实现，仅用于演示
    if (input.requestPositions.size() == 1) {
        // 单个请求，无需特殊处理
        return;  // ✅ 单请求场景直接返回，无开销
    }
    
    // 计算最大序列长度
    size_t maxSeqLength = 0;
    for (const auto& pos : input.requestPositions) {
        maxSeqLength = std::max(maxSeqLength, pos.second);  // ⚠️ 循环开销
    }
    
    // 对输入进行填充
    std::vector<int> paddedInputIds = input.inputIds;  // ⚠️ 拷贝整个vector！
    _padBatch(paddedInputIds, maxSeqLength * input.requestPositions.size());
}
```

**问题**：
1. ✅ 单请求场景直接返回，无开销（这是好的）
2. ⚠️ 多请求场景会拷贝整个 `input.inputIds` vector
3. ⚠️ 但我们的测试是单请求场景，所以这个函数应该直接返回

### 2.2 processBatchOutput() 的问题

```cpp
void BatchProcessor::processBatchOutput(BatchOutput& output) {
    // 处理批处理输出
    // 简化实现，仅用于演示
    if (output.requestPositions.size() == 1) {
        // 单个请求，无需特殊处理
        return;  // ✅ 单请求场景直接返回，无开销
    }
    
    // 计算每个请求的输出长度
    std::vector<size_t> originalLengths;
    for (const auto& pos : output.requestPositions) {
        originalLengths.push_back(pos.second - pos.first);  // ⚠️ 循环开销
    }
    
    // 对输出进行去填充
    _unpadBatch(output, originalLengths);
}
```

**问题**：
1. ✅ 单请求场景直接返回，无开销（这是好的）
2. ⚠️ 但函数调用本身有开销（函数调用、参数传递等）

### 2.3 真正的性能瓶颈

**关键发现**：`BatchProcessor::processBatch()` 本身的开销很小（单请求场景下，`prepareBatchInput()` 和 `processBatchOutput()` 都直接返回）。

**真正的瓶颈可能在**：
1. **BatchManager::prepareBatchInput()** - 每次重新构建整个inputIds
2. **BatchManager::prepareBatchInputIncremental()** - 从previousInput拷贝整个vector

## 3. 当前Stage 3的实现

在 `incremental_benchmark.cpp` 的 `test_stage3_batch_manager()` 中：

```cpp
// 生成 tokens（使用增量更新）
for (int i = generatedTokens.size(); i < n_gen; ++i) {
    requestState.generatedTokens = generatedTokens;
    std::vector<RequestState> batch = {requestState};
    
    // 🔥 优化：使用增量更新，避免重新构建整个inputIds
    BatchInput input = batchManager.prepareBatchInputIncremental(
        batch, cachedInput, cachedTokenCounts
    );
    cachedInput = input;
    cachedTokenCounts = {requestState.getTotalLength()};
    
    BatchOutput output;
    {
        std::lock_guard<std::mutex> lock(executorMutex);
        output = executor.forward(input);
    }
    // ...
}
```

**问题**：
1. `prepareBatchInputIncremental()` 每次从 `previousInput` 拷贝整个 `inputIds` vector
2. 即使优化了单token场景，但 `cachedInput = input;` 又会拷贝一次
3. 对于单token生成，我们应该直接构建只包含新token的BatchInput

## 4. 优化方案

### P0: 关键优化（预期提升300-400%）

1. **优化prepareBatchInputIncremental()**
   - 对于单请求、单token增量生成，直接构建只包含新token的BatchInput
   - 避免从previousInput拷贝整个vector

2. **优化测试代码**
   - 对于单token生成，直接构建BatchInput，跳过BatchManager
   - 或者优化BatchManager，使其能够高效处理单token生成场景

### P1: 重要优化（预期提升20-30%）

3. **优化BatchProcessor::processBatch()**
   - 对于单请求场景，可以跳过`prepareBatchInput()`和`processBatchOutput()`（虽然它们已经直接返回，但函数调用仍有开销）
   - 使用内联或条件编译

4. **优化缓存管理**
   - 避免不必要的BatchInput拷贝
   - 使用移动语义

## 5. 预期性能提升

| 优化项 | 预期提升 | 累计性能 |
|--------|---------|---------|
| **当前Stage 3** | - | 19-20 t/s |
| **P0优化** | +300% | 80-100 t/s |
| **P0+P1优化** | +400% | 100-120 t/s |

**结论**：通过P0+P1优化，预期可以达到**100-120 t/s**，超过第一阶段目标。

## 6. 下一步行动

1. **立即优化prepareBatchInputIncremental()**：对于单token生成，直接构建只包含新token的BatchInput
2. **优化测试代码**：对于单token生成，直接构建BatchInput，跳过BatchManager的复杂逻辑
3. **验证优化效果**：测试Stage 3性能，确保达到80+ t/s
