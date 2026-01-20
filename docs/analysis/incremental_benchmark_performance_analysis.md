# 渐进式性能测试分析报告

## 执行时间
2026-01-20

## 1. 性能衰减分析

通过渐进式测试，从底层开始逐步添加各个组件，我们精确定位了性能衰减点：

| 阶段 | 组件 | 性能 | 相对Stage 0衰减 | 状态 |
|------|------|------|----------------|------|
| **Stage 0** | LlamaCppBackend::forwardBatch() | **120.0 t/s** | 0% (基准) | ✅ 超过目标 |
| **Stage 1** | + InferenceEngine::forwardBatch() | **116.0 t/s** | -3% | ✅ 超过目标 |
| **Stage 2** | + ModelExecutor::forward() | **71.0 t/s** | **-41%** | ❌ 需优化 |
| **Stage 3** | + BatchManager::prepareBatchInput() | **20-27 t/s** | **-72%** | ❌ 需优化 |

## 2. 性能瓶颈定位

### 2.1 Stage 2 (ModelExecutor) - 衰减41%

**主要瓶颈**：数据拷贝

1. **Tensor到FloatArray的拷贝**（607KB per token）
   - 每次forward()调用，需要从`inference::Tensor`拷贝到`FloatArray`
   - 对于50个tokens的生成：50次 × 607KB = 30.35MB总拷贝量

2. **getLogitsForRequest()的拷贝**（607KB per call）
   - 即使使用Tensor，仍需要创建FloatArray result并拷贝数据
   - 50次调用 × 607KB = 30.35MB总拷贝量

**已优化**：
- ✅ 移除modelMutex_锁（因为InferenceEngine内部已有锁）
- ✅ 使用memcpy替代循环拷贝
- ✅ BatchOutput支持直接使用Tensor（logitsTensor字段）

**待优化**：
- ⚠️ getLogitsForRequest()仍需要返回FloatArray，无法完全避免拷贝
- ⚠️ 可以考虑延迟拷贝或使用视图（但需要修改返回类型）

### 2.2 Stage 3 (BatchManager) - 衰减72%（最严重）

**主要瓶颈**：每次重新构建整个inputIds

1. **prepareBatchInput()每次都重新构建**（20 t/s）
   - 对于单token生成，每次都需要重新构建整个inputIds（包括prompt和所有已生成的tokens）
   - 例如：prompt 32 tokens + 已生成49 tokens = 每次重新构建81个tokens
   - 50次调用 × 81 tokens = 4,050次token拷贝

2. **prepareBatchInputIncremental()仍从previousInput拷贝**（20-27 t/s）
   - 虽然尝试增量更新，但仍从previousInput中拷贝已有tokens
   - 没有真正实现增量更新（应该直接重用previousInput，只追加新token）

**已优化**：
- ✅ prepareBatchInput()对单token生成场景优化（只插入最后一个token）
- ✅ 预分配内存，减少重新分配开销
- ✅ 直接插入，避免中间vector的拷贝

**待优化**：
- ⚠️ prepareBatchInputIncremental()应该直接重用previousInput.inputIds，而不是拷贝
- ⚠️ 对于单请求场景，可以直接重用，只追加新token

## 3. 关键发现

### 3.1 Stage 2的主要问题

虽然已经优化了部分内容，但数据拷贝仍然是瓶颈：
- `_executeModelInference()`：从Tensor拷贝到FloatArray（607KB per token）
- `getLogitsForRequest()`：从logits拷贝到result（607KB per call）

对于50个tokens的生成，总拷贝量：60.7MB

### 3.2 Stage 3的主要问题

BatchManager的批处理逻辑导致了大量不必要的数据拷贝：
- `prepareBatchInput()`：每次重新构建整个inputIds
- `prepareBatchInputIncremental()`：从previousInput拷贝数据，没有真正实现增量更新

**关键洞察**：对于单token生成场景，我们只需要最后一个token，不需要重新构建整个inputIds。但BatchManager的设计假设需要完整的inputIds。

## 4. 优化建议

### P0: 关键优化（预期提升50-70%）

1. **优化prepareBatchInputIncremental()**
   - 直接重用previousInput.inputIds，只追加新token
   - 避免从previousInput拷贝已有tokens
   - 对于单请求场景，特殊处理

2. **优化Stage 2的数据拷贝**
   - 延迟拷贝：只在真正需要时才拷贝
   - 使用Tensor视图：如果可以，直接使用Tensor而不是FloatArray

### P1: 重要优化（预期提升20-30%）

3. **优化getLogitsForRequest()**
   - 对于单token生成，直接返回最后一个token的logits
   - 避免创建完整的FloatArray

4. **优化BatchManager的批处理逻辑**
   - 缓存更多信息，避免重复计算
   - 对于单请求场景，使用快速路径

## 5. 预期性能提升

| 优化项 | 预期提升 | 累计性能 |
|--------|---------|---------|
| **当前Stage 3** | - | 20-27 t/s |
| **P0优化** | +50% | 30-40 t/s |
| **P0+P1优化** | +150% | 70-80 t/s |

**结论**：通过P0+P1优化，预期可以达到**70-80 t/s**，接近第一阶段目标。

## 6. 下一步行动

1. **立即优化prepareBatchInputIncremental()**：直接重用previousInput，避免拷贝
2. **优化Stage 2的数据拷贝**：延迟拷贝或使用Tensor视图
3. **继续实现Stage 4+**：验证Scheduler等上层组件的性能影响

## 7. 与Direct Benchmark的对比

| 系统 | 性能 | 说明 |
|------|------|------|
| **direct_benchmark** | **131 t/s** | 直接调用后端，无中间层 |
| **Stage 0 (LlamaCppBackend)** | **120 t/s** | 接近direct_benchmark |
| **Stage 1 (InferenceEngine)** | **116 t/s** | 小幅衰减（3%） |
| **Stage 2 (ModelExecutor)** | **71 t/s** | 严重衰减（41%） |
| **Stage 3 (BatchManager)** | **20-27 t/s** | 最严重衰减（72%） |
| **完整系统** | **45.95 t/s** | 存在调度层和HTTP层开销 |

**分析**：
- Stage 0-1的性能接近direct_benchmark，说明底层后端性能良好
- Stage 2-3的严重衰减说明中间层（ModelExecutor、BatchManager）存在大量不必要的开销
- 完整系统的性能（45.95 t/s）虽然低于direct_benchmark，但高于Stage 3，说明Scheduler等上层组件可能有一些优化
