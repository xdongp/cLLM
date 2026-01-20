# 渐进式性能测试完整结果报告

## 测试概述

本报告记录了从Stage 0到Stage 5的渐进式性能测试结果，逐步添加各个组件，定位性能衰减点。

### 测试配置
- **模型**: `qwen3-0.6b-q4_k_m.gguf`
- **请求数**: 40
- **并发数**: 8
- **Prompt tokens**: 32
- **生成tokens**: 50 per request
- **目标性能**: 80+ tokens/sec

## 测试结果汇总

| Stage | 组件 | 性能 (t/s) | 相对Stage 0衰减 | 相对前阶段衰减 | 状态 |
|-------|------|-----------|---------------|---------------|------|
| **Stage 0** | LlamaCppBackend | **120.195** | 0% (基准) | - | ✅ 达标 |
| **Stage 1** | + InferenceEngine | **108.39** | -9.8% | -9.8% | ✅ 达标 |
| **Stage 2** | + ModelExecutor | **107.758** | -10.3% | -0.6% | ✅ 达标 |
| **Stage 3** | + BatchProcessor | **105.179** | -12.5% | -2.4% | ✅ 达标 |
| **Stage 4** | + SchedulerBatchProcessor | **49.1553** | **-59.1%** | **-53.3%** | ❌ 未达标 |
| **Stage 5** | + Scheduler | **崩溃** | - | - | ❌ 崩溃 |

## 详细分析

### Stage 0: LlamaCppBackend (120.195 t/s) ✅

**组件**: 直接调用`LlamaCppBackend::forwardBatch()`

**性能**: 120.195 t/s，超过目标50%

**分析**: 
- 这是性能基准，直接使用llama.cpp API
- 无额外抽象层开销
- 性能表现优秀

### Stage 1: + InferenceEngine (108.39 t/s) ✅

**组件**: `InferenceEngine::forwardBatch()` → `LlamaCppBackend::forwardBatch()`

**性能**: 108.39 t/s，衰减9.8%

**分析**:
- 添加了InferenceEngine抽象层
- 衰减在可接受范围内（<10%）
- 抽象层开销较小

### Stage 2: + ModelExecutor (107.758 t/s) ✅

**组件**: `ModelExecutor::forward()` → `InferenceEngine::forwardBatch()`

**性能**: 107.758 t/s，相对Stage 1仅衰减0.6%

**优化历史**:
- 初始性能: 71 t/s (严重衰减41%)
- 优化后: 107.758 t/s
- **关键优化**:
  1. 移除冗余`modelMutex_`锁
  2. 消除不必要的`_prepareInput`数据转换
  3. 实现logits零拷贝（使用`std::unique_ptr<kylin::Tensor>`）
  4. 条件编译调试日志和统计更新

### Stage 3: + BatchProcessor (105.179 t/s) ✅

**组件**: `BatchProcessor::processBatch()` → `ModelExecutor::forward()`

**性能**: 105.179 t/s，相对Stage 2衰减2.4%

**优化历史**:
- 初始性能: 19-20 t/s (严重衰减72%)
- 优化后: 105.179 t/s
- **关键优化**:
  1. 使用`BatchProcessor::processBatch()`而不是`BatchManager::prepareBatchInput()`
  2. 对于单请求场景，直接构建BatchInput，避免BatchManager的复杂逻辑
  3. 单token生成时，只包含新token，利用llama.cpp的增量推理能力

### Stage 4: + SchedulerBatchProcessor (49.1553 t/s) ❌

**组件**: `SchedulerBatchProcessor::processBatch()` → `BatchManager::prepareBatchInput()` → `ModelExecutor::forward()`

**性能**: 49.1553 t/s，**严重衰减53.3%**

**问题分析**:
1. **BatchManager开销**: `SchedulerBatchProcessor`内部使用`BatchManager::prepareBatchInput()`和`prepareBatchInputIncremental()`，这些方法在增量更新时存在大量数据拷贝
2. **锁竞争**: 测试代码中使用`executorMutex`保护executor访问，可能导致锁竞争
3. **增量更新效率低**: `BatchManager::prepareBatchInputIncremental()`在单请求、单token场景下，仍然需要拷贝整个`inputIds`向量

**优化方向**:
1. 优化`BatchManager::prepareBatchInputIncremental()`，实现真正的零拷贝或最小拷贝增量更新
2. 减少锁竞争，考虑使用更细粒度的锁或无锁数据结构
3. 对于单请求场景，考虑绕过BatchManager，直接使用BatchProcessor

### Stage 5: + Scheduler (崩溃) ❌

**组件**: `Scheduler::addRequest()` → `Scheduler::schedulerLoop()` → `SchedulerBatchProcessor::processBatch()`

**错误**: Sequence position不一致
```
init: the tokens of sequence 63 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 63 is X = 31
 - the tokens for sequence 63 in the input batch have a starting position of Y = 64
 it is required that the sequence positions remain consecutive: Y = X + 1
```

**问题分析**:
1. **Sequence ID管理**: Scheduler内部通过BatchManager管理sequence ID，但测试代码可能没有正确跟踪每个请求的当前位置
2. **KV Cache状态**: KV cache中存储的sequence位置与输入batch中的位置不一致
3. **并发问题**: 多个并发请求可能导致sequence ID分配和位置跟踪混乱

**修复方向**:
1. 确保Scheduler正确管理sequence ID和位置跟踪
2. 验证BatchManager在准备BatchInput时正确设置sequence positions
3. 检查并发场景下的sequence ID分配逻辑

## 性能衰减趋势

```
性能 (t/s)
120 |                                    ● Stage 0
    |                                    
110 |                    ● Stage 1
    |                    ● Stage 2
100 |                    ● Stage 3
    |                                    
 50 |                    ● Stage 4
    |                                    
  0 |____________________________________
     0    1    2    3    4    5    Stage
```

**关键发现**:
- Stage 0-3: 性能衰减平缓（<13%），均在目标之上
- Stage 4: **性能急剧下降53.3%**，主要瓶颈在`BatchManager`
- Stage 5: 崩溃，需要修复sequence ID管理问题

## 优化建议

### 短期优化（Stage 4）

1. **优化BatchManager增量更新**:
   - 实现真正的零拷贝增量更新
   - 对于单请求、单token场景，直接构建只包含新token的BatchInput
   - 考虑使用`std::vector`的移动语义或引用计数

2. **减少锁竞争**:
   - 使用更细粒度的锁
   - 考虑无锁数据结构
   - 优化executor访问模式

3. **简化SchedulerBatchProcessor**:
   - 对于单请求场景，考虑直接调用BatchProcessor
   - 减少不必要的BatchManager调用

### 长期优化（Stage 5+）

1. **修复Sequence ID管理**:
   - 确保Scheduler正确跟踪每个请求的sequence位置
   - 验证BatchManager的sequence position设置
   - 处理并发场景下的sequence ID分配

2. **优化Scheduler调度逻辑**:
   - 减少调度循环开销
   - 优化批处理形成逻辑
   - 减少不必要的状态检查

3. **HTTP层优化**:
   - 实现Stage 6+测试（HTTP Handler, Endpoint等）
   - 定位HTTP层的性能瓶颈
   - 优化请求解析和响应构建

## 结论

1. **Stage 0-3优化成功**: 通过消除冗余锁、实现零拷贝、优化数据流，成功将Stage 2和Stage 3的性能提升到100+ t/s，超过目标80 t/s

2. **Stage 4是主要瓶颈**: 性能从105 t/s降至49 t/s，主要原因是`BatchManager`的增量更新效率低。需要重点优化`BatchManager::prepareBatchInputIncremental()`

3. **Stage 5需要修复**: Sequence ID管理问题导致崩溃，需要修复后才能继续测试

4. **整体架构良好**: Stage 0-3的性能表现证明底层架构设计合理，主要问题集中在调度层和批处理管理层

## 下一步行动

1. ✅ **完成Stage 0-3测试和优化** - 已完成
2. 🔄 **优化Stage 4性能** - 进行中（49 t/s → 80+ t/s）
3. 🔄 **修复Stage 5崩溃** - 进行中
4. ⏳ **实现Stage 6+测试** - 待开始
5. ⏳ **优化HTTP层性能** - 待开始

---

**报告生成时间**: 2026-01-20
**测试工具**: `tools/incremental_benchmark.cpp`
**模型**: `qwen3-0.6b-q4_k_m.gguf`
