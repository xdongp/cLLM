# 动态批处理优化测试报告

**日期**: 2026-01-22
**作者**: Trae AI Assistant
**版本**: 1.0

---

## 执行摘要

本报告展示了实现动态批处理（Adaptive Batch Processing）优化后的性能测试结果。动态批处理旨在根据系统负载动态调整批处理大小，以优化资源利用率和响应时间。

### 测试配置
- **请求数量**: 72个
- **每个请求最大tokens**: 50
- **测试类型**: Concurrent (8/16/24/32并发)
- **模型**: qwen3-0.6b-q4_k_m
- **测试时间**: 2026-01-22

---

## 动态批处理实现

### 1. 实现概述

动态批处理通过以下机制实现：

1. **自适应批大小计算**: 根据上次批处理时间和队列大小动态调整批大小
2. **批处理时间跟踪**: 记录每次批处理的处理时间
3. **动态调整策略**:
   - 如果上次批处理时间 > 100ms，减半批大小
   - 如果上次批处理时间 < 10ms 且队列较大，加倍批大小

### 2. 核心代码

**[manager.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/batch/manager.cpp#L245-L265)**:

```cpp
size_t BatchManager::adaptiveBatchSize(size_t queueSize, size_t runningCount) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    if (lastBatchProcessingTimeMs_ > 100) {
        adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
        CLLM_DEBUG("[BatchManager::adaptiveBatchSize] Last batch processing time too long (%zu ms), reducing batch size to %zu",
                  lastBatchProcessingTimeMs_, adaptiveBatchSize_);
    } else if (lastBatchProcessingTimeMs_ < 10 && queueSize > adaptiveBatchSize_ * 2) {
        adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 2);
        CLLM_DEBUG("[BatchManager::adaptiveBatchSize] Last batch processing time short (%zu ms) and queue large (%zu), increasing batch size to %zu",
                  lastBatchProcessingTimeMs_, queueSize, adaptiveBatchSize_);
    }
    
    return adaptiveBatchSize_;
}
```

**[batch_processor.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/batch_processor.cpp#L42-L70)**:

```cpp
void SchedulerBatchProcessor::processBatch(std::vector<RequestState>& batch) {
    auto batchStartTime = std::chrono::steady_clock::now();
    
    // ... 批处理逻辑 ...
    
    auto batchEndTime = std::chrono::steady_clock::now();
    auto processingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        batchEndTime - batchStartTime
    ).count();
    
    if (batchManager_) {
        batchManager_->updateBatchProcessingTime(processingTimeMs);
    }
}
```

---

## 测试结果

### 1. 动态批处理优化后性能

| 并发数 | 成功请求 | 失败请求 | 总吞吐量 (t/s) | 平均响应时间 (s) | 总测试时间 (s) |
|--------|---------|---------|---------------|----------------|---------------|
| **8** | 71/72 | 1 | **80.97** | 4.69 | 43.85 |
| **16** | 71/72 | 1 | **85.31** | 8.21 | 41.62 |
| **24** | 72/72 | 0 | **87.10** | 11.26 | 41.33 |
| **32** | 72/72 | 0 | **85.99** | 14.43 | 41.87 |

### 2. 性能趋势分析

#### 吞吐量趋势
- **并发8**: 80.97 t/s
- **并发16**: 85.31 t/s（+5.4%）
- **并发24**: 87.10 t/s（+2.1%）
- **并发32**: 85.99 t/s（-1.3%）

**观察**: 吞吐量在并发24时达到峰值（87.10 t/s），但整体性能较低

#### 稳定性
- **并发8**: 98.6% 成功率（1个失败）
- **并发16**: 98.6% 成功率（1个失败）
- **并发24**: 100% 成功率 ✅
- **并发32**: 100% 成功率 ✅

---

## 与之前优化对比

### 1. 修复后（无动态批处理） vs 动态批处理优化

| 并发数 | 修复后吞吐量 (t/s) | 动态批处理吞吐量 (t/s) | 变化 | 失败数变化 |
|--------|------------------|----------------------|------|----------|
| **8** | 137.73 | 80.97 | **-41.2%** ❌ | 0 → 1 |
| **16** | 289.00 | 85.31 | **-70.5%** ❌ | 0 → 1 |
| **24** | 257.20 | 87.10 | **-66.1%** ❌ | 1 → 0 ✅ |
| **32** | 347.99 | 85.99 | **-75.3%** ❌ | 0 → 0 |

### 2. 性能下降分析

#### 🔴 严重性能下降

动态批处理优化导致**严重的性能下降**：

1. **并发8**: 从137.73 t/s降到80.97 t/s（-41.2%）
2. **并发16**: 从289.00 t/s降到85.31 t/s（-70.5%）
3. **并发24**: 从257.20 t/s降到87.10 t/s（-66.1%）
4. **并发32**: 从347.99 t/s降到85.99 t/s（-75.3%）

**平均性能下降**: **-63.3%**

#### 响应时间增加

| 并发数 | 修复后响应时间 (s) | 动态批处理响应时间 (s) | 变化 |
|--------|------------------|----------------------|------|
| **8** | 2.93 | 4.69 | +60.1% |
| **16** | 5.36 | 8.21 | +53.2% |
| **24** | 9.13 | 11.26 | +23.3% |
| **32** | 11.81 | 14.43 | +22.2% |

---

## 问题分析

### 1. 根本原因

动态批处理优化导致性能严重下降的根本原因：

#### 🔴 问题1: 批大小调整过于激进

**当前实现**:
```cpp
if (lastBatchProcessingTimeMs_ > 100) {
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
}
```

**问题**:
- 批处理时间 > 100ms 就减半批大小
- 对于LLM推理，100ms的批处理时间非常短
- 这导致批大小频繁被减半，无法充分利用GPU并行能力

#### 🔴 问题2: 阈值设置不合理

**当前阈值**:
- 减半阈值: 100ms
- 加倍阈值: 10ms

**问题**:
- 10ms的阈值对于LLM推理来说太短
- 几乎不可能在10ms内完成批处理
- 导致批大小只会减少，不会增加

#### 🔴 问题3: 缺少下限保护

**当前实现**:
```cpp
adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
```

**问题**:
- minAdaptiveBatchSize_ 可能设置得太小
- 导致批大小被减到非常小的值
- 无法形成有效的批处理

### 2. 性能下降机制

**动态批处理的负面影响**:

1. **批大小过小**: 由于频繁减半，批大小可能降到1-2
2. **GPU利用率低**: 小批大小无法充分利用GPU并行能力
3. **批处理开销增加**: 更多的小批处理意味着更多的调度开销
4. **响应时间增加**: 小批处理导致请求排队时间增加

---

## 建议改进方案

### 1. 调整阈值参数

**建议修改**:

```cpp
// 🔥 修复: 调整阈值到合理的范围
constexpr size_t BATCH_PROCESSING_TIME_THRESHOLD_HIGH = 500;  // 500ms
constexpr size_t BATCH_PROCESSING_TIME_THRESHOLD_LOW = 100;  // 100ms

if (lastBatchProcessingTimeMs_ > BATCH_PROCESSING_TIME_THRESHOLD_HIGH) {
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ * 3 / 4);  // 减少25%
} else if (lastBatchProcessingTimeMs_ < BATCH_PROCESSING_TIME_THRESHOLD_LOW && queueSize > adaptiveBatchSize_ * 2) {
    adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 5 / 4);  // 增加25%
}
```

**改进点**:
- 减半阈值从100ms提升到500ms
- 加倍阈值从10ms提升到100ms
- 调整幅度从50%降低到25%，更平滑

### 2. 增加批大小下限

**建议修改**:

```cpp
// 🔥 修复: 提高最小批大小
constexpr size_t MIN_ADAPTIVE_BATCH_SIZE = 8;  // 最小批大小为8
constexpr size_t MAX_ADAPTIVE_BATCH_SIZE = 64;  // 最大批大小为64
```

**改进点**:
- 确保批大小不会太小
- 保持一定的GPU利用率

### 3. 添加平滑调整机制

**建议修改**:

```cpp
// 🔥 修复: 使用平滑调整而不是激进调整
if (lastBatchProcessingTimeMs_ > BATCH_PROCESSING_TIME_THRESHOLD_HIGH) {
    // 渐进式减少，每次减少10-20%
    size_t reduction = std::max(1UL, adaptiveBatchSize_ / 10);
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ - reduction);
} else if (lastBatchProcessingTimeMs_ < BATCH_PROCESSING_TIME_THRESHOLD_LOW && queueSize > adaptiveBatchSize_ * 2) {
    // 渐进式增加，每次增加10-20%
    size_t increase = std::max(1UL, adaptiveBatchSize_ / 10);
    adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ + increase);
}
```

**改进点**:
- 更平滑的调整机制
- 避免批大小剧烈波动

### 4. 考虑队列负载

**建议修改**:

```cpp
// 🔥 修复: 综合考虑队列负载和批处理时间
float queueLoadFactor = static_cast<float>(queueSize) / maxQueueSize_;
float timeLoadFactor = static_cast<float>(lastBatchProcessingTimeMs_) / 1000.0f;

if (queueLoadFactor > 0.8 && timeLoadFactor < 0.5) {
    // 队列负载高，批处理时间短，增加批大小
    adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 5 / 4);
} else if (timeLoadFactor > 1.0) {
    // 批处理时间长，减少批大小
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ * 3 / 4);
}
```

**改进点**:
- 综合考虑多个因素
- 更智能的调整策略

---

## 结论

### 1. 动态批处理优化失败

❌ **动态批处理优化导致严重的性能下降**:
- 平均性能下降: **-63.3%**
- 所有并发级别性能都大幅下降
- 响应时间增加22%-60%

### 2. 根本原因

**核心问题**: 批大小调整策略过于激进且阈值设置不合理

1. **阈值过短**: 100ms和10ms的阈值对于LLM推理来说太短
2. **调整幅度过大**: 50%的调整幅度导致批大小剧烈波动
3. **缺少保护机制**: 批大小可能降到非常小的值

### 3. 建议

**立即执行**:
1. **回滚动态批处理优化**: 恢复到之前的固定批大小策略
2. **重新设计动态批处理**: 使用更合理的阈值和调整策略

**短期优化** (1-2周):
1. 调整阈值参数（500ms/100ms）
2. 减小调整幅度（25%）
3. 提高最小批大小（8）

**长期改进** (1-2个月):
1. 实现基于机器学习的动态批处理
2. 考虑更多因素（队列负载、GPU利用率、响应时间）
3. 实现预测性批处理调度

---

## 附录

### A. 测试数据

**动态批处理测试结果**:
- 并发8: 80.97 t/s, 71/72成功
- 并发16: 85.31 t/s, 71/72成功
- 并发24: 87.10 t/s, 72/72成功
- 并发32: 85.99 t/s, 72/72成功

**修复后测试结果** (来自 [cllm_rebenchmark_after_fix_report.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/analysis/cllm_rebenchmark_after_fix_report.md)):
- 并发8: 137.73 t/s, 72/72成功
- 并发16: 289.00 t/s, 72/72成功
- 并发24: 257.20 t/s, 71/72成功
- 并发32: 347.99 t/s, 72/72成功

### B. 相关代码

- [manager.h](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/include/cllm/batch/manager.h) - BatchManager类定义
- [manager.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/batch/manager.cpp) - 动态批处理实现
- [batch_processor.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/batch_processor.cpp) - 批处理时间跟踪

### C. 测试命令

```bash
# 并发8
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50

# 并发16
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50

# 并发24
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50

# 并发32
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50
```

---

**报告结束**
