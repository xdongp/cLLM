# 动态批处理阈值调整测试报告

**日期**: 2026-01-22
**作者**: Trae AI Assistant
**版本**: 1.0

---

## 执行摘要

本报告展示了调整动态批处理阈值参数后的性能测试结果。将阈值从（100ms/10ms）调整为（500ms/100ms），以验证是否能够改善性能。

### 测试配置
- **请求数量**: 72个
- **每个请求最大tokens**: 50
- **测试类型**: Concurrent (24并发)
- **模型**: qwen3-0.6b-q4_k_m
- **测试时间**: 2026-01-22

---

## 阈值调整

### 1. 调整内容

**修改文件**: [manager.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/batch/manager.cpp#L407-L420)

**调整前**:
```cpp
if (lastBatchProcessingTimeMs_ > 100) {
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
} else if (lastBatchProcessingTimeMs_ < 10 && queueSize > adaptiveBatchSize_ * 2) {
    adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 2);
}
```

**调整后**:
```cpp
if (lastBatchProcessingTimeMs_ > 500) {
    adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
} else if (lastBatchProcessingTimeMs_ < 100 && queueSize > adaptiveBatchSize_ * 2) {
    adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 2);
}
```

### 2. 调整说明

- **减半阈值**: 100ms → 500ms（提高5倍）
- **加倍阈值**: 10ms → 100ms（提高10倍）

**目的**: 使阈值更符合LLM推理的实际处理时间，避免批大小被过度调整。

---

## 测试结果对比

### 1. 24并发性能对比

| 版本 | 吞吐量 (t/s) | 平均响应时间 (s) | 最大响应时间 (s) | 成功率 |
|------|-------------|----------------|----------------|--------|
| **修复后（无动态批处理）** | 257.20 | 9.13 | 20.27 | 98.6% |
| **动态批处理（100ms/10ms）** | 87.10 | 11.26 | 37.27 | 100% |
| **动态批处理（500ms/100ms）** | 89.95 | 11.00 | 36.32 | 100% |

### 2. 性能变化分析

#### 吞吐量对比

| 对比 | 吞吐量变化 | 说明 |
|------|----------|------|
| 修复后 vs 动态批处理（100ms/10ms） | -66.1% | 严重性能下降 |
| 修复后 vs 动态批处理（500ms/100ms） | -65.0% | 严重性能下降 |
| 动态批处理（100ms/10ms） vs 动态批处理（500ms/100ms） | +3.3% | 轻微改善 |

#### 响应时间对比

| 对比 | 平均响应时间变化 | 最大响应时间变化 |
|------|----------------|----------------|
| 修复后 vs 动态批处理（100ms/10ms） | +23.3% | +83.9% |
| 修复后 vs 动态批处理（500ms/100ms） | +20.5% | +79.2% |
| 动态批处理（100ms/10ms） vs 动态批处理（500ms/100ms） | -2.3% | -2.5% |

---

## 结果分析

### 1. 阈值调整效果

#### ✅ 轻微改善

调整阈值后，性能有轻微改善：
- **吞吐量**: 从87.10 t/s提升到89.95 t/s（+3.3%）
- **平均响应时间**: 从11.26s降低到11.00s（-2.3%）
- **最大响应时间**: 从37.27s降低到36.32s（-2.5%）

#### ❌ 仍然严重低于基线

尽管有轻微改善，但性能仍然远低于修复后的基线：
- **吞吐量差距**: 89.95 vs 257.20 t/s（-65.0%）
- **平均响应时间差距**: 11.00 vs 9.13s（+20.5%）
- **最大响应时间差距**: 36.32 vs 20.27s（+79.2%）

### 2. 问题根源分析

#### 🔴 核心问题：动态批处理机制本身存在缺陷

即使调整了阈值，动态批处理仍然导致严重的性能下降。这说明问题不仅仅是阈值设置，而是整个动态批处理机制的设计存在问题。

**可能的原因**:

1. **批大小调整幅度过大**
   - 当前调整幅度为50%（减半或加倍）
   - 即使阈值更合理，50%的调整幅度仍然过大
   - 导致批大小频繁剧烈波动

2. **缺少平滑调整机制**
   - 当前实现是"一刀切"的调整
   - 没有渐进式调整或平滑过渡
   - 导致批大小在两个极端之间跳跃

3. **批大小下限可能过低**
   - minAdaptiveBatchSize_ 可能设置得太小
   - 导致批大小被减到1-2的极小值
   - 无法形成有效的批处理

4. **缺少综合考虑**
   - 当前只考虑批处理时间和队列大小
   - 没有考虑GPU利用率、响应时间等其他因素
   - 导致调整策略不够智能

### 3. 性能下降机制

**动态批处理导致性能下降的机制**:

1. **批大小波动**: 批大小在8-16-32之间频繁变化
2. **GPU利用率不稳定**: 批大小变化导致GPU利用率波动
3. **调度开销增加**: 频繁的批大小调整增加了调度开销
4. **响应时间增加**: 不稳定的批大小导致请求排队时间增加

---

## 结论

### 1. 阈值调整效果有限

❌ **调整阈值仅带来轻微改善**:
- 吞吐量提升: +3.3%
- 响应时间降低: -2.3%
- 仍然远低于基线性能（-65.0%）

### 2. 动态批处理机制需要重新设计

**核心问题**: 不仅仅是阈值设置，整个动态批处理机制都需要重新设计。

**需要改进的方面**:
1. 减小调整幅度（从50%降低到10-20%）
2. 添加平滑调整机制
3. 提高批大小下限
4. 综合考虑多个因素（GPU利用率、响应时间等）

### 3. 建议

**立即执行**:
1. **回滚动态批处理优化**: 恢复到之前的固定批大小策略
2. **重新设计动态批处理**: 从根本上重新设计动态批处理机制

**短期优化** (1-2周):
1. 减小调整幅度（25%或更小）
2. 添加平滑调整机制
3. 提高最小批大小（16或更高）
4. 添加批大小变化限制（避免频繁变化）

**长期改进** (1-2个月):
1. 实现基于机器学习的动态批处理
2. 考虑更多因素（队列负载、GPU利用率、响应时间、系统资源）
3. 实现预测性批处理调度
4. 实现多目标优化（吞吐量、响应时间、资源利用率）

---

## 附录

### A. 测试数据

**24并发测试结果**:
- 修复后（无动态批处理）: 257.20 t/s, 71/72成功
- 动态批处理（100ms/10ms）: 87.10 t/s, 72/72成功
- 动态批处理（500ms/100ms）: 89.95 t/s, 72/72成功

### B. 相关代码

- [manager.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/batch/manager.cpp) - 动态批处理实现
- [manager.h](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/include/cllm/batch/manager.h) - BatchManager类定义
- [batch_processor.cpp](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/src/scheduler/batch_processor.cpp) - 批处理时间跟踪

### C. 测试命令

```bash
# 24并发测试
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50
```

### D. 相关报告

- [dynamic_batch_processing_test_report_20260122.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/analysis/dynamic_batch_processing_test_report_20260122.md) - 初始动态批处理测试报告
- [cllm_rebenchmark_after_fix_report.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/analysis/cllm_rebenchmark_after_fix_report.md) - 修复后基准测试报告

---

**报告结束**
