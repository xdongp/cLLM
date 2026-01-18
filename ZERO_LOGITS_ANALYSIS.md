# 所有 Logits 为 0 的问题分析

## 问题描述

从日志 `logs/cllm_server_q4k_18082.log` 中可以看到：
- `WARNING: All logits are zero! This will cause uniform sampling.`
- 生成的文本是乱码：`%!C8DN#"`
- 所有 logits 都是 0，导致采样器均匀采样，从而产生乱码

## 关键发现

从日志可以看到：
1. **所有 logits 都是 0**：每次采样前都会出现警告 `WARNING: All logits are zero!`
2. **模型推理正常执行**：Transformer 模型的 forward 过程正常执行
3. **日志显示模型输出了 logits**：`TransformerModel::forward: 投影到 vocab 维度完成`
4. **但是提取的 logits 都是 0**：在采样前检查时发现所有 logits 都是 0

## 可能的原因

### 1. Logits 提取逻辑问题

从 `getLogitsForRequest` 函数看：
- `logitsOffset = lastTokenPos * vocabSize`
- 如果 `logitsOffset + vocabSize > logits.size()`，会使用 fallback 逻辑
- Fallback 逻辑中使用了 `start` 而不是 `logitsOffset`，这可能是问题

### 2. Logits 数组初始化问题

从 `ModelExecutor::_executeModelInference` 看：
- Logits 数组是从 `inferenceEngine_->forwardBatch` 返回的 `logitsTensor` 复制而来
- 如果 `logitsTensor.data()` 返回的是未初始化的数据，会导致 logits 全为 0

### 3. Offset 计算错误

如果 `logitsOffset` 计算错误，可能访问到错误的内存位置，导致读取到全 0 的数据。

## 下一步行动

1. 检查 `getLogitsForRequest` 的 offset 计算逻辑
2. 检查模型推理引擎的输出是否正确
3. 添加更多调试日志，查看实际的 logits 值
4. 检查 logits 数组的初始化是否正确
