# Stage 2-3 详细验证实施总结

## 完成时间
2026-01-23

## 概述

完成了 Stage 2 (Token Embedding) 和 Stage 3 (Layer 0 Transformer Block) 的详细验证功能，添加了完整的统计工具和中间节点验证。

---

## 实施内容

### 1. 统计工具函数 ✅

**位置**: `src/kylin/ggml_transformer.cpp`

**新增功能**:
- `TensorStats` 结构体：包含详细的统计信息
  - min/max/mean/stddev
  - NaN/Inf/Zero 计数
  - 分位数（25%, 50%, 75%, 95%, 99%）
  - 前 N 个值
- `computeTensorStats()`: 计算张量的详细统计信息
- `printTensorStats()`: 格式化打印统计信息

**代码示例**:
```cpp
TensorStats stats = computeTensorStats(data, size, 10);
printTensorStats("Stage 2: Embedding", tensor, stats);
```

**输出示例**:
```
[Kylin Debug] Stage 2: Embedding stats:
  Shape: [1024, 1, 1, 1]
  Type: F32
  Min: -0.123456, Max: 0.987654, Mean: 0.001234, StdDev: 0.123456
  NaN: 0, Inf: 0, Zero: 5
  Percentiles: P25=-0.05, P50=0.00, P75=0.05, P95=0.20, P99=0.50
  First 10 values: 0.123 -0.456 0.789 ...
```

---

### 2. Layer 0 中间节点保存 ✅

**位置**: 
- `include/cllm/kylin/ggml_transformer.h` - 添加 `Layer0DebugNodes` 结构
- `src/kylin/ggml_transformer.cpp` - 在 `buildLayerGraph` 和 `buildAttentionGraph` 中保存节点

**保存的中间节点**:
1. `attnNormOutput` - Attention 归一化输出
2. `qNormOutput` - Q 归一化输出
3. `kNormOutput` - K 归一化输出
4. `attentionOutput` - Attention 输出
5. `ffnNormOutput` - FFN 归一化输出
6. `ffnOutput` - FFN 输出

**实现方式**:
```cpp
// 在 buildLayerGraph 中
if (layerIdx == 0) {
    debugLayer0Nodes_.attnNormOutput = attnInput;
    debugLayer0Nodes_.attentionOutput = attnOutput;
    debugLayer0Nodes_.ffnNormOutput = ffnInput;
    debugLayer0Nodes_.ffnOutput = ffnOutput;
}

// 在 buildAttentionGraph 中
if (layerIdx == 0) {
    debugLayer0Nodes_.qNormOutput = q;
    debugLayer0Nodes_.kNormOutput = kNew;
}
```

---

### 3. Stage 2 详细验证 ✅

**位置**: `src/kylin/ggml_transformer.cpp::forward()`

**验证内容**:
- Embedding 输出的详细统计
- 形状验证
- 数值范围验证
- NaN/Inf 检查

**输出**:
```
[Kylin Debug] Stage 2: Embedding stats:
  Shape: [1024, 1, 1, 1]
  Type: F32
  Min: -0.123456, Max: 0.987654, Mean: 0.001234, StdDev: 0.123456
  NaN: 0, Inf: 0, Zero: 5
  Percentiles: P25=-0.05, P50=0.00, P75=0.05, P95=0.20, P99=0.50
  First 10 values: 0.123 -0.456 0.789 ...
```

---

### 4. Stage 3 详细验证 ✅

**位置**: `src/kylin/ggml_transformer.cpp::forward()`

**验证内容**:
- Stage 3.1: Layer 0 AttnNorm 输出
- Stage 3.2: Layer 0 Q Norm 输出
- Stage 3.3: Layer 0 K Norm 输出
- Stage 3.4: Layer 0 Attention 输出
- Stage 3.5: Layer 0 FFN 输出
- Stage 3: Layer 0 最终输出

**输出示例**:
```
[Kylin Debug] Stage 3.1: Layer 0 AttnNorm stats:
  Shape: [1024, 1, 1, 1]
  Type: F32
  Min: -0.234567, Max: 0.876543, Mean: 0.002345, StdDev: 0.234567
  ...

[Kylin Debug] Stage 3.2: Layer 0 Q Norm stats:
  Shape: [128, 16, 1, 1]
  Type: F32
  ...

[Kylin Debug] Stage 3: Layer 0 Final Output stats:
  Shape: [1024, 1, 1, 1]
  Type: F32
  ...
```

---

### 5. 测试脚本 ✅

**位置**: `tools/test_stage2_3.sh`

**功能**:
- 自动启动服务器
- 发送测试请求
- 提取 Stage 2-3 的验证信息
- 检查错误

**使用方法**:
```bash
./tools/test_stage2_3.sh [model_path] [prompt]
```

**输出**:
- Stage 2 验证信息
- Stage 3 验证信息
- 错误检查结果

---

## 代码变更

### 新增文件
- `tools/test_stage2_3.sh` - Stage 2-3 测试脚本

### 修改文件
1. **`include/cllm/kylin/ggml_transformer.h`**
   - 添加 `Layer0DebugNodes` 结构体定义
   - 添加 `debugLayer0Nodes_` 成员变量

2. **`src/kylin/ggml_transformer.cpp`**
   - 添加统计工具函数（`TensorStats`, `computeTensorStats`, `printTensorStats`）
   - 在构造函数中初始化 `debugLayer0Nodes_`
   - 在 `buildLayerGraph` 中保存 Layer 0 中间节点
   - 在 `buildAttentionGraph` 中保存 Q/K norm 输出
   - 在 `forward()` 中添加 Stage 2-3 的详细验证日志

---

## 验证方法

### 1. 运行测试脚本
```bash
./tools/test_stage2_3.sh
```

### 2. 查看日志
```bash
grep "Stage 2\|Stage 3" /tmp/kylin_stage2_3_test.log
```

### 3. 手动测试
```bash
# 启动服务器
./build/bin/cllm_server --config config/config.yaml > /tmp/test.log 2>&1 &

# 发送请求
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hi", "max_tokens": 1, "temperature": 0.0}'

# 查看验证信息
grep "\[Kylin Debug\]" /tmp/test.log | grep "Stage"
```

---

## 预期输出

### Stage 2 输出
- ✅ Embedding 形状: `[1024, 1]`
- ✅ 数值范围合理（通常在 [-1, 1]）
- ✅ 无 NaN/Inf
- ✅ 均值接近 0
- ✅ 标准差合理

### Stage 3 输出
- ✅ 所有中间节点形状正确
- ✅ 数值范围合理
- ✅ 无 NaN/Inf
- ✅ 各阶段输出统计正常

---

## 下一步

### 短期（1-2天）
1. **实现与 llama_cpp 的自动对比**
   - 提取 llama_cpp 的中间结果
   - 对比数值差异
   - 生成对比报告

2. **完善 Stage 4 的验证**
   - 添加注意力计算的详细验证
   - 验证 RoPE、GQA 等

### 中期（3-5天）
1. **实现多层验证**
   - 验证所有层的输出
   - 检查数值稳定性

2. **性能基准测试**
   - 记录各阶段的性能
   - 对比 llama_cpp 的性能

---

## 注意事项

1. **CPU 后端**: 建议使用 CPU 后端进行调试，Metal 后端可能有段错误
2. **数据访问**: 使用 try-catch 保护数据访问，避免段错误
3. **内存限制**: 只验证合理大小的张量（< 1M 元素）
4. **日志级别**: 使用 INFO 级别确保验证信息被打印

---

## 相关文档

- [kylin_debug_plan.md](./kylin_debug_plan.md) - 完整调试规划
- [kylin_stage_implementation_guide.md](./kylin_stage_implementation_guide.md) - 实施指南
- [kylin_flow_diagram.md](./kylin_flow_diagram.md) - 流程调用图

---

**完成状态**: ✅ Stage 2-3 详细验证已完成  
**最后更新**: 2026-01-23
