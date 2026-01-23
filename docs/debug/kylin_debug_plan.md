# Kylin 后端分阶段调试规划

## 概述

本文档定义了 Kylin 后端的完整调试流程，采用分阶段测试方法，确保每一步都正确执行。参考 `incremental_benchmark_all_stages_final_results.md` 的测试方法，逐步验证每个组件的正确性。

### 测试目标

1. **正确性验证**：确保每个阶段的输出与预期一致
2. **数值精度验证**：确保计算过程中没有 NaN/Inf，数值范围合理
3. **与 llama_cpp 对比**：每个阶段都与 llama_cpp 后端进行对比
4. **性能基准**：记录每个阶段的性能指标

### 测试模型

- **模型**: `qwen3-0.6b-q4_k_m.gguf`
- **测试 Prompt**: "Hi" (简单), "What is the capital of France?" (中等), 更长文本 (复杂)
- **Temperature**: 0.0 (确定性输出)
- **Max tokens**: 根据阶段调整

---

## Kylin 完整流程梳理

### 初始化流程

```
1. KylinBackend 构造
   ├─ 检测模型格式 (.gguf / .bin)
   ├─ 设置 useGGMLDirect_ 标志
   ├─ 读取设备后端配置 (CPU/Metal/CUDA)
   └─ 创建 GGMLTransformerModel 或 TransformerModel

2. KylinBackend::initialize()
   ├─ initializeGGMLDirect() [GGUF 模式]
   │  └─ GGMLTransformerModel::loadFromGGUF()
   │     ├─ 创建 GGUFLoader
   │     ├─ 加载模型配置
   │     ├─ 创建 GGML 上下文 (weight, kvCache, compute)
   │     ├─ 映射权重 (mapWeights)
   │     └─ 分配 KV Cache
   └─ initialize() [.bin 模式]
      ├─ 加载权重 (loadRealWeights)
      └─ 绑定权重到模型 (bindWeightsToModel)
```

### 推理流程

```
3. KylinBackend::forward()
   └─ GGMLTransformerModel::forward()
      ├─ 重置计算上下文
      ├─ buildForwardGraph()
      │  ├─ Stage 0: 创建输入张量
      │  ├─ Stage 1: Token Embedding Lookup
      │  ├─ Stage 2-N: Transformer Layers (buildLayerGraph)
      │  │  ├─ Stage 2.1: Attention 归一化
      │  │  ├─ Stage 2.2: Self-Attention (buildAttentionGraph)
      │  │  │  ├─ QKV 投影
      │  │  │  ├─ Reshape 为多头格式
      │  │  │  ├─ Q/K 归一化
      │  │  │  ├─ RoPE 位置编码
      │  │  │  ├─ KV Cache 更新/读取
      │  │  │  ├─ GQA 扩展 (如需要)
      │  │  │  ├─ 注意力计算 (Q@K^T, softmax, @V)
      │  │  │  └─ 输出投影
      │  │  ├─ Stage 2.3: 残差连接 (Attention)
      │  │  ├─ Stage 2.4: FFN 归一化
      │  │  ├─ Stage 2.5: FFN (buildFFNGraph)
      │  │  │  ├─ Gate 投影
      │  │  │  ├─ Up 投影
      │  │  │  ├─ SiLU 激活
      │  │  │  └─ Down 投影
      │  │  └─ Stage 2.6: 残差连接 (FFN)
      │  ├─ Stage N+1: 最终归一化
      │  └─ Stage N+2: LM Head 投影
      ├─ 构建计算图
      ├─ 执行计算
      └─ 提取 logits
```

---

## 分阶段测试计划

### Stage 0: 基础环境验证 ✅

**目标**: 验证基础环境和模型加载

**测试内容**:
- [x] KylinBackend 构造成功
- [x] GGMLTransformerModel 创建成功
- [x] GGUF 文件检测正确
- [x] 设备后端配置正确

**验证方法**:
```bash
# 检查日志
grep "KylinBackend.*Initializing" /tmp/kylin_test.log
grep "GGMLTransformerModel.*Created" /tmp/kylin_test.log
grep "Detected GGUF format" /tmp/kylin_test.log
```

**预期结果**:
- 所有组件成功初始化
- 无错误或警告

**状态**: ✅ 已完成

---

### Stage 1: 模型加载验证 ✅

**目标**: 验证 GGUF 模型加载和权重映射

**测试内容**:
- [x] GGUFLoader 创建成功
- [x] 模型配置加载正确
- [x] GGML 上下文创建成功
- [x] 权重映射完成
- [x] KV Cache 分配成功

**验证方法**:
```bash
# 检查配置
grep "Model config:" /tmp/kylin_test.log
grep "weights mapped" /tmp/kylin_test.log
grep "KV cache allocated" /tmp/kylin_test.log
```

**预期结果**:
- 配置参数正确 (vocab=151936, hidden=1024, layers=28, heads=16, kv_heads=8)
- 所有权重成功映射
- KV Cache 分配成功

**状态**: ✅ 已完成

---

### Stage 2: Token Embedding 验证 🔄

**目标**: 验证 Token Embedding Lookup 的正确性

**测试内容**:
- [ ] 输入 token ID 正确传递
- [ ] `ggml_get_rows` 正确执行
- [ ] Embedding 输出形状正确 `[hidden, seq_len]`
- [ ] Embedding 数值范围合理
- [ ] 与 llama_cpp 的 embedding 输出对比

**测试用例**:
```cpp
// 测试用例 1: 单个 token
inputIds = [9707]  // "Hello"
expected_shape = [1024, 1]

// 测试用例 2: 多个 tokens
inputIds = [9707, 11]  // "Hello "
expected_shape = [1024, 2]
```

**验证方法**:
```bash
# 检查 embedding 统计
grep "\[Kylin Debug\] Embedding" /tmp/kylin_test.log

# 对比 llama_cpp (需要添加类似日志)
```

**预期结果**:
- Shape: `[1024, seq_len]`
- Min/Max: 通常在 [-1, 1] 范围内
- Mean: 接近 0
- NaN/Inf: 0
- 前10个值与 llama_cpp 一致（或非常接近）

**状态**: 🔄 进行中（已添加调试日志）

---

### Stage 3: 第一层 Transformer Block 验证 🔄

**目标**: 验证第一层 Transformer Block 的完整计算流程

**测试内容**:
- [ ] Attention 归一化正确
- [ ] QKV 投影正确
- [ ] Q/K 归一化正确
- [ ] RoPE 正确应用
- [ ] 注意力计算正确
- [ ] FFN 计算正确
- [ ] 残差连接正确
- [ ] Layer 0 输出统计合理

**测试用例**:
```cpp
// 测试用例: 单 token 首次推理
inputIds = [9707]  // "Hello"
startPos = 0
seqLen = 1

// 验证点:
// 1. Attention 输入: [1024, 1]
// 2. QKV 投影后: Q[2048, 1], K[1024, 1], V[1024, 1]
// 3. Reshape 后: Q[128, 16, 1], K[128, 8, 1], V[128, 8, 1]
// 4. RoPE 后: 形状不变，值已旋转
// 5. 注意力分数: [1, 1, 16]
// 6. 注意力输出: [128, 1, 16] -> [2048, 1]
// 7. FFN 输出: [1024, 1]
// 8. Layer 0 输出: [1024, 1]
```

**验证方法**:
```bash
# 检查 Layer 0 统计
grep "\[Kylin Debug\] Layer 0" /tmp/kylin_test.log

# 检查注意力计算日志
grep "\[Attention L0\]" /tmp/kylin_test.log
```

**预期结果**:
- Layer 0 输出 shape: `[1024, 1]`
- Min/Max: 合理范围（可能比 embedding 大）
- Mean: 可能不为 0，但不应过大
- NaN/Inf: 0
- 与 llama_cpp 的 Layer 0 输出对比（如果可获取）

**状态**: 🔄 进行中（已添加调试日志）

---

### Stage 4: 注意力计算详细验证 🔄

**目标**: 验证注意力计算的每个步骤

**子阶段**:

#### Stage 4.1: QKV 投影验证
- [ ] Q 投影输出正确
- [ ] K 投影输出正确
- [ ] V 投影输出正确
- [ ] 维度正确

#### Stage 4.2: Q/K 归一化验证
- [ ] Q norm 正确应用
- [ ] K norm 正确应用
- [ ] 广播正确

#### Stage 4.3: RoPE 验证
- [ ] RoPE 参数正确 (freq_base=1000000)
- [ ] 位置编码正确应用
- [ ] Q 和 K 都正确旋转

#### Stage 4.4: KV Cache 验证
- [ ] KV Cache 正确写入
- [ ] KV Cache 正确读取
- [ ] 增量推理时 cache 正确更新

#### Stage 4.5: GQA 扩展验证
- [ ] GQA 扩展正确 (8 KV heads -> 16 Q heads)
- [ ] Head 映射正确

#### Stage 4.6: 注意力分数计算验证
- [ ] Q@K^T 计算正确
- [ ] 缩放正确 (1/sqrt(head_dim))
- [ ] 因果 mask 正确应用
- [ ] Softmax 正确应用

#### Stage 4.7: 注意力输出验证
- [ ] Attention@V 计算正确
- [ ] 输出投影正确
- [ ] 维度正确

**验证方法**:
```bash
# 检查每个子阶段的日志
grep "\[Attention L0\]" /tmp/kylin_test.log | grep -E "QKV|norm|RoPE|GQA|scores"
```

**状态**: 🔄 待实现详细日志

---

### Stage 5: FFN 计算验证 🔄

**目标**: 验证 FFN (Feed-Forward Network) 的正确性

**测试内容**:
- [ ] FFN 归一化正确
- [ ] Gate 投影正确
- [ ] Up 投影正确
- [ ] SiLU 激活正确
- [ ] Down 投影正确
- [ ] SwiGLU 组合正确

**验证方法**:
```bash
# 检查 FFN 日志（需要添加）
grep "\[FFN L0\]" /tmp/kylin_test.log
```

**状态**: 🔄 待实现详细日志

---

### Stage 6: 多层累积验证 🔄

**目标**: 验证多层 Transformer 的累积效果

**测试内容**:
- [ ] 每层的输出统计
- [ ] 残差连接正确
- [ ] 数值稳定性（无爆炸/消失）
- [ ] 与 llama_cpp 的中间层输出对比

**测试用例**:
```cpp
// 测试用例: 多 token prompt
inputIds = [9707, 11, 1234]  // "Hello world"
// 验证每一层的输出
```

**验证方法**:
```bash
# 检查每层输出（需要添加）
grep "\[Layer.*output\]" /tmp/kylin_test.log
```

**状态**: 🔄 待实现

---

### Stage 7: 最终归一化和 LM Head 验证 🔄

**目标**: 验证最终输出层的正确性

**测试内容**:
- [ ] 最终 RMSNorm 正确
- [ ] LM Head 投影正确
- [ ] Logits 形状正确 `[seq_len, vocab]`
- [ ] Logits 数值范围合理
- [ ] Top-k tokens 合理

**验证方法**:
```bash
# 检查 logits 统计
grep "Logits stats" /tmp/kylin_test.log
grep "top-5" /tmp/kylin_test.log
```

**状态**: 🔄 部分完成（已有 logits 统计）

---

### Stage 8: 增量推理验证 🔄

**目标**: 验证增量推理（KV Cache 复用）的正确性

**测试内容**:
- [ ] 首次推理正确
- [ ] 增量推理时 KV Cache 正确复用
- [ ] 位置编码正确更新
- [ ] 输出一致性

**测试用例**:
```cpp
// 测试用例: 增量推理
// Step 1: inputIds = [9707], startPos = 0
// Step 2: inputIds = [11], startPos = 1
// Step 3: inputIds = [1234], startPos = 2
// 验证每一步的输出
```

**验证方法**:
```bash
# 检查 KV Cache 状态
grep "KV cache\|Flushing" /tmp/kylin_test.log
```

**状态**: 🔄 待实现

---

### Stage 9: 端到端输出对比 🔄

**目标**: 对比 Kylin 和 llama_cpp 的最终输出

**测试内容**:
- [ ] 相同 prompt 的输出对比
- [ ] Logits 分布对比
- [ ] Top-k tokens 对比
- [ ] 生成文本对比

**验证方法**:
```bash
# 使用对比工具
./tools/full_comparison.sh "Hi" 5 0.0
```

**状态**: 🔄 进行中

---

## 测试工具

### 1. 分阶段测试框架

创建 `tools/kylin_stage_test.cpp`，支持：
- 按阶段执行测试
- 自动对比 llama_cpp
- 生成详细报告

### 2. 中间结果提取工具

创建 `tools/extract_intermediate_results.py`，支持：
- 从日志中提取各阶段的统计信息
- 生成对比报告
- 可视化数值分布

### 3. 自动化测试脚本

创建 `tools/run_kylin_stages.sh`，支持：
- 自动运行所有阶段
- 生成阶段报告
- 失败时停止并报告

---

## 实施计划

### Phase 1: 基础验证（已完成）✅

- [x] Stage 0: 基础环境验证
- [x] Stage 1: 模型加载验证
- [x] 添加基础调试日志

### Phase 2: 单层验证（进行中）🔄

- [x] Stage 2: Token Embedding 验证（已添加日志）
- [x] Stage 3: 第一层 Transformer Block 验证（已添加日志）
- [ ] Stage 4: 注意力计算详细验证（需要添加详细日志）
- [ ] Stage 5: FFN 计算验证（需要添加日志）

### Phase 3: 多层验证（待开始）⏳

- [ ] Stage 6: 多层累积验证
- [ ] Stage 7: 最终归一化和 LM Head 验证
- [ ] 添加每层的中间结果日志

### Phase 4: 增量推理验证（待开始）⏳

- [ ] Stage 8: 增量推理验证
- [ ] KV Cache 状态验证

### Phase 5: 端到端验证（进行中）🔄

- [x] Stage 9: 端到端输出对比
- [ ] 性能基准测试

---

## 调试策略

### 1. 自底向上验证

从最基础的组件开始，逐步向上验证：
1. 先验证 Embedding
2. 再验证单层 Attention
3. 然后验证完整 Layer
4. 最后验证多层累积

### 2. 对比验证

每个阶段都与 llama_cpp 进行对比：
- 如果可能，提取 llama_cpp 的中间结果
- 对比数值分布
- 对比输出形状

### 3. 数值稳定性检查

每个阶段都检查：
- NaN/Inf 计数
- 数值范围
- 均值是否合理

### 4. 逐步启用功能

- 先禁用可能有问题的功能（如 Q/K norm）
- 验证基础功能正确后，再逐步启用
- 每次只启用一个功能，验证后再继续

---

## 成功标准

### 每个阶段的成功标准

1. **无错误**: 无崩溃、无异常、无断言失败
2. **数值合理**: 无 NaN/Inf，数值范围合理
3. **形状正确**: 所有张量形状符合预期
4. **对比一致**: 与 llama_cpp 的输出一致（或非常接近）

### 整体成功标准

1. **所有阶段通过**: Stage 0-9 全部通过
2. **输出正确**: 生成的文本与 llama_cpp 一致
3. **性能可接受**: 推理速度在可接受范围内

---

## 下一步行动

### 立即行动

1. **完善 Stage 2-3 的验证**
   - 添加更详细的 embedding 对比
   - 添加 Layer 0 输出的详细分析

2. **实现 Stage 4 的详细验证**
   - 添加注意力计算的每个步骤的日志
   - 对比每个步骤的输出

3. **创建测试框架**
   - 实现 `kylin_stage_test.cpp`
   - 实现自动化测试脚本

### 短期目标（1-2天）

1. 完成 Stage 2-5 的详细验证
2. 实现中间结果提取工具
3. 生成第一份详细对比报告

### 中期目标（3-5天）

1. 完成所有阶段的验证
2. 定位并修复所有问题
3. 实现性能基准测试

---

**文档版本**: v1.0  
**创建时间**: 2026-01-23  
**最后更新**: 2026-01-23
