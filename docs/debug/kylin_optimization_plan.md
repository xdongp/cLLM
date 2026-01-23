# Kylin Backend 优化调试方案

## 1. 问题概述

### 1.1 当前症状
- Kylin backend 输出乱码文本
- logits 与 llama.cpp 差异显著：
  - **Kylin**: mean=1.36, range=[-12.06, 15.89]
  - **llama.cpp**: mean=-2.43, range=[-20.73, 11.24]
  - **Mean 偏移**: ~3.8

### 1.2 已完成的修复
| 修复项 | 状态 | 说明 |
|--------|------|------|
| RoPE type 参数 | ✅ | 从 0 改为 2 (NEOX) for Qwen3 |
| 配置传递 | ✅ | ropeType 正确传递到 AttentionConfig |

### 1.3 已验证正确的组件
| 组件 | 状态 | 备注 |
|------|------|------|
| 权重加载 | ✅ | 类型和维度正确 |
| Q/K Norm 权重 | ✅ | 值符合预期 (max=96.5) |
| RMS Norm 计算 | ✅ | std≈1.0 |
| Attention 输出 | ✅ | Layer 0 范围正常 [-0.62, 0.79] |
| FFN 实现 | ✅ | SwiGLU 顺序正确 |
| RoPE type | ✅ | 正确设置为 NEOX (2) |

### 1.4 发现的关键问题 ⚠️

**误差在最后一层爆炸**：

| Layer | std | std ratio | 状态 |
|-------|-----|-----------|------|
| 0 | 0.29 | 1.00 | ✅ 正常 |
| 1 | 0.64 | 2.22 | ⚠️ 首次增长 |
| 2-24 | 逐渐增长 | ~1.2 | ⚠️ 累积 |
| 25 | 26.0 | 1.22 | ⚠️ 高 |
| 26 | 30.4 | 1.17 | ⚠️ 高 |
| **27** | **109.9** | **3.61** | ❌ **爆炸** |

**结论**：误差在 Layer 27（最后一层）突然增长 3.6x，从 std=30 跳到 std=110

### 1.5 待调查的问题
1. **Layer 27 数值爆炸** - 最后一层的 std 突然增长 3.6x
2. **累积误差** - 从 Layer 1 开始每层 std ratio > 1.0
3. **可能的原因**:
   - 残差连接的累积效应
   - 某些权重的量化精度问题
   - GGML 操作的数值稳定性

---

## 2. 调试策略

### 2.1 分层对比调试法

将推理过程分为以下阶段，逐层对比 Kylin 与 llama.cpp 的输出：

```
Stage 1: Embedding Lookup
    ↓
Stage 2: Input RMS Norm (attn_norm)
    ↓
Stage 3: QKV Projection
    ↓
Stage 4: Q/K Reshape + Norm
    ↓
Stage 5: RoPE Position Encoding
    ↓
Stage 6: Attention Computation (Q@K, softmax, @V)
    ↓
Stage 7: Output Projection (wo)
    ↓
Stage 8: Residual Connection
    ↓
Stage 9: FFN Norm + FFN
    ↓
Stage 10: Final Norm + LM Head
```

### 2.2 调试工具设计

创建 `debug_kylin_vs_llamacpp.cpp`，实现：

1. **同步推理**: 同时运行 Kylin 和 llama.cpp，使用相同输入
2. **中间值提取**: 提取每个 stage 的中间张量
3. **统计对比**: 计算 mean, std, max, min 差异
4. **元素级对比**: 找出差异最大的位置

---

## 3. 已识别的实现差异

### 3.1 RoPE 参数 ✅ 已修复

**llama.cpp (qwen3.cpp:49-62)**:
```cpp
Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
    ext_factor, attn_factor, beta_fast, beta_slow);
```

**Kylin (修复后)**:
```cpp
const int ropeMode = config_.ropeType;  // 从配置读取，Qwen3 为 2 (NEOX)
q = ggml_rope_ext(ctx, q, positions, nullptr,
    nRot, ropeMode, nCtxOrig, freqBase, freqScale,
    extFactor, attnFactor, betaFast, betaSlow);
```

**状态**: ✅ 已修复 - ropeType 正确设置为 NEOX (2) for Qwen3

### 3.2 Attention Scale

**llama.cpp (qwen3.cpp:70)**:
```cpp
cur = build_attn(..., 1.0f/sqrtf(float(n_embd_head)), il);
```

**Kylin (attention_graph.cpp:252)**:
```cpp
const float scale = 1.0f / std::sqrt(static_cast<float>(actualHeadDim));
```

**状态**: ✅ 一致

### 3.3 GQA 扩展方式

**llama.cpp**: 使用 `ggml_repeat` 内部处理 GQA

**Kylin**: 手动实现 GQA 扩展（可能有差异）

---

## 4. 修复计划

### Phase 1: RoPE 参数修复 ✅ 已完成

**修改内容**:
1. `GGUFModelConfig` 添加 `ropeType` 字段
2. `parseArchConfig()` 根据架构推断 rope_type
3. `AttentionConfig` 添加 `ropeType` 传递
4. `applyRoPE()` 使用 `config_.ropeType`

**验证**: RoPE type=2 已正确应用于 Qwen3

---

### Phase 2: Layer 27 数值爆炸调查 (当前优先)

#### 4.2.1 问题描述

最后一层 (Layer 27) 的 std 从 30 突然增长到 110 (ratio=3.61)，远高于其他层的 ~1.2。

#### 4.2.2 可能原因

1. **FFN 权重问题** - 最后一层的 ffn_down 权重可能有异常
2. **累积误差放大** - 大值输入 + 权重 = 更大输出
3. **数值精度** - Q4_K/Q6_K 量化在大值时精度下降

#### 4.2.3 调试步骤

```cpp
// 在 buildLayerGraph 中添加最后一层的详细调试
if (layerIdx == 27) {
    CLLM_INFO("[Layer 27 Debug] Input: min=%.2f, max=%.2f, std=%.2f", ...);
    CLLM_INFO("[Layer 27 Debug] Attn output: min=%.2f, max=%.2f", ...);
    CLLM_INFO("[Layer 27 Debug] FFN input: min=%.2f, max=%.2f", ...);
    CLLM_INFO("[Layer 27 Debug] FFN output: min=%.2f, max=%.2f", ...);
}
```

#### 4.2.4 调试结果

**Layer 27 详细分析**：
| 阶段 | min | max | std |
|------|-----|-----|-----|
| Attn Input | -92 | 91 | 30.4 |
| Attn Output | -66 | 77 | 19.4 |
| FFN Input | -33 | 164 | 6.2 |
| **FFN Output** | **-1948** | **715** | **105** |

FFN 输出的 std 从 6.2 → 105（17x 放大），这是问题所在。

**已尝试的修复**：
- ✅ 使用 `ggml_swiglu_split` 替代分开的 silu + mul → 无效果

**可能的根本原因**：
1. 量化权重在大值输入时精度下降
2. 残差连接累积了 28 层的误差
3. 某些 GGML 操作在 CPU backend 上的数值稳定性问题

#### 4.2.5 下一步建议

1. **检查 ffn_down 权重** - 使用 GGML API 直接检查权重值范围
2. **对比 llama.cpp 中间值** - 添加 callback 提取 llama.cpp 的层输出
3. **尝试 F32 权重** - 用非量化模型测试以排除量化问题

---

### Phase 3: 累积误差控制

#### 4.3.1 问题描述

每层的 std ratio 平均为 1.2，28 层累积后 = 1.2^28 ≈ 1000x 放大

#### 4.3.2 误差累积的根本原因（基于研究调研）

根据最新研究和 llama.cpp 社区的讨论，误差累积有以下几个根本原因：

##### 1. Pre-LN 方差累积问题 (Variance Accumulation)

**问题描述**：Pre-Layer Normalization (Pre-LN) 虽然避免了梯度消失，但会导致隐藏状态的方差随深度指数增长。

**原理**：
```
# Pre-LN 结构
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# 问题：每次残差连接都会累积方差
Var(x_out) = Var(x_in) + Var(sublayer_output)
```

**参考论文**：
- "The Curse of Depth in Large Language Models" (arXiv:2502.05795)
- "Peri-LN: Revisiting Layer Normalization" (arXiv:2502.02732)

**影响**：深层的输出方差指数增长，导致 "massive activations"（巨大激活值）。

##### 2. 量化精度损失 (Quantization Precision Loss)

**Q4_K/Q6_K 结构**：
- 使用 256 值的 super-block
- 每个块有 1-2 个浮点缩放因子
- 量化误差 = Round(Scale × Original) / Scale - Original

**大值时的精度损失**：
- 缩放因子由块内最大值决定
- 大值范围导致更大的量化误差
- 当输入已经很大时（如 std=30），量化误差被放大

##### 3. FP16 累加器溢出

**llama.cpp 已知问题 (Issue #11920)**：
- 在 AArch64 fp16fml 配置下，`ggml_vec_dot_f16` 产生无穷值
- FP16 最大值 = 65,504，累加器可能溢出
- 导致输出降级为随机 token

##### 4. 浮点运算的非关联性

**问题**：`(a + b) + c ≠ a + (b + c)` 在有限精度下
- 不同硬件配置产生不同舍入模式
- 误差随深度累积并发散
- 研究显示可导致高达 9% 的准确率差异

#### 4.3.3 为什么 llama.cpp 没有这个问题？

可能的原因：

1. **计算图优化**：llama.cpp 使用更优化的计算图构建
2. **后端差异**：llama.cpp 的 backend 可能有数值稳定性优化
3. **内存布局**：GGML 的内存布局与 Kylin 的手动实现可能有差异
4. **算子融合**：llama.cpp 可能使用了融合算子（如 `ggml_swiglu_split`）避免中间精度损失

#### 4.3.4 解决方案

| 方案 | 难度 | 效果 | 说明 |
|------|------|------|------|
| **LayerCast** | 中 | 高 | 权重用 16-bit 存储，计算用 FP32 |
| **LayerNorm Scaling** | 低 | 中 | 按深度的平方根缩放归一化输出 |
| **Peri-LN** | 高 | 高 | 改变层归一化位置（需要重训练） |
| **使用 F32 权重** | 低 | 高 | 排除量化误差（内存翻倍） |
| **完全对齐 llama.cpp** | 中 | 高 | 确保每个操作与 llama.cpp 完全一致 |

#### 4.3.5 已应用的修复

1. **✅ Q@K 矩阵乘法 F32 精度** (2025-01-23)
   ```cpp
   // src/kylin/attention_graph.cpp
   ggml_tensor* scores = ggml_mul_mat(ctx, kExpanded, q);
   ggml_mul_mat_set_prec(scores, GGML_PREC_F32);  // 新增
   ```
   - 参考 llama.cpp: "this op tends to require high floating point range"
   - **结果**：暂未改善，可能需要其他修复

#### 4.3.6 推荐的修复路径

1. **短期（立即）**：
   - ✅ 添加 Q@K F32 精度（已完成）
   - 使用 F32 模型验证问题是否来自量化
   - 逐算子对比 Kylin vs llama.cpp 的输出

2. **中期（1-2 周）**：
   - 实现 LayerCast：计算时转为 FP32
   - 添加 LayerNorm Scaling 因子
   - 检查残差连接的累积方式

3. **长期（如需要）**：
   - 完全重构以对齐 llama.cpp 的计算图
   - 考虑直接使用 llama.cpp 的后端

---

## 8. 调研总结：误差累积的根本原因

### 8.1 理论背景

根据最新研究（2025年），深层 Transformer 的数值不稳定是一个已知问题：

| 来源 | 发现 |
|------|------|
| "The Curse of Depth in Large Language Models" | Pre-LN 的输出方差随深度指数增长 |
| "Peri-LN: Revisiting Layer Normalization" | 残差连接导致 "massive activations" |
| llama.cpp Issue #11920 | FP16 累加器溢出导致输出降级 |
| "Give Me FP32 or Give Me Death?" | 浮点精度差异可导致 9% 准确率变化 |

### 8.2 Kylin 特定问题

| 问题 | llama.cpp 的处理 | Kylin 状态 |
|------|-----------------|-----------|
| Q@K 精度 | `ggml_mul_mat_set_prec(kq, GGML_PREC_F32)` | ✅ 已修复 |
| Flash Attention 精度 | `ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32)` | ⚠️ 未使用 Flash Attention |
| RoPE type | 根据架构设置 | ✅ 已修复 |
| SwiGLU 实现 | `ggml_swiglu_split` | ✅ 已对齐 |

### 8.3 未解决的问题

1. **Layer 27 FFN 输出爆炸**：std 从 6.2 → 105（17x 放大）
2. **累积误差**：每层 std ratio ~1.2，28 层累积 ~1000x
3. **可能的根本原因**：
   - Kylin 的计算图与 llama.cpp 有细微差异
   - 某些 GGML 操作的参数设置不同
   - 内存布局或张量操作顺序差异

### 8.4 下一步建议

1. **使用 llama.cpp 的 callback 机制**提取每层中间值，进行逐元素对比
2. **尝试 F32 权重模型**排除量化问题
3. **检查张量形状和内存布局**是否与 llama.cpp 完全一致
4. **考虑直接复用 llama.cpp 的图构建代码**而非独立实现

### Phase 2: 创建对比调试工具

#### 4.2.1 工具设计

```cpp
// tools/debug_layer_comparison.cpp
class LayerComparator {
public:
    // 比较单层输出
    void compareLayer(const char* name, 
                      const float* kylin, 
                      const float* llamacpp, 
                      size_t size);
    
    // 生成报告
    void generateReport(const std::string& outputPath);
    
private:
    struct LayerStats {
        std::string name;
        float meanDiff;
        float maxDiff;
        float stdDiff;
        size_t maxDiffIdx;
    };
    std::vector<LayerStats> stats_;
};
```

#### 4.2.2 llama.cpp 钩子

使用 llama.cpp 的 callback 机制提取中间值：

```cpp
// 在 llama_decode 时设置 callback
llama_set_causal_attn(ctx, true);
// 使用 cb(cur, "layer_name", il) 提取值
```

### Phase 3: GQA 实现验证

#### 4.3.1 检查点

1. K/V 扩展前后的形状
2. 扩展后的数值一致性
3. permute 操作的正确性

#### 4.3.2 简化测试

创建一个简化的 GQA 测试，使用已知输入验证输出。

### Phase 4: 累积误差分析

#### 4.4.1 逐层误差追踪

```cpp
// 在每层后记录误差
for (size_t i = 0; i < config_.blockCount; ++i) {
    hidden_states = buildLayerGraph(...);
    
    // 记录该层后的统计信息
    LayerStats stats = computeLayerStats(hidden_states);
    accumulatedError_ += stats.meanAbsDiff;
    
    if (i == 0 || i == config_.blockCount - 1) {
        CLLM_INFO("Layer %zu: mean=%.6f, std=%.6f, accumulated=%.6f",
                  i, stats.mean, stats.std, accumulatedError_);
    }
}
```

#### 4.4.2 误差阈值检查

```cpp
// 如果某层误差突然增大，标记为可疑
if (layerError > previousError * 2.0f) {
    CLLM_WARN("Layer %zu: Error spike detected (%.4f -> %.4f)",
              i, previousError, layerError);
    suspiciousLayers_.push_back(i);
}
```

---

## 5. 验证测试计划

### 5.1 单元测试

| 测试名 | 描述 | 预期结果 |
|--------|------|----------|
| test_rope_type | 验证 RoPE 参数正确读取 | rope_type=2 (Qwen3) |
| test_embedding | 验证 embedding lookup | 与 llama.cpp 一致 (diff < 1e-5) |
| test_qk_norm | 验证 Q/K Norm | 与 llama.cpp 一致 (diff < 1e-4) |
| test_attention | 验证 attention 输出 | 与 llama.cpp 一致 (diff < 1e-3) |
| test_ffn | 验证 FFN 输出 | 与 llama.cpp 一致 (diff < 1e-3) |
| test_logits | 验证最终 logits | 与 llama.cpp 一致 (diff < 0.1) |

### 5.2 集成测试

```bash
# 运行完整的对比测试
./bin/debug_layer_comparison \
    --model ../model/Qwen/qwen3-0.6b-q4_k_m.gguf \
    --prompt "Hello" \
    --output comparison_report.json
```

### 5.3 端到端测试

```bash
# 使用相同 prompt 生成文本
curl -X POST http://localhost:8080/generate \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.0}'
```

---

## 6. 实施时间表

### Week 1: 基础调试
- [ ] Phase 1: RoPE 参数修复
- [ ] 创建对比调试工具框架

### Week 2: 深度分析
- [ ] Phase 2: 完成对比工具
- [ ] 运行全面对比，收集数据

### Week 3: 修复与验证
- [ ] Phase 3: 修复 GQA 实现（如有问题）
- [ ] Phase 4: 处理累积误差
- [ ] 运行所有验证测试

### Week 4: 优化与文档
- [ ] 性能优化
- [ ] 更新文档
- [ ] 代码审查

---

## 7. 附录

### A. 快速诊断命令

```bash
# 编译调试工具
cd build && make debug_kylin_layers -j4

# 运行调试
DYLD_LIBRARY_PATH=../third_party/llama.cpp/build/bin \
./bin/debug_kylin_layers ../model/Qwen/qwen3-0.6b-q4_k_m.gguf 2>&1 | \
grep -E "mean|diff|Layer"
```

### B. 关键代码位置

| 文件 | 功能 |
|------|------|
| `src/kylin/ggml_transformer.cpp` | 主模型实现 |
| `src/kylin/attention_graph.cpp` | Attention 计算图 |
| `src/kylin/gguf_loader.cpp` | GGUF 加载器 |
| `include/cllm/kylin/gguf_loader.h` | 配置结构定义 |

### C. 参考实现

- llama.cpp Qwen3: `third_party/llama.cpp/src/models/qwen3.cpp`
- llama.cpp Attention: `third_party/llama.cpp/src/llama-graph.cpp`
