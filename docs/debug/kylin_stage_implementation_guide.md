# Kylin 分阶段测试实施指南

## 概述

本文档提供每个阶段的具体实施步骤、测试用例和验证方法。

---

## Stage 0: 基础环境验证 ✅

### 实施状态
- ✅ 已完成

### 测试用例
```cpp
TEST(Stage0, BasicEnvironment) {
    ModelConfig config;
    config.vocabSize = 151936;
    // ... 其他配置
    
    KylinBackend backend(config, modelPath);
    ASSERT_TRUE(backend.initialize());
}
```

### 验证点
- [x] KylinBackend 构造成功
- [x] 模型路径检测正确
- [x] 设备后端配置正确
- [x] 初始化成功

---

## Stage 1: 模型加载验证 ✅

### 实施状态
- ✅ 已完成

### 测试用例
```cpp
TEST(Stage1, ModelLoading) {
    KylinBackend backend(config, modelPath);
    backend.initialize();
    
    const auto& loadedConfig = backend.getConfig();
    ASSERT_EQ(loadedConfig.vocabSize, 151936);
    ASSERT_EQ(loadedConfig.hiddenSize, 1024);
    ASSERT_EQ(loadedConfig.numLayers, 28);
    // ...
}
```

### 验证点
- [x] 配置加载正确
- [x] 权重映射完成
- [x] KV Cache 分配成功

---

## Stage 2: Token Embedding 验证 🔄

### 实施状态
- 🔄 已添加调试日志，需要完善验证

### 测试用例

#### 用例 2.1: 单个 Token
```cpp
TEST(Stage2, SingleTokenEmbedding) {
    KylinBackend backend(config, modelPath);
    backend.initialize();
    
    std::vector<int> inputIds = {9707};  // "Hello"
    auto output = backend.forward(inputIds);
    
    // 验证输出形状
    ASSERT_EQ(output.shape()[0], 1);
    ASSERT_EQ(output.shape()[1], 151936);
    
    // 验证 embedding 统计（从日志提取）
    // min, max, mean, nan, inf
}
```

#### 用例 2.2: 多个 Tokens
```cpp
TEST(Stage2, MultiTokenEmbedding) {
    std::vector<int> inputIds = {9707, 11, 1234};
    // 验证 embedding 输出形状 [1024, 3]
}
```

#### 用例 2.3: 与 llama_cpp 对比
```cpp
TEST(Stage2, CompareWithLlamaCpp) {
    // 运行 Kylin
    auto kylinOutput = kylinBackend.forward({9707});
    
    // 运行 llama_cpp
    auto llamaOutput = llamaCppBackend.forward({9707});
    
    // 对比 embedding 输出（需要从中间结果提取）
    // 对比前10个值
}
```

### 验证方法

**方法 1: 从日志提取**
```bash
# 运行测试
./build/bin/cllm_server --config config/config.yaml > /tmp/test.log 2>&1 &
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Hi", "max_tokens": 1, "temperature": 0.0}'

# 提取 embedding 统计
grep "\[Kylin Debug\] Embedding stats" /tmp/test.log
```

**方法 2: 添加回调接口**
```cpp
// 在 GGMLTransformerModel 中添加回调
class EmbeddingCallback {
public:
    virtual void onEmbedding(const float* data, size_t size) = 0;
};

// 在 forward() 中调用
if (embeddingCallback_) {
    embeddingCallback_->onEmbedding(
        static_cast<const float*>(debugEmbedding_->data),
        debugEmbedding_->ne[0] * debugEmbedding_->ne[1]
    );
}
```

### 预期结果
- Shape: `[1024, seq_len]`
- Min: 通常在 [-1, 1]
- Max: 通常在 [-1, 1]
- Mean: 接近 0
- NaN/Inf: 0
- 前10个值与 llama_cpp 一致（误差 < 1e-5）

---

## Stage 3: 第一层 Transformer Block 验证 🔄

### 实施状态
- 🔄 已添加调试日志，需要完善验证

### 测试用例

#### 用例 3.1: 单 Token 首次推理
```cpp
TEST(Stage3, Layer0SingleToken) {
    std::vector<int> inputIds = {9707};
    
    // 验证点:
    // 1. Attention 归一化输出
    // 2. QKV 投影输出
    // 3. Q/K 归一化输出
    // 4. RoPE 后输出
    // 5. 注意力输出
    // 6. FFN 输出
    // 7. Layer 0 最终输出
}
```

#### 用例 3.2: 多 Token 首次推理
```cpp
TEST(Stage3, Layer0MultiToken) {
    std::vector<int> inputIds = {9707, 11, 1234};
    // 验证每个步骤的输出形状和数值
}
```

### 验证方法

**添加中间节点保存**
```cpp
// 在 buildLayerGraph 中保存中间节点
class LayerDebugNodes {
public:
    ggml_tensor* attnNormOutput;
    ggml_tensor* qkvOutput;
    ggml_tensor* qNormOutput;
    ggml_tensor* kNormOutput;
    ggml_tensor* ropeQOutput;
    ggml_tensor* ropeKOutput;
    ggml_tensor* attentionOutput;
    ggml_tensor* ffnNormOutput;
    ggml_tensor* ffnOutput;
    ggml_tensor* layerOutput;
};

// 在 forward() 中打印统计
void printLayerStats(const LayerDebugNodes& nodes, size_t layerIdx) {
    if (nodes.attnNormOutput && nodes.attnNormOutput->data) {
        printTensorStats("AttnNorm", nodes.attnNormOutput, layerIdx);
    }
    // ... 其他节点
}
```

### 预期结果
- Layer 0 输出 shape: `[1024, seq_len]`
- 所有中间步骤无 NaN/Inf
- 数值范围合理
- 与 llama_cpp 的 Layer 0 输出对比

---

## Stage 4: 注意力计算详细验证 ⏳

### 实施状态
- ⏳ 待实现

### 子阶段测试

#### Stage 4.1: QKV 投影验证

**测试用例**:
```cpp
TEST(Stage4_1, QKVProjection) {
    // 输入: [1024, 1]
    // 验证:
    // - Q: [2048, 1] (16 heads * 128 head_dim)
    // - K: [1024, 1] (8 KV heads * 128 head_dim)
    // - V: [1024, 1]
    
    // 验证数值范围
    // 验证与 llama_cpp 的 Q/K/V 输出对比
}
```

**实施步骤**:
1. 在 `buildAttentionGraph` 中保存 Q/K/V 投影后的节点
2. 在 `forward()` 中打印统计信息
3. 对比 llama_cpp 的输出

#### Stage 4.2: Q/K 归一化验证

**测试用例**:
```cpp
TEST(Stage4_2, QKNormalization) {
    // 验证 Q/K norm 正确应用
    // 验证广播正确
    // 验证数值范围
}
```

**验证点**:
- Q norm 权重形状: `[128]`
- K norm 权重形状: `[128]`
- 归一化后数值范围合理
- 与 llama_cpp 对比

#### Stage 4.3: RoPE 验证

**测试用例**:
```cpp
TEST(Stage4_3, RoPE) {
    // 验证 RoPE 参数
    // - freq_base = 1000000
    // - n_rot = 128
    // - n_ctx_orig = 40960
    
    // 验证位置编码正确应用
    // 对比不同位置的 Q/K 值
}
```

**验证方法**:
```cpp
// 在 buildAttentionGraph 中
CLLM_DEBUG("[Attention L%zu] RoPE Q before: first 5 values: %.6f %.6f %.6f %.6f %.6f",
           layerIdx, qData[0], qData[1], qData[2], qData[3], qData[4]);

// 应用 RoPE
q = ggml_rope_ext(...);

// 验证 RoPE 后
CLLM_DEBUG("[Attention L%zu] RoPE Q after: first 5 values: %.6f %.6f %.6f %.6f %.6f",
           layerIdx, qData[0], qData[1], qData[2], qData[3], qData[4]);
```

#### Stage 4.4: KV Cache 验证

**测试用例**:
```cpp
TEST(Stage4_4, KVCache) {
    // 首次推理
    forward({9707});  // startPos=0
    
    // 验证 KV Cache 写入
    // - K cache 形状: [128, 2048, 8]
    // - V cache 形状: [128, 2048, 8]
    // - 数据正确写入位置 0
    
    // 增量推理
    forward({11});  // startPos=1
    
    // 验证 KV Cache 读取
    // - 从 cache 读取位置 0 的数据
    // - 新数据写入位置 1
    // - totalLen = 2
}
```

**验证方法**:
```cpp
// 在 flushKVCache() 后验证
bool verifyKVCache(size_t layerIdx, size_t expectedLen) {
    auto kCache = kCaches_[layerIdx];
    auto vCache = vCaches_[layerIdx];
    
    // 验证形状
    // 验证数据完整性（无 NaN/Inf）
    // 验证数据范围
    return true;
}
```

#### Stage 4.5: GQA 扩展验证

**测试用例**:
```cpp
TEST(Stage4_5, GQAExpansion) {
    // 验证 GQA 扩展
    // - 输入: K[128, total_len, 8], V[128, total_len, 8]
    // - 输出: K[128, total_len, 16], V[128, total_len, 16]
    // - 验证 head 映射: Q head i -> KV head i/2
    
    // 验证扩展后的数据正确
    // 验证 head 顺序正确
}
```

**验证方法**:
```cpp
// 在 buildAttentionGraph 中
if (nKVHeads < nHeads) {
    CLLM_DEBUG("[Attention L%zu] GQA: Before expansion - K shape: [%lld, %lld, %lld]",
               layerIdx, kFull->ne[0], kFull->ne[1], kFull->ne[2]);
    
    // GQA 扩展
    kExpanded = ...;
    
    CLLM_DEBUG("[Attention L%zu] GQA: After expansion - K shape: [%lld, %lld, %lld]",
               layerIdx, kExpanded->ne[0], kExpanded->ne[1], kExpanded->ne[2]);
    
    // 验证 head 映射
    // head 0,1 -> KV head 0
    // head 2,3 -> KV head 1
    // ...
}
```

#### Stage 4.6: 注意力分数计算验证

**测试用例**:
```cpp
TEST(Stage4_6, AttentionScores) {
    // 验证 Q@K^T 计算
    // - Q: [128, seq_len, 16]
    // - K: [128, total_len, 16]
    // - Scores: [total_len, seq_len, 16]
    
    // 验证缩放
    // - scale = 1/sqrt(128) ≈ 0.0884
    
    // 验证因果 mask
    // - 位置 i 不能看到位置 j (j > i + startPos)
    
    // 验证 softmax
    // - 每行的和 = 1
    // - 所有值 >= 0
}
```

**验证方法**:
```cpp
// 在 buildAttentionGraph 中
ggml_tensor* scores = ggml_mul_mat(ctx, kExpanded, q);
CLLM_DEBUG("[Attention L%zu] Scores shape: [%lld, %lld, %lld]",
           layerIdx, scores->ne[0], scores->ne[1], scores->ne[2]);

scores = ggml_scale(ctx, scores, scale);
// 打印缩放后的统计

scores = ggml_diag_mask_inf(ctx, scores, startPos);
// 验证 mask 正确应用

ggml_tensor* attnWeights = ggml_soft_max(ctx, scores);
// 验证 softmax 后每行和 = 1
```

#### Stage 4.7: 注意力输出验证

**测试用例**:
```cpp
TEST(Stage4_7, AttentionOutput) {
    // 验证 Attention@V
    // - attnWeights: [total_len, seq_len, 16]
    // - V: [128, total_len, 16]
    // - Output: [128, seq_len, 16]
    
    // 验证输出投影
    // - Output: [2048, seq_len]
}
```

---

## Stage 5: FFN 计算验证 ⏳

### 实施状态
- ⏳ 待实现

### 测试用例

#### 用例 5.1: FFN 完整流程
```cpp
TEST(Stage5, FFNComputation) {
    // 验证 FFN 归一化
    // 验证 Gate/Up 投影
    // 验证 SiLU 激活
    // 验证 Down 投影
    // 验证 SwiGLU 组合
}
```

### 验证方法

**添加 FFN 调试节点**
```cpp
// 在 buildFFNGraph 中
ggml_tensor* gate = ggml_mul_mat(ctx, layer.wGate, input);
ggml_tensor* up = ggml_mul_mat(ctx, layer.wUp, input);

// 保存中间节点
debugFFNGate_ = gate;
debugFFNUp_ = up;

gate = ggml_silu(ctx, gate);
ggml_tensor* hidden = ggml_mul(ctx, gate, up);
debugFFNHidden_ = hidden;

ggml_tensor* output = ggml_mul_mat(ctx, layer.wDown, hidden);
debugFFNOutput_ = output;
```

---

## Stage 6: 多层累积验证 ⏳

### 实施状态
- ⏳ 待实现

### 测试用例

#### 用例 6.1: 逐层输出验证
```cpp
TEST(Stage6, MultiLayerOutput) {
    // 验证每一层的输出
    // 检查数值稳定性
    // 检查残差连接
}
```

### 验证方法

**保存每层输出**
```cpp
// 在 buildForwardGraph 中
std::vector<ggml_tensor*> layerOutputs;
for (size_t i = 0; i < config_.blockCount; ++i) {
    hidden_states = buildLayerGraph(...);
    layerOutputs.push_back(hidden_states);
    
    // 打印每层统计
    if (layerOutputs[i] && layerOutputs[i]->data) {
        printTensorStats("Layer " + std::to_string(i), 
                        layerOutputs[i], i);
    }
}
```

---

## Stage 7: 最终输出验证 🔄

### 实施状态
- 🔄 部分完成（已有 logits 统计）

### 测试用例

#### 用例 7.1: 最终归一化
```cpp
TEST(Stage7, FinalNormalization) {
    // 验证最终 RMSNorm
    // 验证 outputNorm 权重应用
}
```

#### 用例 7.2: LM Head
```cpp
TEST(Stage7, LMHead) {
    // 验证 LM Head 投影
    // 验证 logits 形状 [seq_len, vocab]
    // 验证 logits 数值范围
    // 验证 top-k tokens
}
```

---

## Stage 8: 增量推理验证 ✅

### 实施状态
- ✅ 已完善（2026-01-23）

### 新增接口

#### KV Cache 验证接口
```cpp
// 在 GGMLTransformerModel 中新增

struct KVCacheStats {
    size_t layerIdx;          // 层索引
    size_t headDim;           // head 维度
    size_t maxSeq;            // 最大序列长度
    size_t nKVHeads;          // KV head 数量
    size_t currentLen;        // 当前有效长度
    TensorStats kStats;       // K cache 统计
    TensorStats vStats;       // V cache 统计
    bool isValid;             // 是否有效（无 NaN/Inf）
};

// 获取指定层的 KV Cache 统计
KVCacheStats getKVCacheStats(size_t layerIdx) const;

// 获取所有层的 KV Cache 统计
std::vector<KVCacheStats> getAllKVCacheStats() const;

// 验证 KV Cache 数据完整性
bool validateKVCacheIntegrity(size_t expectedLen) const;

// 获取指定位置的 KV 数据
bool getKVAtPosition(size_t layerIdx, size_t position, 
                     std::vector<float>& kData, std::vector<float>& vData) const;
```

### 测试用例

#### 用例 8.1: 首次推理
```cpp
TEST(Stage8, FirstInference) {
    model.clearKVCache();
    std::vector<int32_t> firstToken = {9707};  // "Hello"
    
    auto firstLogits = model.forward(firstToken);
    
    // 验证 logits 形状
    ASSERT_EQ(firstLogits.size(), vocabSize);
    
    // 验证 KV Cache 长度
    size_t kvCacheLen = model.getKVCacheLength();
    ASSERT_EQ(kvCacheLen, 1);
    
    // 验证 KV Cache 数据完整性
    ASSERT_TRUE(model.validateKVCacheIntegrity(kvCacheLen));
    
    // 验证 Layer 0 KV Cache 统计
    auto layer0Stats = model.getKVCacheStats(0);
    ASSERT_TRUE(layer0Stats.isValid);
    ASSERT_EQ(layer0Stats.kStats.nanCount, 0);
    ASSERT_EQ(layer0Stats.kStats.infCount, 0);
}
```

#### 用例 8.2: 增量推理
```cpp
TEST(Stage8, IncrementalInference) {
    model.clearKVCache();
    std::vector<int32_t> tokens = {9707, 11, 1234};
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto tokenLogits = model.forwardOneToken(tokens[i], i);
        
        // 验证 logits 形状
        ASSERT_EQ(tokenLogits.size(), vocabSize);
        
        // 验证 KV Cache 长度递增
        ASSERT_EQ(model.getKVCacheLength(), i + 1);
        
        // 验证 KV Cache 数据完整性
        ASSERT_TRUE(model.validateKVCacheIntegrity(i + 1));
    }
}
```

#### 用例 8.3: 批量推理 vs 增量推理一致性
```cpp
TEST(Stage8, BatchVsIncrementalConsistency) {
    std::vector<int32_t> tokens = {9707, 11, 1234};
    
    // 增量推理
    model.clearKVCache();
    std::vector<std::vector<float>> incrementalLogits;
    for (size_t i = 0; i < tokens.size(); ++i) {
        incrementalLogits.push_back(model.forwardOneToken(tokens[i], i));
    }
    
    // 批量推理
    model.clearKVCache();
    auto batchLogits = model.forward(tokens);
    
    // 对比最后一个位置的 logits
    size_t lastPos = tokens.size() - 1;
    std::vector<float> batchLastLogits(
        batchLogits.begin() + lastPos * vocabSize,
        batchLogits.begin() + (lastPos + 1) * vocabSize
    );
    
    // 验证一致性（容差 1e-2）
    float maxDiff = 0.0f;
    for (size_t i = 0; i < vocabSize; ++i) {
        float diff = std::abs(batchLastLogits[i] - incrementalLogits.back()[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    ASSERT_LT(maxDiff, 1e-2f);
}
```

#### 用例 8.4: 中间步骤一致性
```cpp
TEST(Stage8, IntermediateStepConsistency) {
    std::vector<int32_t> tokens = {9707, 11, 1234};
    
    // 先做增量推理保存结果
    model.clearKVCache();
    std::vector<std::vector<float>> incrementalLogits;
    for (size_t i = 0; i < tokens.size(); ++i) {
        incrementalLogits.push_back(model.forwardOneToken(tokens[i], i));
    }
    
    // 对每个中间步骤验证
    for (size_t step = 1; step < tokens.size(); ++step) {
        model.clearKVCache();
        std::vector<int32_t> partialTokens(tokens.begin(), tokens.begin() + step + 1);
        auto partialBatchLogits = model.forward(partialTokens);
        
        // 提取最后一个位置
        std::vector<float> partialBatchLast(
            partialBatchLogits.begin() + step * vocabSize,
            partialBatchLogits.begin() + (step + 1) * vocabSize
        );
        
        // 验证一致性
        float maxDiff = 0.0f;
        for (size_t i = 0; i < vocabSize; ++i) {
            maxDiff = std::max(maxDiff, std::abs(partialBatchLast[i] - incrementalLogits[step][i]));
        }
        ASSERT_LT(maxDiff, 1e-2f);
    }
}
```

#### 用例 8.5: KV Cache 位置数据验证
```cpp
TEST(Stage8, KVCachePositionDataConsistency) {
    // 首次推理单个token
    model.clearKVCache();
    model.forward({9707});
    
    std::vector<float> firstKData, firstVData;
    model.getKVAtPosition(0, 0, firstKData, firstVData);
    
    // 增量推理多个token
    model.clearKVCache();
    model.forwardOneToken(9707, 0);
    model.forwardOneToken(11, 1);
    model.forwardOneToken(1234, 2);
    
    std::vector<float> incrKData, incrVData;
    model.getKVAtPosition(0, 0, incrKData, incrVData);
    
    // 验证位置0的KV数据一致
    float maxDiff = 0.0f;
    for (size_t i = 0; i < firstKData.size(); ++i) {
        maxDiff = std::max(maxDiff, std::abs(firstKData[i] - incrKData[i]));
        maxDiff = std::max(maxDiff, std::abs(firstVData[i] - incrVData[i]));
    }
    ASSERT_LT(maxDiff, 1e-5f);
}
```

#### 用例 8.6: 所有层 KV Cache 验证
```cpp
TEST(Stage8, AllLayersKVCacheValidation) {
    model.clearKVCache();
    model.forward({9707, 11, 1234});
    
    auto allLayerStats = model.getAllKVCacheStats();
    
    for (const auto& stats : allLayerStats) {
        ASSERT_TRUE(stats.isValid) << "Layer " << stats.layerIdx << " KV cache is invalid";
        ASSERT_EQ(stats.kStats.nanCount, 0);
        ASSERT_EQ(stats.kStats.infCount, 0);
        ASSERT_EQ(stats.vStats.nanCount, 0);
        ASSERT_EQ(stats.vStats.infCount, 0);
    }
}
```

### 验证点
- [x] 首次推理 KV Cache 写入位置 0
- [x] 增量推理 KV Cache 长度递增
- [x] KV Cache 数据完整性（无 NaN/Inf）
- [x] 批量推理与增量推理输出一致（容差 < 1e-2）
- [x] 中间步骤输出一致性
- [x] KV Cache 位置数据一致性
- [x] 所有层 KV Cache 有效性

### 预期结果
- 每次推理后 KV Cache 长度正确递增
- 所有 KV Cache 数据无 NaN/Inf
- 批量推理和增量推理的 logits 差异 < 1e-2
- 相同 token 在相同位置的 KV 数据差异 < 1e-5

---

## Stage 9: 端到端对比 🔄

### 实施状态
- 🔄 进行中

### 测试用例

#### 用例 9.1: 输出文本对比
```cpp
TEST(Stage9, OutputComparison) {
    // 运行 Kylin
    auto kylinText = generate("Hi", maxTokens=5, temp=0.0);
    
    // 运行 llama_cpp
    auto llamaText = generateLlamaCpp("Hi", maxTokens=5, temp=0.0);
    
    // 对比输出
    ASSERT_EQ(kylinText, llamaText);
}
```

#### 用例 9.2: Logits 对比
```cpp
TEST(Stage9, LogitsComparison) {
    // 对比 logits 分布
    // 对比 top-k tokens
    // 对比数值差异
}
```

---

## 实施优先级

### 高优先级（立即实施）

1. **完善 Stage 2-3 的验证**
   - 添加 embedding 和 Layer 0 的详细对比
   - 实现与 llama_cpp 的自动对比

2. **实现 Stage 4 的详细验证**
   - 添加注意力计算的每个步骤的日志
   - 实现中间节点的统计打印

3. **创建测试框架**
   - 完善 `kylin_stage_test.cpp`
   - 实现自动化测试脚本

### 中优先级（1-2天内）

4. **实现 Stage 5-6 的验证**
   - 添加 FFN 的详细日志
   - 实现多层输出的验证

5. ~~**实现 Stage 8 的验证**~~（已完成 2026-01-23）
   - ✅ 添加增量推理的测试用例
   - ✅ 验证 KV Cache 的正确性
   - ✅ 添加 KV Cache 统计接口
   - ✅ 实现批量推理 vs 增量推理一致性验证

### 低优先级（3-5天内）

6. **性能基准测试**
   - 记录每个阶段的性能
   - 对比 llama_cpp 的性能

7. **自动化报告生成**
   - 生成详细的测试报告
   - 可视化数值分布

---

## 工具和脚本

### 1. 分阶段测试程序

**文件**: `tools/kylin_stage_test.cpp`

**功能**:
- 按阶段执行测试
- 自动生成报告
- 支持与 llama_cpp 对比

**使用方法**:
```bash
./build/tools/kylin_stage_test <model_path> [prompt] [max_tokens] [temperature]
```

### 2. 自动化测试脚本

**文件**: `tools/run_kylin_stages.sh`

**功能**:
- 自动运行所有阶段
- 生成阶段报告
- 失败时停止并报告

**使用方法**:
```bash
./tools/run_kylin_stages.sh [model_path] [prompt] [max_tokens] [temperature]
```

### 3. 日志分析工具

**文件**: `tools/analyze_kylin_stages.py`

**功能**:
- 从日志中提取各阶段的统计信息
- 生成对比报告
- 可视化数值分布

**使用方法**:
```bash
python3 tools/analyze_kylin_stages.py /tmp/kylin_test.log
```

---

## 成功标准

### 每个阶段的成功标准

1. **无错误**: 无崩溃、无异常、无断言失败
2. **数值合理**: 无 NaN/Inf，数值范围合理
3. **形状正确**: 所有张量形状符合预期
4. **对比一致**: 与 llama_cpp 的输出一致（误差 < 1e-3）

### 整体成功标准

1. **所有阶段通过**: Stage 0-9 全部通过
2. **输出正确**: 生成的文本与 llama_cpp 一致
3. **性能可接受**: 推理速度在可接受范围内（> 50 tokens/sec）

---

**文档版本**: v1.1  
**创建时间**: 2026-01-23  
**最后更新**: 2026-01-23

### 更新历史
- v1.1 (2026-01-23): 完善 Stage 8 增量推理验证，添加 KV Cache 详细验证接口
- v1.0 (2026-01-23): 初始版本
