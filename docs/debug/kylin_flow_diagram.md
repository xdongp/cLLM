# Kylin 后端完整流程调用图

## 初始化流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. KylinBackend 构造                                        │
│    ├─ 检测模型格式 (.gguf / .bin)                           │
│    ├─ 设置 useGGMLDirect_ 标志                             │
│    ├─ 读取设备后端配置 (CPU/Metal/CUDA)                    │
│    └─ 创建模型实例                                           │
│        ├─ GGUF → GGMLTransformerModel                       │
│        └─ .bin → TransformerModel + ModelLoader            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. KylinBackend::initialize()                               │
│    │                                                         │
│    ├─ [GGUF 模式] initializeGGMLDirect()                   │
│    │   └─ GGMLTransformerModel::loadFromGGUF()             │
│    │       ├─ 创建 GGUFLoader                              │
│    │       ├─ loader_->loadConfig()                        │
│    │       ├─ 创建 GGML 上下文                              │
│    │       │   ├─ weightCtx_ (权重存储)                    │
│    │       │   ├─ kvCacheCtx_ (KV Cache)                  │
│    │       │   └─ computeCtx_ (计算临时)                   │
│    │       ├─ mapWeights()                                 │
│    │       │   ├─ 查找 token_embd.weight                  │
│    │       │   ├─ 查找 output.weight / lm_head.weight     │
│    │       │   ├─ 查找 output_norm.weight                 │
│    │       │   └─ 查找各层权重 (blk.N.*)                  │
│    │       └─ allocateKVCache()                           │
│    │           └─ 为每层分配 K/V cache                    │
│    │                                                         │
│    └─ [.bin 模式] initialize()                             │
│        ├─ loadRealWeights()                                │
│        └─ bindWeightsToModel()                             │
└─────────────────────────────────────────────────────────────┘
```

## 推理流程（GGML 直接模式）

```
┌─────────────────────────────────────────────────────────────┐
│ 3. KylinBackend::forward(inputIds)                         │
│    └─ GGMLTransformerModel::forward(inputIds32)            │
│        │                                                     │
│        ├─ computeCtx_->reset()                             │
│        ├─ buildForwardGraph(inputIds, kvCacheLen_)         │
│        │   │                                                 │
│        │   ├─ Stage 0: 创建输入张量                        │
│        │   │   └─ ggml_new_tensor_1d(I32, seqLen)         │
│        │   │                                                 │
│        │   ├─ Stage 1: Token Embedding                    │
│        │   │   └─ ggml_get_rows(tokEmbed_, inputTensor)   │
│        │   │       → [hidden, seq_len]                     │
│        │   │                                                 │
│        │   ├─ Stage 2-N: Transformer Layers               │
│        │   │   └─ buildLayerGraph(hidden_states, layer, i) │
│        │   │       │                                         │
│        │   │       ├─ Stage 2.1: Attention 归一化          │
│        │   │       │   ├─ ggml_rms_norm(input, eps)        │
│        │   │       │   └─ ggml_mul(normed, attnNorm)       │
│        │   │       │                                         │
│        │   │       ├─ Stage 2.2: Self-Attention            │
│        │   │       │   └─ buildAttentionGraph(...)         │
│        │   │       │       │                                 │
│        │   │       │       ├─ QKV 投影                     │
│        │   │       │       │   ├─ ggml_mul_mat(wq, input) │
│        │   │       │       │   ├─ ggml_mul_mat(wk, input) │
│        │   │       │       │   └─ ggml_mul_mat(wv, input) │
│        │   │       │       │                                 │
│        │   │       │       ├─ Reshape 为多头               │
│        │   │       │       │   └─ ggml_reshape_3d(...)     │
│        │   │       │       │                                 │
│        │   │       │       ├─ Q/K 归一化                   │
│        │   │       │       │   ├─ ggml_rms_norm(q, eps)    │
│        │   │       │       │   ├─ ggml_mul(q, attnQNorm)   │
│        │   │       │       │   ├─ ggml_rms_norm(k, eps)    │
│        │   │       │       │   └─ ggml_mul(k, attnKNorm)   │
│        │   │       │       │                                 │
│        │   │       │       ├─ RoPE 位置编码                │
│        │   │       │       │   ├─ ggml_rope_ext(q, pos, ...)│
│        │   │       │       │   └─ ggml_rope_ext(k, pos, ...)│
│        │   │       │       │                                 │
│        │   │       │       ├─ KV Cache 处理               │
│        │   │       │       │   ├─ 首次: 使用新 K/V        │
│        │   │       │       │   └─ 增量: 从 cache 读取     │
│        │   │       │       │                                 │
│        │   │       │       ├─ GQA 扩展 (如需要)           │
│        │   │       │       │   └─ repeat + permute        │
│        │   │       │       │                                 │
│        │   │       │       ├─ 注意力计算                   │
│        │   │       │       │   ├─ ggml_mul_mat(k, q)      │
│        │   │       │       │   ├─ ggml_scale(scores, 1/sqrt)│
│        │   │       │       │   ├─ ggml_diag_mask_inf(...) │
│        │   │       │       │   ├─ ggml_soft_max(scores)    │
│        │   │       │       │   └─ ggml_mul_mat(v, attn)   │
│        │   │       │       │                                 │
│        │   │       │       └─ 输出投影                     │
│        │   │       │           └─ ggml_mul_mat(wo, attnOut)│
│        │   │       │                                         │
│        │   │       ├─ Stage 2.3: 残差连接 (Attention)      │
│        │   │       │   └─ ggml_add(residual, attnOutput)   │
│        │   │       │                                         │
│        │   │       ├─ Stage 2.4: FFN 归一化                │
│        │   │       │   ├─ ggml_rms_norm(x, eps)           │
│        │   │       │   └─ ggml_mul(normed, ffnNorm)        │
│        │   │       │                                         │
│        │   │       ├─ Stage 2.5: FFN                        │
│        │   │       │   └─ buildFFNGraph(...)               │
│        │   │       │       ├─ ggml_mul_mat(wGate, input)  │
│        │   │       │       ├─ ggml_mul_mat(wUp, input)     │
│        │   │       │       ├─ ggml_silu(gate)              │
│        │   │       │       ├─ ggml_mul(silu, up)           │
│        │   │       │       └─ ggml_mul_mat(wDown, hidden) │
│        │   │       │                                         │
│        │   │       └─ Stage 2.6: 残差连接 (FFN)           │
│        │   │           └─ ggml_add(residual, ffnOutput)   │
│        │   │                                                 │
│        │   ├─ Stage N+1: 最终归一化                        │
│        │   │   ├─ ggml_rms_norm(hidden_states, eps)        │
│        │   │   └─ ggml_mul(normed, outputNorm)             │
│        │   │                                                 │
│        │   └─ Stage N+2: LM Head                           │
│        │       ├─ ggml_mul_mat(output_, hidden_states)     │
│        │       └─ ggml_transpose(logits) [if seqLen > 1]   │
│        │                                                 │
│        ├─ computeCtx_->buildGraph(logits)                  │
│        ├─ computeCtx_->compute(graph)                     │
│        ├─ flushKVCache()                                  │
│        └─ 提取 logits 并返回                               │
└─────────────────────────────────────────────────────────────┘
```

## 关键函数调用链

### 初始化链
```
KylinBackend::initialize()
  └─ initializeGGMLDirect()
      └─ GGMLTransformerModel::loadFromGGUF()
          ├─ GGUFLoader::loadConfig()
          ├─ GGMLTransformerModel::mapWeights()
          └─ GGMLTransformerModel::allocateKVCache()
```

### 推理链
```
KylinBackend::forward()
  └─ GGMLTransformerModel::forward()
      └─ buildForwardGraph()
          ├─ ggml_get_rows()  [Embedding]
          ├─ buildLayerGraph() × N
          │   └─ buildAttentionGraph()
          │       ├─ ggml_mul_mat() × 3  [QKV]
          │       ├─ ggml_reshape_3d() × 3
          │       ├─ ggml_rms_norm() × 2  [Q/K norm]
          │       ├─ ggml_rope_ext() × 2
          │       ├─ ggml_mul_mat()  [Q@K^T]
          │       ├─ ggml_soft_max()
          │       └─ ggml_mul_mat()  [Attn@V]
          └─ buildFFNGraph()
              ├─ ggml_mul_mat() × 2  [Gate/Up]
              ├─ ggml_silu()
              └─ ggml_mul_mat()  [Down]
```

## 数据流

### 张量形状变化

```
输入: [seq_len] (I32)
  ↓
Embedding: [hidden, seq_len] (F32)
  ↓
Layer 输入: [hidden, seq_len]
  ↓
Attention 归一化: [hidden, seq_len]
  ↓
QKV 投影: Q[2048, seq_len], K[1024, seq_len], V[1024, seq_len]
  ↓
Reshape: Q[128, 16, seq_len], K[128, 8, seq_len], V[128, 8, seq_len]
  ↓
Q/K Norm: 形状不变
  ↓
RoPE: 形状不变
  ↓
KV Cache: K[128, total_len, 8], V[128, total_len, 8]
  ↓
GQA 扩展: K[128, total_len, 16], V[128, total_len, 16]
  ↓
注意力分数: [total_len, seq_len, 16]
  ↓
注意力输出: [128, seq_len, 16]
  ↓
输出投影: [2048, seq_len]
  ↓
Reshape: [hidden, seq_len]
  ↓
残差连接: [hidden, seq_len]
  ↓
FFN: [hidden, seq_len]
  ↓
残差连接: [hidden, seq_len]
  ↓
最终归一化: [hidden, seq_len]
  ↓
LM Head: [vocab, seq_len]
  ↓
转置: [seq_len, vocab]
```

## 关键检查点

### 检查点 1: Embedding 输出
- **位置**: Stage 1 后
- **验证**: Shape, 数值范围, 与 llama_cpp 对比

### 检查点 2: QKV 投影后
- **位置**: buildAttentionGraph 中
- **验证**: Q/K/V 的形状和数值

### 检查点 3: RoPE 后
- **位置**: RoPE 应用后
- **验证**: 位置编码是否正确应用

### 检查点 4: 注意力分数
- **位置**: Softmax 前
- **验证**: 分数范围, 因果 mask

### 检查点 5: 注意力输出
- **位置**: Attention@V 后
- **验证**: 输出形状和数值

### 检查点 6: Layer 0 输出
- **位置**: 第一层后
- **验证**: Shape, 数值范围, 与 llama_cpp 对比

### 检查点 7: 最终 Logits
- **位置**: LM Head 后
- **验证**: Shape, Top-k tokens, 与 llama_cpp 对比
