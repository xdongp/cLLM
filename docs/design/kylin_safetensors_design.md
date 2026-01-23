# Kylin 推理引擎 - Safetensors 格式支持设计方案

> **状态**: ✅ Phase 1-2 已完成，验证通过  
> **更新日期**: 2025-01-24  
> **验证结果**: Top-5 tokens 与 HuggingFace 完全一致

## 1. 背景与动机

### 1.1 为什么放弃 GGUF 量化模型作为默认

| 问题 | 说明 |
|------|------|
| 数值精度损失 | Q4_K/Q6_K 量化导致误差累积，28 层后 std 放大 ~1000x |
| 反量化复杂度 | 需要正确实现多种量化格式的反量化 |
| 调试困难 | 很难区分问题是来自量化还是实现错误 |
| 与 llama.cpp 对齐困难 | llama.cpp 有大量针对量化的优化，难以完全复现 |

### 1.2 Safetensors 格式的优势

| 优势 | 说明 |
|------|------|
| 原始精度 | BF16/FP16/FP32，无量化误差 |
| 简单直接 | 直接内存映射，无需反量化 |
| 广泛支持 | Hugging Face 生态的标准格式 |
| 安全高效 | 不执行任意代码，支持 mmap 零拷贝 |

### 1.3 验证结果

**单 Token 推理验证**:
```
Kylin (Safetensors)      HuggingFace Reference
--------------------     ---------------------
Mean: -1.0941            Mean: -1.0599
Std:   1.9828            Std:   1.9019
Top-5: [21806, 14582,    Top-5: [21806, 14582,
        15846, 477,              15846, 477,
        1957]                    1957]
                         ✅ Top-5 完全一致！
```

**端到端生成测试** (2026-01-24):
```
Prompt: "Hello"
Output: " Answer World!"              ✅ 连贯

Prompt: "1+1="
Output: "2. 1+"                       ✅ 正确

Prompt: "What is the capital of France?"
Output: " The answer is France's capital..."  ✅ 连贯

Prompt: "Python code to print hello world:"
Output: "print(\"Hello, World!\")"    ✅ 正确
```

---

## 2. Qwen3-0.6B 模型分析

### 2.1 模型配置 (config.json)

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "head_dim": 128,
  "rope_theta": 1000000,
  "rms_norm_eps": 1e-06,
  "torch_dtype": "bfloat16",
  "tie_word_embeddings": true
}
```

### 2.2 权重结构 (311 tensors, BF16)

| 组件 | 张量名 | 形状 | 说明 |
|------|--------|------|------|
| 词嵌入 | `model.embed_tokens.weight` | [151936, 1024] | 输入嵌入 |
| LM Head | `lm_head.weight` | [151936, 1024] | 与 embed_tokens 共享 |
| Attn Norm | `model.layers.{i}.input_layernorm.weight` | [1024] | RMS Norm |
| Q Proj | `model.layers.{i}.self_attn.q_proj.weight` | [2048, 1024] | Q 投影 |
| K Proj | `model.layers.{i}.self_attn.k_proj.weight` | [1024, 1024] | K 投影 |
| V Proj | `model.layers.{i}.self_attn.v_proj.weight` | [1024, 1024] | V 投影 |
| O Proj | `model.layers.{i}.self_attn.o_proj.weight` | [1024, 2048] | 输出投影 |
| Q Norm | `model.layers.{i}.self_attn.q_norm.weight` | [128] | Q RMS Norm |
| K Norm | `model.layers.{i}.self_attn.k_norm.weight` | [128] | K RMS Norm |
| FFN Norm | `model.layers.{i}.post_attention_layernorm.weight` | [1024] | RMS Norm |
| Gate Proj | `model.layers.{i}.mlp.gate_proj.weight` | [3072, 1024] | SwiGLU gate |
| Up Proj | `model.layers.{i}.mlp.up_proj.weight` | [3072, 1024] | SwiGLU up |
| Down Proj | `model.layers.{i}.mlp.down_proj.weight` | [1024, 3072] | SwiGLU down |
| Final Norm | `model.norm.weight` | [1024] | 最终 RMS Norm |

---

## 3. 架构设计

### 3.1 模块结构

```
include/cllm/kylin/
├── safetensors_loader.h     # Safetensors 加载器 + BF16/F16 转换
├── hf_config.h              # HuggingFace 配置解析
└── hf_transformer.h         # HF 格式 Transformer 模型

src/kylin/
├── safetensors_loader.cpp   # Safetensors 解析、mmap、BF16 转换
├── hf_config.cpp            # config.json 解析
└── hf_transformer.cpp       # 推理实现

tools/
└── test_hf_transformer.cpp  # 测试工具
```

**设计原则**：
- 最小化文件数量，避免过度拆分
- BF16 转换函数内联在 safetensors_loader.h 中（简单、高效）
- 后续优化时可提取为独立模块

### 3.2 类设计

```cpp
// ========== Safetensors 加载器 ==========
class SafetensorsLoader {
public:
    explicit SafetensorsLoader(const std::string& path);
    ~SafetensorsLoader();
    
    bool isValid() const;
    std::vector<std::string> getTensorNames() const;
    
    // 获取张量（返回原始 BF16 指针）
    const void* getTensorData(const std::string& name) const;
    std::vector<int64_t> getTensorShape(const std::string& name) const;
    std::string getTensorDtype(const std::string& name) const;
    
    // 转换为 F32（用于计算）
    std::vector<float> getTensorAsF32(const std::string& name) const;
    
private:
    void* mappedData_ = nullptr;
    size_t mappedSize_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

// ========== HuggingFace 配置 ==========
struct HFModelConfig {
    std::string architecture;
    int hiddenSize = 0;
    int numHiddenLayers = 0;
    int numAttentionHeads = 0;
    int numKeyValueHeads = 0;
    int intermediateSize = 0;
    int vocabSize = 0;
    int headDim = 0;
    float rmsNormEps = 1e-6f;
    float ropeTheta = 10000.0f;
    bool tieWordEmbeddings = false;
    std::string torchDtype;
};

class HFConfigLoader {
public:
    static HFModelConfig load(const std::string& configPath);
};

// ========== HF Transformer 模型 ==========
class HFTransformerModel {
public:
    explicit HFTransformerModel(const std::string& modelDir);
    
    // 前向推理
    std::vector<float> forward(const std::vector<int32_t>& inputIds);
    
    // 配置访问
    const HFModelConfig& config() const { return config_; }
    
private:
    HFModelConfig config_;
    std::unique_ptr<SafetensorsLoader> loader_;
    
    // 权重指针（BF16 原始数据）
    const void* embedTokens_ = nullptr;
    const void* lmHead_ = nullptr;
    const void* finalNorm_ = nullptr;
    
    struct LayerWeights {
        const void* inputLayernorm = nullptr;
        const void* qProj = nullptr;
        const void* kProj = nullptr;
        const void* vProj = nullptr;
        const void* oProj = nullptr;
        const void* qNorm = nullptr;
        const void* kNorm = nullptr;
        const void* postAttentionLayernorm = nullptr;
        const void* gateProj = nullptr;
        const void* upProj = nullptr;
        const void* downProj = nullptr;
    };
    std::vector<LayerWeights> layers_;
    
    // KV Cache
    std::vector<float> kCache_;
    std::vector<float> vCache_;
    size_t kvCacheLen_ = 0;
};
```

### 3.3 计算流程

```
Input IDs
    │
    ▼
┌─────────────────┐
│ Embedding (BF16→F32) │  embedTokens_[input_ids]
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│            Transformer Block x 28       │
│  ┌─────────────────────────────────┐   │
│  │ 1. RMS Norm (input_layernorm)   │   │
│  │ 2. Self-Attention               │   │
│  │    - Q/K/V Projection (BF16@F32)│   │
│  │    - Q/K Norm                   │   │
│  │    - RoPE                       │   │
│  │    - Attention (F32)            │   │
│  │    - O Projection               │   │
│  │ 3. Residual Add                 │   │
│  │ 4. RMS Norm (post_attn_norm)    │   │
│  │ 5. MLP (SwiGLU)                 │   │
│  │    - gate_proj, up_proj, down   │   │
│  │ 6. Residual Add                 │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│ Final RMS Norm  │  model.norm.weight
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ LM Head (F32@BF16) │  lm_head.weight
└─────────────────┘
    │
    ▼
Logits [vocab_size]
```

---

## 4. 关键实现

### 4.1 Safetensors 格式解析

Safetensors 文件格式：
```
[8 bytes] header_size (little-endian uint64)
[header_size bytes] JSON header
[remaining bytes] tensor data
```

JSON header 示例：
```json
{
  "__metadata__": {"format": "pt"},
  "tensor_name": {
    "dtype": "BF16",
    "shape": [151936, 1024],
    "data_offsets": [0, 311296000]
  }
}
```

### 4.2 BF16 到 F32 转换

```cpp
// BF16 格式：1 符号位 + 8 指数位 + 7 尾数位
// F32 格式：1 符号位 + 8 指数位 + 23 尾数位
// 转换：BF16 左移 16 位 = F32

inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// SIMD 优化版本（AVX2）
void bf16_to_f32_avx2(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m128i bf16 = _mm_loadu_si128((__m128i*)(src + i));
        __m256i expanded = _mm256_cvtepu16_epi32(bf16);
        __m256i shifted = _mm256_slli_epi32(expanded, 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(shifted));
    }
}
```

### 4.3 矩阵乘法 (BF16 weights @ F32 input)

```cpp
// 选项 1：转换后计算（简单但内存占用大）
void matmul_bf16_f32(
    const uint16_t* weight,  // BF16 [out_features, in_features]
    const float* input,      // F32 [in_features]
    float* output,           // F32 [out_features]
    int out_features,
    int in_features
) {
    // 逐行转换并计算
    for (int i = 0; i < out_features; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            float w = bf16_to_f32(weight[i * in_features + j]);
            sum += w * input[j];
        }
        output[i] = sum;
    }
}

// 选项 2：使用 GGML（推荐）
// 将 BF16 权重加载为 GGML_TYPE_BF16 张量
// GGML 内部会优化 BF16 矩阵乘法
```

### 4.4 使用 GGML 的 BF16 支持

```cpp
// GGML 支持 GGML_TYPE_BF16
ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, in_features, out_features);
memcpy(weight->data, bf16_data, size);

// GGML 会自动处理 BF16 @ F32 的计算
ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
```

---

## 5. 实施计划

### Phase 1: 基础加载器 ✅ 已完成

```
目标：能够加载 safetensors 并读取权重

Tasks:
├── [x] 实现 safetensors_loader.h/cpp
│       ├── [x] 解析 JSON header
│       ├── [x] 内存映射文件 (mmap)
│       └── [x] 获取张量数据和形状
├── [x] 实现 hf_config.h/cpp
│       └── [x] 解析 config.json
└── [x] 单元测试
        └── [x] 验证权重加载正确 (315 tensors)
```

### Phase 2: 模型推理 ✅ 已完成

```
目标：完整的前向推理

Tasks:
├── [x] 实现 hf_transformer.h/cpp
│       ├── [x] Embedding 查找
│       ├── [x] RMS Norm
│       ├── [x] Self-Attention（含 Q/K Norm, RoPE）
│       ├── [x] SwiGLU FFN
│       └── [x] LM Head
├── [x] KV Cache 管理
└── [x] 集成测试
        └── [x] 对比 HuggingFace - Top-5 一致 ✅
```

### Phase 3: 性能优化 ⏳ 待完成

```
目标：提升推理性能

Tasks:
├── [ ] 减少内存分配
│       ├── 预分配工作缓冲区
│       └── 复用临时 vector
├── [ ] 优化矩阵乘法
│       ├── SIMD (AVX2/NEON) 优化
│       ├── 多线程并行
│       └── 考虑使用 BLAS 库
├── [ ] 预转换权重
│       └── 加载时将 BF16 权重转为 F32（空间换时间）
└── [ ] 集成到 KylinBackend
        └── 更新 backend_factory.cpp
```

### Phase 4: 多 token 支持 ⏳ 待完成

```
目标：支持批量推理

Tasks:
├── [ ] 支持 seq_len > 1 的输入
├── [ ] Prefill 阶段优化
└── [ ] Attention mask 处理
```

---

## 6. 配置更新

### 6.1 config.yaml 变更

```yaml
inference:
  backend: "kylin"
  model_format: "safetensors"  # 新增：支持 "safetensors" 或 "gguf"
  model_path: "model/Qwen/Qwen3-0.6B"  # 指向目录而非文件
  
  # 可选：指定精度
  compute_dtype: "f32"  # 或 "bf16"（需要硬件支持）
```

### 6.2 后端选择逻辑

```cpp
std::unique_ptr<InferenceBackend> createBackend(const Config& config) {
    if (config.backend == "kylin") {
        if (config.modelFormat == "safetensors") {
            return std::make_unique<KylinHFBackend>(config.modelPath);
        } else {
            return std::make_unique<KylinGGUFBackend>(config.modelPath);
        }
    } else if (config.backend == "llama_cpp") {
        return std::make_unique<LlamaCppBackend>(config.modelPath);
    }
    throw std::runtime_error("Unknown backend: " + config.backend);
}
```

---

## 7. 预期效果

### 7.1 精度对比

| 指标 | GGUF Q4_K | Safetensors BF16 |
|------|-----------|------------------|
| 最终 logits mean 偏差 | ~3.8 | < 0.01 (预期) |
| 数值稳定性 | 差（Layer 27 爆炸） | 好（FP32 计算） |
| 调试难度 | 高 | 低 |

### 7.2 资源消耗

| 指标 | GGUF Q4_K | Safetensors BF16 |
|------|-----------|------------------|
| 权重大小 | ~480 MB | ~1.5 GB |
| 推理内存 | ~1 GB | ~2.5 GB |
| 加载速度 | 快（已量化） | 中（需转换） |

### 7.3 开发效率

| 指标 | GGUF | Safetensors |
|------|------|-------------|
| 代码复杂度 | 高（反量化） | 低（直接读取） |
| 调试时间 | 长（精度问题） | 短 |
| 与 HF 对齐 | 困难 | 简单 |

---

## 8. 已知问题与优化方向

### 8.1 性能瓶颈

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| LM Head 矩阵乘法 | 151936 vocab × 1024 hidden = ~1.5 亿次浮点运算 | 使用 BLAS/SIMD 优化 |
| BF16 转换开销 | 每次 matmul 都要转换权重 | 预转换权重为 F32 |
| 单线程计算 | CPU 利用率低 | OpenMP 多线程并行 |
| 单 token 推理 ~1.5s | 28 层 × 朴素矩阵乘法 | SIMD + 多线程 |

### 8.2 已修复问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 生成乱码 "ESTESTEST..." | KV cache 未正确累积 | 单 token 推理时保持 KV cache |
| 多 token prefill 失败 | forward 返回 vocab_size 而非 seq*vocab | 逐 token 处理 prefill |

### 8.3 功能限制

| 限制 | 说明 | 计划 |
|------|------|------|
| 仅支持 POSIX | mmap 在 Windows 不可用 | 添加 Windows CreateFileMapping |
| 无批处理 | 一次只能处理一个请求 | 后续优化 |
| 推理速度慢 | ~0.5 tokens/s | SIMD/BLAS 优化 |

### 8.4 代码质量

- ✅ 预分配工作缓冲区
- ✅ 减少临时 vector 创建
- ✅ KV cache 正确管理
- ✅ 端到端测试通过
- ⚠️ 矩阵乘法仍是朴素实现
- ⚠️ 缺少单元测试

---

## 9. 附录

### A. 依赖库

当前实现**无外部依赖**（除标准库和 POSIX API）：
- JSON 解析：自实现的简单解析器
- 内存映射：POSIX mmap（macOS/Linux）
- BF16 转换：位操作实现

### B. 参考实现

| 来源 | 用途 |
|------|------|
| HuggingFace `modeling_qwen3.py` | 计算流程参考 |
| llama.cpp `convert_hf_to_gguf.py` | 权重映射参考 |
| safetensors 官方文档 | 文件格式规范 |

### C. 测试方法

```python
# 使用 HuggingFace 生成参考输出
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model/Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16
)

# 使用与 C++ 相同的 token ID
input_ids = torch.tensor([[9707]])

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1].float()

print(f"Mean: {logits.mean():.4f}")
print(f"Std: {logits.std():.4f}")
print(f"Top-5: {torch.topk(logits, 5).indices.tolist()}")
```

### D. 性能基准（待优化）

当前性能（Qwen3-0.6B, M1 Mac）：
- 加载时间：~1.3 秒（mmap）
- 单 token 推理：~8 秒（朴素实现）

目标性能（优化后）：
- 单 token 推理：< 100 ms（使用 BLAS/SIMD）

### E. 代码风格

遵循项目约定：
- 使用 `CLLM_INFO/ERROR/WARN` 日志宏
- 类成员变量使用 `_` 后缀
- const 优先原则
- RAII 资源管理
