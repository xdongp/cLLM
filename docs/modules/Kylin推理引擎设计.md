# Kylin (麒麟) 推理引擎设计文档

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](C++编程规范.md)：定义编码风格、命名规范等

---

## 0. 文档概述

### 0.1 设计目标

**Kylin (麒麟)** 是 cLLM 的自研高性能推理引擎，基于 **GGML** 构建，专注于 CPU 极致性能优化，可选支持 GPU。

**核心目标**：
- 🎯 **基于 GGML**：复用成熟的高性能张量计算库
- 🎯 **GGUF 格式优先**：原生支持量化模型，直接使用预量化模型
- 🎯 **CPU 优先**：充分利用 SIMD 指令（AVX2/AVX-512/NEON）
- 🎯 **GPU 可选**：通过 GGML 的 CUDA/Metal 后端支持 GPU 加速
- 🎯 **量化支持**：原生支持 Q4_K_M、Q8_0 等多种量化格式
- 🎯 **模块化设计**：易于扩展和定制

**命名含义**：
- **Kylin (麒麟)**：中国传统神兽，象征吉祥、智慧、速度
- 代表自研引擎的**高性能**和**中国原创**特色

### 0.2 与原设计的主要变更

| 方面 | 原设计 | 新设计 |
|-----|-------|-------|
| **底层计算** | 自研算子（朴素实现） | GGML（成熟优化） |
| **模型格式** | 扁平 .bin | **GGUF**（优先）+ safetensors |
| **量化支持** | 待开发 | **原生支持**（Q2_K ~ Q8_0） |
| **SIMD 优化** | 需自研 | **GGML 内置** |
| **GPU 支持** | 无 | **可选**（GGML CUDA/Metal） |
| **开发周期** | 12-18 周 | **6-10 周** |

### 0.3 技术挑战评估（更新后）

| 技术领域 | 难度 | 工作量估算 | 关键挑战 |
|---------|------|----------|---------|
| GGML 集成 | ⭐⭐⭐ | 1-2周 | API 封装、CMake 配置 |
| GGUF 模型加载器 | ⭐⭐⭐ | 2-3周 | 元数据解析、张量映射 |
| Transformer 适配 | ⭐⭐⭐ | 2-3周 | 基于 GGML 算子组装模型 |
| KV Cache 管理 | ⭐⭐ | 1周 | 与 GGML 内存管理协调 |
| GPU 支持（可选） | ⭐⭐⭐ | 1-2周 | GGML CUDA/Metal 后端 |
| **总计** | - | **6-10周** | - |

### 0.4 开发路线图（更新后）

```
阶段1: GGML 集成 (2周)
  ├─ 集成 GGML 库
  ├─ CMake 配置
  ├─ C++ 封装层
  └─ 基础算子验证

阶段2: GGUF 模型加载 (2-3周)
  ├─ GGUF 格式解析器
  ├─ 模型元数据读取
  ├─ 量化张量加载
  └─ Tokenizer 集成

阶段3: Transformer 实现 (2-3周)
  ├─ 基于 GGML 的 Attention
  ├─ FFN / RMSNorm / RoPE
  ├─ 完整推理流程
  └─ KV Cache 管理

阶段4: 优化与测试 (2周)
  ├─ Flash Attention（可选）
  ├─ 性能调优
  ├─ GPU 支持（可选）
  └─ 与 llama.cpp 后端对比

阶段5: 生产就绪 (1-2周)
  ├─ 集成到 cLLM 框架
  ├─ 文档完善
  └─ 压力测试
```

---

## 1. 系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                   InferenceEngine (接口层)                    │
└─────────────────────────┬────────────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │      KylinBackend (麒麟)      │
          ├───────────────────────────────┤
          │  - GGUFLoader (模型加载)      │
          │  - TransformerModel (推理)    │
          │  - KVCacheManager (缓存)      │
          └───────────────┬───────────────┘
                          │
┌─────────────────────────▼─────────────────────────────────────┐
│                      GGML 计算层                              │
├───────────────────────────────────────────────────────────────┤
│  Tensor 操作        │  量化支持           │  硬件后端          │
│  ─────────────────  │  ─────────────────  │  ─────────────────│
│  ggml_mul_mat       │  Q4_0, Q4_1         │  CPU (默认)        │
│  ggml_rms_norm      │  Q5_0, Q5_1         │  ├─ AVX2          │
│  ggml_rope          │  Q8_0, Q8_1         │  ├─ AVX-512       │
│  ggml_soft_max      │  Q4_K, Q5_K, Q6_K   │  └─ ARM NEON      │
│  ggml_silu          │  FP16, BF16         │  GPU (可选)        │
│  ggml_flash_attn    │                     │  ├─ CUDA          │
│                     │                     │  └─ Metal         │
└───────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖关系

```
cLLM Framework
    │
    ├──> ModelExecutor
    │       │
    │       └──> InferenceEngine (接口层)
    │               │
    │               └──> KylinBackend
    │                       │
    │                       ├──> GGUFLoader
    │                       │       ├─ 解析 GGUF 文件头
    │                       │       ├─ 读取模型配置
    │                       │       ├─ 加载量化张量
    │                       │       └─ 提取 Tokenizer 信息
    │                       │
    │                       ├──> GGMLContext
    │                       │       ├─ 内存管理
    │                       │       ├─ 计算图构建
    │                       │       └─ 后端调度 (CPU/GPU)
    │                       │
    │                       └──> TransformerModel
    │                               ├─ Embedding
    │                               ├─ TransformerBlock (x N)
    │                               │   ├─ RMSNorm
    │                               │   ├─ MultiHeadAttention (GQA)
    │                               │   │   └─ RoPE
    │                               │   └─ FeedForward (SwiGLU)
    │                               ├─ FinalNorm
    │                               └─ LMHead
    │
    ├──> KVCache (复用 cLLM 现有)
    ├──> Sampler (复用 cLLM 现有)
    └──> Tokenizer (复用 cLLM 现有 / 或从 GGUF 提取)
```

### 1.3 与其他后端的关系

```
                    ┌─────────────────────────────────┐
                    │        cLLM Server              │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │         ModelExecutor           │
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  llama.cpp    │         │     Kylin       │         │    LibTorch     │
│   Backend     │         │    Backend      │         │    Backend      │
├───────────────┤         ├─────────────────┤         ├─────────────────┤
│ ✅ GGUF       │         │ ✅ GGUF         │         │ ⚠️ safetensors  │
│ ✅ 量化       │         │ ✅ 量化 (GGML)  │         │ ❌ 量化          │
│ ✅ CUDA       │         │ ✅ CPU 优先     │         │ ✅ CUDA         │
│ ✅ 生产级     │         │ 🎯 可定制       │         │ ⚠️ 开发用       │
└───────────────┘         └─────────────────┘         └─────────────────┘
```

**定位差异**：
- **llama.cpp**：生产级，开箱即用，性能最优
- **Kylin**：自研可控，可深度定制，学习目的
- **LibTorch**：开发调试，快速原型验证

---

## 2. 核心组件设计

### 2.1 GGML 集成层

#### 2.1.1 GGMLContext

**职责**：封装 GGML 的上下文管理，提供 C++ 友好的接口。

```cpp
// include/cllm/inference/ggml_context.h
namespace cllm::inference {

class GGMLContext {
public:
    explicit GGMLContext(size_t memSize);
    ~GGMLContext();
    
    // 张量创建
    ggml_tensor* newTensor1D(ggml_type type, int64_t ne0);
    ggml_tensor* newTensor2D(ggml_type type, int64_t ne0, int64_t ne1);
    ggml_tensor* newTensor3D(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
    
    // 计算图
    ggml_cgraph* buildGraph(ggml_tensor* output);
    void compute(ggml_cgraph* graph);
    
    // 后端管理
    void setBackend(BackendType type);  // CPU, CUDA, Metal
    
    ggml_context* raw() { return ctx_; }
    
private:
    ggml_context* ctx_;
    std::vector<uint8_t> buffer_;
    BackendType backend_ = BackendType::CPU;
};

} // namespace cllm::inference
```

#### 2.1.2 后端类型

```cpp
enum class BackendType {
    CPU,      // 默认，支持 AVX2/AVX-512/NEON
    CUDA,     // NVIDIA GPU（可选）
    Metal,    // Apple GPU（可选）
    Auto      // 自动选择最优后端
};
```

### 2.2 GGUF 模型加载器

#### 2.2.1 GGUF 格式概述

```
GGUF 文件结构:
┌─────────────────────────────────────┐
│  Magic Number: "GGUF"               │  4 bytes
├─────────────────────────────────────┤
│  Version: 3                         │  4 bytes
├─────────────────────────────────────┤
│  Tensor Count                       │  8 bytes
├─────────────────────────────────────┤
│  Metadata KV Count                  │  8 bytes
├─────────────────────────────────────┤
│  Metadata Key-Value Pairs           │  Variable
│  ├─ general.architecture: "qwen2"   │
│  ├─ general.name: "Qwen3-0.6B"      │
│  ├─ qwen2.context_length: 32768     │
│  ├─ qwen2.embedding_length: 1024    │
│  ├─ qwen2.block_count: 28           │
│  ├─ tokenizer.ggml.model: "gpt2"    │
│  └─ ...                             │
├─────────────────────────────────────┤
│  Tensor Infos                       │  Variable
│  ├─ name, dims, type, offset        │
│  └─ ...                             │
├─────────────────────────────────────┤
│  Alignment Padding                  │
├─────────────────────────────────────┤
│  Tensor Data (量化/FP16/FP32)       │  Bulk data
└─────────────────────────────────────┘
```

#### 2.2.2 GGUFLoader 接口

```cpp
// include/cllm/inference/gguf_loader.h
namespace cllm::inference {

struct GGUFModelConfig {
    std::string architecture;      // "qwen2", "llama", etc.
    std::string name;
    size_t contextLength;
    size_t embeddingLength;
    size_t blockCount;
    size_t headCount;
    size_t headCountKV;            // GQA
    size_t feedForwardLength;
    float rmsNormEps;
    float ropeTheta;
    size_t vocabSize;
    // ... 其他配置
};

class GGUFLoader {
public:
    explicit GGUFLoader(const std::string& path);
    ~GGUFLoader();
    
    // 检查文件有效性
    bool isValid() const;
    
    // 加载模型配置（从元数据）
    GGUFModelConfig loadConfig();
    
    // 加载张量到 GGML 上下文
    void loadTensors(GGMLContext* ctx, std::map<std::string, ggml_tensor*>& tensors);
    
    // 获取 Tokenizer 信息（如果内嵌）
    std::optional<TokenizerInfo> getTokenizerInfo();
    
    // 获取量化类型
    ggml_type getQuantizationType() const;
    
private:
    std::string path_;
    void* mmapData_;      // 内存映射
    size_t fileSize_;
    
    // GGUF 解析
    bool parseHeader();
    bool parseMetadata();
    bool parseTensorInfos();
};

} // namespace cllm::inference
```

#### 2.2.3 支持的量化类型

| 类型 | 描述 | 压缩比 | 精度损失 | 推荐场景 |
|-----|------|-------|---------|---------|
| `Q4_0` | 4-bit 块量化 | 4x | 中 | 快速测试 |
| `Q4_K_M` | 4-bit K-quants | 4x | 低 | **推荐** |
| `Q5_K_M` | 5-bit K-quants | 3.2x | 很低 | 精度优先 |
| `Q8_0` | 8-bit 量化 | 2x | 极低 | 高精度 |
| `F16` | 半精度浮点 | 2x | 无 | 基准对比 |
| `F32` | 单精度浮点 | 1x | 无 | 调试 |

### 2.3 Transformer 模型

#### 2.3.1 TransformerModel 接口

```cpp
// include/cllm/inference/transformer_model.h
namespace cllm::inference {

class TransformerModel {
public:
    explicit TransformerModel(const GGUFModelConfig& config, GGMLContext* ctx);
    ~TransformerModel();
    
    // 加载权重
    void loadWeights(const std::map<std::string, ggml_tensor*>& tensors);
    
    // 前向传播
    // 输入: token IDs
    // 输出: logits [seq_len, vocab_size]
    ggml_tensor* forward(
        const std::vector<int32_t>& inputIds,
        size_t pastLength = 0    // KV Cache 已有长度
    );
    
    // 单 token 生成（增量推理）
    ggml_tensor* forwardOneToken(
        int32_t tokenId,
        size_t position
    );
    
    // KV Cache 管理
    void clearKVCache();
    size_t getKVCacheLength() const;
    
private:
    GGUFModelConfig config_;
    GGMLContext* ctx_;
    
    // 模型组件（使用 GGML 张量）
    ggml_tensor* embedding_;
    std::vector<TransformerBlock> blocks_;
    ggml_tensor* finalNorm_;
    ggml_tensor* lmHead_;
    
    // KV Cache
    std::vector<ggml_tensor*> kCaches_;
    std::vector<ggml_tensor*> vCaches_;
};

} // namespace cllm::inference
```

#### 2.3.2 核心算子（基于 GGML）

| 算子 | GGML 函数 | 说明 |
|-----|----------|-----|
| 矩阵乘法 | `ggml_mul_mat` | 自动处理量化 |
| RMS Norm | `ggml_rms_norm` | 支持 eps 参数 |
| RoPE | `ggml_rope` | 支持多种 RoPE 变体 |
| Softmax | `ggml_soft_max` | 数值稳定实现 |
| SiLU | `ggml_silu` | SwiGLU 激活函数 |
| Flash Attention | `ggml_flash_attn_ext` | 可选，长序列优化 |

#### 2.3.3 Attention 计算原理

```
Multi-Head Attention (GQA) 流程:

1. QKV 投影:
   Q = X @ Wq    [seq, num_heads * head_dim]
   K = X @ Wk    [seq, num_kv_heads * head_dim]
   V = X @ Wv    [seq, num_kv_heads * head_dim]

2. 重塑为多头:
   Q: [num_heads, seq, head_dim]
   K: [num_kv_heads, seq, head_dim]
   V: [num_kv_heads, seq, head_dim]

3. 应用 RoPE:
   Q, K = RoPE(Q, K, positions)

4. GQA 广播 (如果 num_kv_heads < num_heads):
   K, V 广播到 num_heads

5. Attention 计算:
   scores = Q @ K^T / sqrt(head_dim)
   scores = scores + causal_mask
   weights = softmax(scores)
   output = weights @ V

6. 输出投影:
   output = concat(heads) @ Wo
```

### 2.4 KV Cache 管理

#### 2.4.1 KVCacheManager 接口

```cpp
// include/cllm/inference/kv_cache_manager.h
namespace cllm::inference {

class KVCacheManager {
public:
    KVCacheManager(
        size_t numLayers,
        size_t numKVHeads,
        size_t headDim,
        size_t maxSeqLen,
        GGMLContext* ctx
    );
    
    // 获取指定层的 KV Cache
    std::pair<ggml_tensor*, ggml_tensor*> getCache(size_t layerIdx);
    
    // 更新 Cache（追加新的 K, V）
    void updateCache(
        size_t layerIdx,
        ggml_tensor* newK,
        ggml_tensor* newV,
        size_t position
    );
    
    // 清空 Cache（新对话）
    void clear();
    
    // 获取当前序列长度
    size_t getCurrentLength() const;
    
    // 内存使用统计
    size_t getMemoryUsage() const;
    
private:
    std::vector<ggml_tensor*> kCaches_;
    std::vector<ggml_tensor*> vCaches_;
    size_t currentLength_ = 0;
};

} // namespace cllm::inference
```

---

## 3. 性能优化策略

### 3.1 CPU 优化（默认）

| 优化技术 | 来源 | 说明 |
|---------|------|-----|
| **AVX2/AVX-512** | GGML 内置 | 向量化矩阵运算 |
| **ARM NEON** | GGML 内置 | Apple Silicon / ARM 优化 |
| **量化计算** | GGML 内置 | Q4/Q8 直接计算，无需反量化 |
| **内存映射** | mmap | 快速加载大模型 |
| **缓存友好** | GGML 内置 | 分块计算，提高缓存命中 |
| **多线程** | GGML 内置 | 自动利用多核 |

### 3.2 GPU 优化（可选）

| 后端 | 支持平台 | 启用方式 |
|-----|---------|---------|
| **CUDA** | NVIDIA GPU | 编译时 `-DGGML_CUDA=ON` |
| **Metal** | Apple GPU | 编译时 `-DGGML_METAL=ON` |

**GPU 加速效果**（参考）：
- 小模型 (<1B): 2-3x 加速
- 中模型 (1-7B): 5-10x 加速
- 大模型 (>7B): 10-20x 加速

### 3.3 Flash Attention

```
启用条件:
├─ 序列长度 > 512（短序列收益低）
├─ 需要处理长上下文
└─ 内存受限场景

GGML 实现: ggml_flash_attn_ext()
├─ 支持因果 mask
├─ 支持 GQA
├─ 支持 ALiBi
└─ CPU/GPU 均可用
```

---

## 4. 配置与使用

### 4.1 编译配置

```cmake
# CMakeLists.txt 关键配置

# GGML 选项
option(KYLIN_ENABLE_CUDA "Enable CUDA support" OFF)
option(KYLIN_ENABLE_METAL "Enable Metal support" OFF)
option(KYLIN_ENABLE_FLASH_ATTN "Enable Flash Attention" ON)

# 集成 GGML
add_subdirectory(third_party/ggml)

# Kylin 后端
add_library(kylin_backend
    src/inference/ggml_context.cpp
    src/inference/gguf_loader.cpp
    src/inference/transformer_model.cpp
    src/inference/kv_cache_manager.cpp
    src/inference/kylin_backend.cpp
)
target_link_libraries(kylin_backend PRIVATE ggml)
```

### 4.2 运行时配置

```yaml
# config.yaml
backend:
  type: kylin  # 使用 Kylin 后端
  
kylin:
  device: cpu           # cpu / cuda / metal / auto
  threads: 0            # 0 = 自动检测
  use_mmap: true        # 内存映射加载
  use_flash_attn: true  # Flash Attention
  
  # GPU 配置（可选）
  gpu_layers: 0         # 0 = 全 CPU，>0 = 部分层在 GPU
```

### 4.3 使用示例

```cpp
#include "cllm/inference/kylin_backend.h"

// 创建 Kylin 后端
KylinBackend backend;

// 加载 GGUF 模型
if (!backend.loadModel("/path/to/model.gguf")) {
    std::cerr << "Failed to load model" << std::endl;
    return -1;
}

// 推理
std::vector<int32_t> inputIds = {1, 72, 105};  // "Hi"
auto logits = backend.forward(inputIds);

// 增量生成
int32_t nextToken = backend.forwardOneToken(72, 3);
```

---

## 5. 与 llama.cpp 后端的对比

| 方面 | Kylin Backend | llama.cpp Backend |
|-----|--------------|-------------------|
| **代码复杂度** | 低（封装 GGML） | 高（直接使用 llama.cpp API） |
| **可定制性** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中 |
| **性能** | ⭐⭐⭐⭐ 接近 | ⭐⭐⭐⭐⭐ 最优 |
| **学习价值** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐ 低（黑盒） |
| **维护成本** | 中（需跟进 GGML） | 低（社区维护） |
| **新功能支持** | 需自行实现 | 自动获得 |

**选择建议**：
- **生产环境**：优先使用 llama.cpp 后端
- **学习研究**：使用 Kylin 后端，可深入理解推理原理
- **定制需求**：使用 Kylin 后端，便于修改和扩展

---

## 6. 开发指南

### 6.1 目录结构

```
src/inference/
├── ggml_context.cpp       # GGML 上下文封装
├── gguf_loader.cpp        # GGUF 加载器
├── transformer_model.cpp  # Transformer 模型
├── kv_cache_manager.cpp   # KV Cache 管理
└── kylin_backend.cpp      # Kylin 后端主类

include/cllm/inference/
├── ggml_context.h
├── gguf_loader.h
├── transformer_model.h
├── kv_cache_manager.h
└── kylin_backend.h

third_party/
└── ggml/                  # GGML 库（git submodule）
```

### 6.2 编译和测试

```bash
# 获取 GGML
cd third_party
git clone https://github.com/ggerganov/ggml.git

# 编译（CPU）
cd ../build
cmake .. -DKYLIN_ENABLE_CUDA=OFF
make -j$(nproc)

# 编译（CUDA）
cmake .. -DKYLIN_ENABLE_CUDA=ON
make -j$(nproc)

# 测试
./bin/test_kylin_backend --model /path/to/model.gguf
```

### 6.3 调试建议

1. **正确性验证**：与 llama.cpp 后端对比输出
2. **性能分析**：使用 `perf` 或 `Instruments` 分析热点
3. **内存检查**：使用 `valgrind` 或 `AddressSanitizer`

---

## 7. 参考资料

- [GGML GitHub](https://github.com/ggerganov/ggml) - GGML 张量计算库
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - GGUF 格式文档
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 参考实现
- [推理引擎接口设计.md](推理引擎接口设计.md) - cLLM 接口规范

---

## 8. 总结

**Kylin (麒麟) 推理引擎** v2.0 采用 GGML 作为底层计算库，实现了：

✅ **GGUF 原生支持**：直接加载预量化模型  
✅ **高性能计算**：复用 GGML 的 SIMD 优化  
✅ **量化推理**：Q4_K_M、Q8_0 等多种格式  
✅ **CPU 优先**：开箱即用，无需 GPU  
✅ **GPU 可选**：通过 GGML CUDA/Metal 支持  
✅ **模块化设计**：易于理解、修改和扩展  

**设计理念**：
- 站在巨人肩膀上（复用 GGML），而非重复造轮子
- 保持自研可控，便于深入学习和定制
- 优先实现核心功能，逐步完善优化
