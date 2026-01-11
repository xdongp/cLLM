# Kylin (麒麟) 推理引擎设计文档

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](C++编程规范.md)：定义编码风格、命名规范等

## 0. 文档概述

### 0.1 设计目标

**Kylin (麒麟)** 是 cLLM 的自研高性能推理引擎，专注于 CPU 极致性能优化。

**核心目标**：
- 纯 C++ 实现，无依赖 (除标准库外)
- 支持 Qwen3 等主流 Transformer 架构
- CPU 极致优化，充分利用 SIMD 指令
- 支持多种量化格式（FP32/FP16/INT8/INT4）
- 高性能，低延迟，低内存占用
- 模块化设计，易于扩展

**命名含义**：
- **Kylin (麒麟)**：中国传统神兽，象征吉祥、智慧、速度
- 代表自研引擎的**高性能**和**中国原创**特色

### 0.2 技术挑战评估

| 技术领域 | 难度 | 工作量估算 | 关键挑战 |
|---------|------|----------|---------|
| Transformer 架构实现 | ⭐⭐⭐⭐⭐ | 4-6周 | Multi-head attention、RoPE、RMSNorm |
| 模型加载器 | ⭐⭐⭐⭐ | 2-3周 | 扁平 .bin 解析、权重映射 |
| SIMD 优化 | ⭐⭐⭐⭐⭐ | 3-4周 | AVX2/AVX-512 矩阵运算 |
| 量化支持 | ⭐⭐⭐⭐ | 2-3周 | INT8/INT4 量化推理 |
| 内存管理 | ⭐⭐⭐ | 1-2周 | 高效内存分配和 KV Cache |
| **总计** | - | **12-18周** | - |

### 0.3 开发路线图

```
阶段1: MVP 基础架构 (3周) - ✅ 已完成
  ├─ 张量抽象层 (FP32, CPU)
  ├─ 简化内存管理
  └─ 扁平 .bin 模型加载器

阶段2: Transformer 核心 (6周) - 🚧 进行中
  ├─ Attention 机制
  ├─ Feed-Forward 网络
  ├─ Normalization 层
  └─ Position Encoding (RoPE)

阶段3: 优化与量化 (5周) - ⏳ 待开发
  ├─ SIMD 优化 (AVX2/AVX-512)
  ├─ 量化支持 (INT8/INT4)
  └─ KV Cache 优化

阶段4: 集成与测试 (4周) - 🚧 进行中
  ├─ 集成到 cLLM 框架 ✅
  ├─ 单元测试
  ├─ 性能测试
  └─ 与 LibTorch 后端对比
```

## 1. 系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│              InferenceEngine (接口层)                     │
└─────────────────────┬────────────────────────────────────┘
                      │
        ┌─────────────▼────────────┐
        │   KylinBackend (麒麟)    │
        ├──────────────────────────┤
        │ - TransformerModel       │
        │ - ModelLoader            │
        │ - 自研算子库             │
        └──────────┬───────────────┘
                   │
        ┌──────────▼────────────────────────────────┐
        │      Layer 1: 模型抽象层                  │
        │  ┌──────────────────────────────────┐    │
        │  │ ModelLoader  │  ModelWeights     │    │
        │  │ ModelConfig  │  Tokenizer (复用) │    │
        │  └──────────────────────────────────┘    │
        ├───────────────────────────────────────────┤
        │      Layer 2: Transformer 核心层          │
        │  ┌──────────────────────────────────┐    │
        │  │ TransformerModel                 │    │
        │  │  ├─ TransformerBlock (x N)       │    │
        │  │  │   ├─ MultiHeadAttention       │    │
        │  │  │   ├─ FeedForwardNetwork       │    │
        │  │  │   └─ RMSNorm                  │    │
        │  │  ├─ Embedding Layer              │    │
        │  │  └─ LM Head                      │    │
        │  └──────────────────────────────────┘    │
        ├───────────────────────────────────────────┤
        │      Layer 3: 算子层 (Operators)          │
        │  ┌──────────────────────────────────┐    │
        │  │ MatMul    │ Softmax │ LayerNorm │    │
        │  │ Embedding │ RoPE    │ SwiGLU    │    │
        │  │ Add/Mul   │ Reshape │ Transpose │    │
        │  └──────────────────────────────────┘    │
        ├───────────────────────────────────────────┤
        │      Layer 4: 张量与内存层                │
        │  ┌──────────────────────────────────┐    │
        │  │ Tensor (MVP)  │ TensorView       │    │
        │  │ std::vector   │ Allocator        │    │
        │  │ KVCacheBuffer │ MemoryMonitor    │    │
        │  └──────────────────────────────────┘    │
        ├───────────────────────────────────────────┤
        │      Layer 5: 优化层 (待开发)             │
        │  ┌──────────────────────────────────┐    │
        │  │ SIMD Kernels (AVX2/AVX-512)      │    │
        │  │ Quantization (INT8/INT4)         │    │
        │  │ Kernel Fusion                    │    │
        │  │ Memory Optimization              │    │
        │  └──────────────────────────────────┘    │
        └───────────────────────────────────────────┘
```

### 1.2 模块依赖关系

```
ModelExecutor (现有)
    │
    ├──> InferenceEngine (接口层)
    │       │
    │       └──> KylinBackend (Kylin 后端)
    │               │
    │               ├──> ModelLoader (扁平 .bin)
    │               ├──> TransformerModel
    │               │       ├──> TransformerBlock (x N)
    │               │       │       ├──> MultiHeadAttention
    │               │       │       │       ├──> RoPE
    │               │       │       │       └──> Kernels (MatMul, Softmax)
    │               │       │       ├──> FeedForwardNetwork
    │               │       │       │       └──> SwiGLU
    │               │       │       └──> RMSNorm
    │               │       ├──> Embedding
    │               │       └──> LMHead
    │               │
    │               ├──> Tensor / TensorView
    │               ├──> MemoryAllocator (未来)
    │               └──> SIMD Kernels (未来)
    │
    ├──> KVCache (复用现有)
    ├──> Sampler (复用现有)
    └──> Tokenizer (复用现有)
```

## 2. 核心组件设计

### 2.1 张量抽象层

#### 2.1.1 Tensor 类

**文件**: `include/cllm/inference/tensor.h`

**实现状态**: ✅ 已完成（MVP 简化版）

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 数据类型枚举（MVP 阶段仅支持 FP32）
 */
enum class DataType {
    FP32,   // 32位浮点（当前实现）
    FP16,   // 16位浮点（待支持）
    INT8,   // 8位整数（待支持）
    INT4    // 4位整数（待支持）
};

/**
 * @brief 设备类型枚举（MVP 阶段仅支持 CPU）
 */
enum class Device {
    CPU,    // CPU（当前实现）
    GPU     // GPU（待支持）
};

/**
 * @brief 简化版张量类
 *
 * MVP 阶段的目标是提供一个足够承载 Transformer 前向计算的最小实现：
 * - 仅支持 float 数据类型
 * - 仅支持 CPU 设备
 * - 以 row-major 方式存储
 * - 形状信息通过 std::vector<size_t> 维护
 */
class Tensor {
public:
    /// 默认构造，得到一个空张量
    Tensor() = default;

    /// 通过形状构造张量
    explicit Tensor(const std::vector<size_t>& shape);

    /// 通过初始化列表构造张量，例如 Tensor({batch, seq, hidden})
    Tensor(std::initializer_list<size_t> shape);

    /// 获取张量形状
    const std::vector<size_t>& shape() const;

    /// 获取维度个数
    size_t ndim() const;

    /// 获取元素总数
    size_t size() const;

    /// 获取数据指针
    float* data();
    const float* data() const;

    /// 按索引访问元素
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    /// 重新设置形状（重新分配内存）
    void resize(const std::vector<size_t>& newShape);

    /// 填充值
    void fill(float value);

    /// 打印张量信息（调试用）
    void print(const std::string& name = "") const;

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;

    void allocate();
};

} // namespace inference
} // namespace cllm
```

**实现示例**：

```cpp
// src/inference/tensor.cpp
namespace cllm {
namespace inference {

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    allocate();
}

Tensor::Tensor(std::initializer_list<size_t> shape) : shape_(shape) {
    allocate();
}

void Tensor::allocate() {
    size_t totalSize = 1;
    for (size_t dim : shape_) {
        totalSize *= dim;
    }
    data_.resize(totalSize, 0.0f);
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::ndim() const {
    return shape_.size();
}

size_t Tensor::size() const {
    return data_.size();
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

void Tensor::resize(const std::vector<size_t>& newShape) {
    shape_ = newShape;
    allocate();
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

} // namespace inference
} // namespace cllm
```

#### 2.1.2 内存管理

**实现状态**: ⚠️ MVP 阶段暂未实现自定义内存管理

**当前方案**：使用 `std::vector` 自动管理内存

**未来规划**：

**文件**: `include/cllm/inference/memory/allocator.h`

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 自定义内存分配器（未来实现）
 *
 * 特性：
 * - 内存池管理，减少分配/释放开销
 * - 内存对齐（64 字节，优化缓存行）
 * - 内存复用，减少碎片
 * - 支持 huge pages（大页内存）
 */
class MemoryAllocator {
public:
    MemoryAllocator(size_t poolSize = 1024 * 1024 * 1024);  // 默认 1GB
    ~MemoryAllocator();
    
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);
    
    size_t getTotalMemory() const;
    size_t getUsedMemory() const;
    size_t getAvailableMemory() const;
    
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
    };
    
    std::vector<MemoryBlock> blocks_;
    size_t poolSize_;
    size_t usedSize_;
};

} // namespace inference
} // namespace cllm
```

### 2.2 模型加载器

#### 2.2.1 ModelLoader

**文件**: `include/cllm/inference/model_loader.h`

**实现状态**: ✅ 已完成（MVP 版本）

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 模型加载器（扁平 .bin 格式）
 *
 * 支持：
 * - 从扁平 .bin 文件加载权重
 * - FP32/FP16/INT8 数据类型
 * - Qwen3 GQA 架构
 */
class ModelLoader {
public:
    explicit ModelLoader(const std::string &binPath);
    ~ModelLoader();

    /**
     * @brief 加载模型权重
     * @param config 模型配置
     * @return 权重字典 {name: Tensor}
     */
    std::map<std::string, Tensor> loadWeights(const ModelConfig &config);

    /**
     * @brief 检查文件是否存在且可读
     */
    bool isValid() const;

    /**
     * @brief 获取文件大小
     */
    size_t getFileSize() const;

private:
    std::string binPath_;
    
    /**
     * @brief 从二进制文件读取数据
     */
    std::vector<float> readBinaryFile(const std::string &path, size_t expectedSize);
    
    /**
     * @brief 权重名称映射（HF 格式 -> 内部格式）
     */
    std::string mapWeightName(const std::string &hfName) const;
};

} // namespace inference
} // namespace cllm
```

**权重文件格式**（扁平 .bin）：

```
文件结构：
- 所有权重按顺序存储为 float32
- 无元数据头，纯二进制数据
- 权重顺序由导出脚本定义

导出脚本：model/export_weights.py
```

**使用流程**：

```cpp
// 1. 创建加载器
ModelLoader loader("/path/to/model.bin");

if (!loader.isValid()) {
    std::cerr << "Invalid model file" << std::endl;
    return false;
}

// 2. 加载权重
ModelConfig config;
config.loadFromJson("/path/to/config.json");

std::map<std::string, Tensor> weights = loader.loadWeights(config);

// 3. 获取权重张量
Tensor embedding = weights["embedding"];
Tensor lmHead = weights["lm_head"];
Tensor wq0 = weights["layer.0.attention.wq"];
// ... 等等
```

#### 2.2.2 GQA 支持

**Grouped Query Attention (GQA)** 特殊处理：

```cpp
// Qwen3 架构中：
// numAttentionHeads = 16
// numKeyValueHeads = 2 (GQA)

// 权重形状：
// wq: [hidden_size, num_attention_heads * head_dim]
// wk: [hidden_size, num_key_value_heads * head_dim]
// wv: [hidden_size, num_key_value_heads * head_dim]

// KV 头需要广播到查询头数量
// 每个 KV 头对应 num_attention_heads / num_key_value_heads 个 Q 头
```

### 2.3 Transformer 核心组件

#### 2.3.1 RMSNorm (Layer Normalization)

**文件**: `include/cllm/inference/layers/rms_norm.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief RMS Normalization
 *
 * 公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
 */
class RMSNorm {
public:
    RMSNorm(size_t hiddenSize, float eps = 1e-6);
    ~RMSNorm();
    
    /**
     * @brief 加载权重
     * @param weight 形状 [hiddenSize]
     */
    void loadWeights(const Tensor& weight);
    
    /**
     * @brief 前向传播
     * @param input 形状 [batch, seq, hidden]
     * @return 形状 [batch, seq, hidden]
     */
    Tensor forward(const Tensor& input);
    
private:
    size_t hiddenSize_;
    float eps_;
    Tensor weight_;  // [hiddenSize]
    
    /**
     * @brief 计算 RMS 归一化
     */
    void computeRMSNorm(
        const float* input,
        float* output,
        size_t batchSize,
        size_t seqLen
    );
};

} // namespace inference
} // namespace cllm
```

**实现原理**:

```cpp
// 步骤：
// 1. 对每个 token 计算 x^2 的均值
//    rms = sqrt(mean(x^2) + eps)
// 2. 归一化：x_norm = x / rms
// 3. 缩放：output = x_norm * weight

void RMSNorm::computeRMSNorm(
    const float* input,
    float* output,
    size_t batchSize,
    size_t seqLen
) {
    const float* weightData = weight_.data();
    
    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t offset = (b * seqLen + s) * hiddenSize_;
            
            // 1. 计算 mean(x^2)
            float sumSquare = 0.0f;
            for (size_t i = 0; i < hiddenSize_; ++i) {
                float val = input[offset + i];
                sumSquare += val * val;
            }
            float meanSquare = sumSquare / hiddenSize_;
            
            // 2. 计算 rms
            float rms = std::sqrt(meanSquare + eps_);
            
            // 3. 归一化并缩放
            for (size_t i = 0; i < hiddenSize_; ++i) {
                output[offset + i] = (input[offset + i] / rms) * weightData[i];
            }
        }
    }
}
```

**优化方向**（未来）：
```cpp
// SIMD 优化（AVX2）
__m256 sum_vec = _mm256_setzero_ps();
for (size_t i = 0; i < hiddenSize_; i += 8) {
    __m256 x = _mm256_loadu_ps(&input[offset + i]);
    sum_vec = _mm256_fmadd_ps(x, x, sum_vec);  // x^2 累加
}
```

#### 2.3.2 RoPE (Rotary Position Embedding)

**文件**: `include/cllm/inference/layers/rope.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief Rotary Position Embedding
 *
 * 将位置信息编码到查询和键中，使用旋转矩阵
 */
class RoPE {
public:
    RoPE(size_t dimPerHead, size_t maxSeqLen, float theta = 10000.0f);
    ~RoPE();
    
    /**
     * @brief 应用 RoPE 到查询和键
     * @param q 查询张量 [batch, num_heads, seq, head_dim]
     * @param k 键张量 [batch, num_kv_heads, seq, head_dim]
     * @param seqLen 序列长度
     * @param posOffset 位置偏移（用于增量生成）
     */
    void apply(
        Tensor& q,
        Tensor& k,
        size_t seqLen,
        size_t posOffset = 0
    );
    
private:
    size_t dimPerHead_;
    size_t maxSeqLen_;
    float theta_;
    
    // 预计算的 cos/sin 表
    Tensor cosCache_;  // [maxSeqLen, dimPerHead/2]
    Tensor sinCache_;  // [maxSeqLen, dimPerHead/2]
    
    /**
     * @brief 预计算频率表
     */
    void precomputeFreqs();
    
    /**
     * @brief 应用旋转
     */
    void applyRotary(
        float* data,
        size_t batchSize,
        size_t numHeads,
        size_t seqLen,
        size_t posOffset
    );
};

} // namespace inference
} // namespace cllm
```

**实现原理**:

```cpp
// 对于每个位置 pos 和维度对 (2i, 2i+1):
// freq = pos / (theta ^ (2i / dim))
// x[2i]'   = x[2i] * cos(freq) - x[2i+1] * sin(freq)
// x[2i+1]' = x[2i] * sin(freq) + x[2i+1] * cos(freq)

void RoPE::precomputeFreqs() {
    cosCache_.resize({maxSeqLen_, dimPerHead_ / 2});
    sinCache_.resize({maxSeqLen_, dimPerHead_ / 2});
    
    float* cosData = cosCache_.data();
    float* sinData = sinCache_.data();
    
    for (size_t pos = 0; pos < maxSeqLen_; ++pos) {
        for (size_t i = 0; i < dimPerHead_ / 2; ++i) {
            float freq = pos / std::pow(theta_, 2.0f * i / dimPerHead_);
            cosData[pos * (dimPerHead_ / 2) + i] = std::cos(freq);
            sinData[pos * (dimPerHead_ / 2) + i] = std::sin(freq);
        }
    }
}

void RoPE::applyRotary(
    float* data,
    size_t batchSize,
    size_t numHeads,
    size_t seqLen,
    size_t posOffset
) {
    const float* cosData = cosCache_.data();
    const float* sinData = sinCache_.data();
    
    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t h = 0; h < numHeads; ++h) {
            for (size_t s = 0; s < seqLen; ++s) {
                size_t pos = posOffset + s;
                size_t offset = ((b * numHeads + h) * seqLen + s) * dimPerHead_;
                
                for (size_t i = 0; i < dimPerHead_ / 2; ++i) {
                    float x0 = data[offset + 2 * i];
                    float x1 = data[offset + 2 * i + 1];
                    
                    float cos_val = cosData[pos * (dimPerHead_ / 2) + i];
                    float sin_val = sinData[pos * (dimPerHead_ / 2) + i];
                    
                    data[offset + 2 * i]     = x0 * cos_val - x1 * sin_val;
                    data[offset + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}
```

#### 2.3.3 Multi-Head Attention

**文件**: `include/cllm/inference/attention.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 多头注意力机制
 *
 * 支持：
 * - Multi-Head Attention (MHA)
 * - Grouped Query Attention (GQA)
 * - KV Cache
 * - RoPE 位置编码
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(
        size_t hiddenSize,
        size_t numHeads,
        size_t numKVHeads,  // 用于 GQA
        size_t maxSeqLen,
        float ropeTheta = 10000.0f
    );
    ~MultiHeadAttention();
    
    /**
     * @brief 加载权重
     */
    void loadWeights(
        const Tensor& wq,  // [hidden, num_heads * head_dim]
        const Tensor& wk,  // [hidden, num_kv_heads * head_dim]
        const Tensor& wv,  // [hidden, num_kv_heads * head_dim]
        const Tensor& wo   // [num_heads * head_dim, hidden]
    );
    
    /**
     * @brief 前向传播（带 KV Cache）
     * @param input 形状 [batch, seq, hidden]
     * @param kCache KV Cache，形状 [batch, num_kv_heads, max_seq, head_dim]
     * @param vCache KV Cache，形状 [batch, num_kv_heads, max_seq, head_dim]
     * @param layerIdx 层索引
     * @param seqLen 当前序列长度
     * @param posOffset 位置偏移（增量生成时使用）
     * @return 形状 [batch, seq, hidden]
     */
    Tensor forward(
        const Tensor& input,
        Tensor* kCache,
        Tensor* vCache,
        size_t layerIdx,
        size_t seqLen,
        size_t posOffset
    );
    
private:
    size_t hiddenSize_;
    size_t numHeads_;
    size_t numKVHeads_;
    size_t headDim_;
    size_t maxSeqLen_;
    
    Tensor wq_;  // [hidden, num_heads * head_dim]
    Tensor wk_;  // [hidden, num_kv_heads * head_dim]
    Tensor wv_;  // [hidden, num_kv_heads * head_dim]
    Tensor wo_;  // [num_heads * head_dim, hidden]
    
    std::unique_ptr<RoPE> rope_;
    
    /**
     * @brief QKV 投影
     */
    void projectQKV(
        const Tensor& input,
        Tensor& q,
        Tensor& k,
        Tensor& v
    );
    
    /**
     * @brief 计算注意力
     */
    Tensor computeAttention(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        size_t seqLen
    );
    
    /**
     * @brief 输出投影
     */
    Tensor projectOutput(const Tensor& attnOut);
    
    /**
     * @brief GQA 广播（将 KV 头广播到 Q 头数量）
     */
    void broadcastKVForGQA(Tensor& k, Tensor& v);
};

} // namespace inference
} // namespace cllm
```

**Attention 计算流程**:

```cpp
Tensor MultiHeadAttention::forward(
    const Tensor& input,  // [batch, seq, hidden]
    Tensor* kCache,
    Tensor* vCache,
    size_t layerIdx,
    size_t seqLen,
    size_t posOffset
) {
    const size_t batchSize = input.shape()[0];
    
    // 1. QKV 投影
    Tensor q({batchSize, seqLen, numHeads_ * headDim_});
    Tensor k({batchSize, seqLen, numKVHeads_ * headDim_});
    Tensor v({batchSize, seqLen, numKVHeads_ * headDim_});
    projectQKV(input, q, k, v);
    
    // 2. 重塑为多头
    // Q: [batch, num_heads, seq, head_dim]
    // K: [batch, num_kv_heads, seq, head_dim]
    // V: [batch, num_kv_heads, seq, head_dim]
    
    // 3. 应用 RoPE
    rope_->apply(q, k, seqLen, posOffset);
    
    // 4. 更新 KV Cache
    // kCache[batch, num_kv_heads, posOffset:posOffset+seqLen, head_dim] = k
    // vCache[batch, num_kv_heads, posOffset:posOffset+seqLen, head_dim] = v
    
    // 5. GQA 广播（如果需要）
    if (numKVHeads_ != numHeads_) {
        broadcastKVForGQA(k, v);
    }
    
    // 6. 计算 Attention
    Tensor attnOut = computeAttention(q, k, v, seqLen);
    
    // 7. 输出投影
    Tensor output = projectOutput(attnOut);
    
    return output;
}

Tensor MultiHeadAttention::computeAttention(
    const Tensor& q,    // [batch, num_heads, seq, head_dim]
    const Tensor& k,    // [batch, num_heads, ctx, head_dim]
    const Tensor& v,    // [batch, num_heads, ctx, head_dim]
    size_t seqLen
) {
    // scores = (Q @ K^T) / sqrt(head_dim)
    // scores shape: [batch, num_heads, seq, ctx]
    
    // 应用因果 mask
    // mask[i, j] = -inf if j > i else 0
    
    // attn_weights = softmax(scores)
    // attn_out = attn_weights @ V
    // attn_out shape: [batch, num_heads, seq, head_dim]
    
    // 重塑为 [batch, seq, num_heads * head_dim]
}
```

#### 2.3.4 Feed-Forward Network (SwiGLU)

**文件**: `include/cllm/inference/layers/feed_forward.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 前馈神经网络（SwiGLU 激活）
 *
 * 公式: FFN(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down
 */
class FeedForwardNetwork {
public:
    FeedForwardNetwork(size_t hiddenSize, size_t intermediateSize);
    ~FeedForwardNetwork();
    
    /**
     * @brief 加载权重
     */
    void loadWeights(
        const Tensor& wGate,  // [hidden, intermediate]
        const Tensor& wUp,    // [hidden, intermediate]
        const Tensor& wDown   // [intermediate, hidden]
    );
    
    /**
     * @brief 前向传播
     * @param input 形状 [batch, seq, hidden]
     * @return 形状 [batch, seq, hidden]
     */
    Tensor forward(const Tensor& input);
    
private:
    size_t hiddenSize_;
    size_t intermediateSize_;
    
    Tensor wGate_;  // [hidden, intermediate]
    Tensor wUp_;    // [hidden, intermediate]
    Tensor wDown_;  // [intermediate, hidden]
    
    /**
     * @brief SwiGLU 激活函数
     * 
     * SwiGLU(gate, up) = gate * SiLU(up)
     * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     */
    Tensor swiGLU(const Tensor& gate, const Tensor& up);
};

} // namespace inference
} // namespace cllm
```

**实现示例**:

```cpp
Tensor FeedForwardNetwork::forward(const Tensor& input) {
    // 1. gate = input @ W_gate
    Tensor gate = matmul(input, wGate_);
    
    // 2. up = input @ W_up
    Tensor up = matmul(input, wUp_);
    
    // 3. activated = gate * SiLU(up)
    Tensor activated = swiGLU(gate, up);
    
    // 4. output = activated @ W_down
    Tensor output = matmul(activated, wDown_);
    
    return output;
}

Tensor FeedForwardNetwork::swiGLU(const Tensor& gate, const Tensor& up) {
    Tensor result(gate.shape());
    float* dst = result.data();
    const float* gateData = gate.data();
    const float* upData = up.data();
    
    for (size_t i = 0; i < gate.size(); ++i) {
        float x = upData[i];
        float silu = x / (1.0f + std::exp(-x));  // SiLU(x)
        dst[i] = gateData[i] * silu;
    }
    
    return result;
}
```

#### 2.3.5 Transformer Block

**文件**: `include/cllm/inference/transformer_block.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief Transformer 块（Pre-Norm 架构）
 *
 * 结构：
 * x = x + Attention(RMSNorm(x))
 * x = x + FFN(RMSNorm(x))
 */
class TransformerBlock {
public:
    TransformerBlock(
        size_t hiddenSize,
        size_t numHeads,
        size_t numKVHeads,
        size_t intermediateSize,
        size_t maxSeqLen,
        float rmsNormEps,
        float ropeTheta
    );
    ~TransformerBlock();
    
    /**
     * @brief 加载权重
     */
    void loadWeights(const std::map<std::string, Tensor>& weights);
    
    /**
     * @brief 前向传播
     */
    Tensor forward(
        const Tensor& input,
        Tensor* kCache,
        Tensor* vCache,
        size_t layerIdx,
        size_t seqLen,
        size_t posOffset
    );
    
private:
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForwardNetwork> ffn_;
    std::unique_ptr<RMSNorm> inputNorm_;
    std::unique_ptr<RMSNorm> postAttnNorm_;
};

} // namespace inference
} // namespace cllm
```

**实现示例**:

```cpp
Tensor TransformerBlock::forward(
    const Tensor& input,
    Tensor* kCache,
    Tensor* vCache,
    size_t layerIdx,
    size_t seqLen,
    size_t posOffset
) {
    // 1. Pre-Norm Attention
    Tensor residual = input;
    Tensor x = inputNorm_->forward(input);
    x = attention_->forward(x, kCache, vCache, layerIdx, seqLen, posOffset);
    x = add(x, residual);  // 残差连接
    
    // 2. Pre-Norm FFN
    residual = x;
    x = postAttnNorm_->forward(x);
    x = ffn_->forward(x);
    x = add(x, residual);  // 残差连接
    
    return x;
}
```

#### 2.3.6 完整 Transformer 模型

**文件**: `include/cllm/inference/transformer_model.h`

**实现状态**: 🚧 进行中

```cpp
namespace cllm {
namespace inference {

/**
 * @brief 完整的 Transformer 模型
 */
class TransformerModel {
public:
    explicit TransformerModel(const ModelConfig& config);
    ~TransformerModel();
    
    /**
     * @brief 加载所有权重
     */
    void loadWeights(const std::map<std::string, Tensor>& weights);
    
    /**
     * @brief 前向传播
     * @param inputIds token id 序列
     * @param kCache KV Cache（可选，用于增量生成）
     * @param vCache KV Cache（可选）
     * @param posOffset 位置偏移
     * @return [seq_len, vocab_size] logits
     */
    Tensor forward(
        const std::vector<int>& inputIds,
        Tensor* kCache = nullptr,
        Tensor* vCache = nullptr,
        size_t posOffset = 0
    );
    
    ModelConfig getConfig() const { return config_; }
    
private:
    ModelConfig config_;
    
    // 组件
    Tensor embedding_;  // [vocab_size, hidden_size]
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    std::unique_ptr<RMSNorm> finalNorm_;
    Tensor lmHead_;  // [hidden_size, vocab_size]
    
    /**
     * @brief Embedding 查表
     */
    Tensor embed(const std::vector<int>& inputIds);
    
    /**
     * @brief 生成 logits
     */
    Tensor generateLogits(const Tensor& hiddenStates);
};

} // namespace inference
} // namespace cllm
```

**实现示例**:

```cpp
Tensor TransformerModel::forward(
    const std::vector<int>& inputIds,
    Tensor* kCache,
    Tensor* vCache,
    size_t posOffset
) {
    const size_t seqLen = inputIds.size();
    
    // 1. Embedding
    Tensor x = embed(inputIds);  // [seq_len, hidden_size]
    
    // 2. Transformer Layers
    for (size_t i = 0; i < config_.numLayers; ++i) {
        x = layers_[i]->forward(x, kCache, vCache, i, seqLen, posOffset);
    }
    
    // 3. Final Norm
    x = finalNorm_->forward(x);
    
    // 4. LM Head
    Tensor logits = generateLogits(x);  // [seq_len, vocab_size]
    
    return logits;
}

Tensor TransformerModel::embed(const std::vector<int>& inputIds) {
    const size_t seqLen = inputIds.size();
    Tensor result({seqLen, config_.hiddenSize});
    
    float* dst = result.data();
    const float* embData = embedding_.data();
    
    for (size_t i = 0; i < seqLen; ++i) {
        int tokenId = inputIds[i];
        size_t srcOffset = tokenId * config_.hiddenSize;
        size_t dstOffset = i * config_.hiddenSize;
        
        std::memcpy(dst + dstOffset, embData + srcOffset, 
                    config_.hiddenSize * sizeof(float));
    }
    
    return result;
}

Tensor TransformerModel::generateLogits(const Tensor& hiddenStates) {
    // logits = hiddenStates @ lmHead^T
    // 形状：[seq_len, hidden_size] @ [vocab_size, hidden_size]^T
    //     = [seq_len, vocab_size]
    
    return matmul(hiddenStates, lmHead_, false, true);  // transpose B
}
```

### 2.4 高性能算子

#### 2.4.1 矩阵乘法 (GEMM)

**文件**: `include/cllm/inference/kernels/matmul.h`

**实现状态**: ⏳ 待开发（当前使用朴素实现）

```cpp
namespace cllm {
namespace inference {
namespace kernels {

/**
 * @brief 通用矩阵乘法接口
 * 
 * C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 */
void matmul(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K,
    bool transposeA = false,
    bool transposeB = false
);

/**
 * @brief SIMD 优化版本（AVX2）
 */
void matmul_avx2(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);

/**
 * @brief SIMD 优化版本（AVX-512）
 */
void matmul_avx512(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);

/**
 * @brief 量化矩阵乘法（INT8）
 */
void matmul_int8(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    size_t M,
    size_t N,
    size_t K,
    const float* scaleA,
    const float* scaleB
);

} // namespace kernels
} // namespace inference
} // namespace cllm
```

**朴素实现（当前）**:

```cpp
void matmul(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K,
    bool transposeA,
    bool transposeB
) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                size_t aIdx = transposeA ? (k * M + i) : (i * K + k);
                size_t bIdx = transposeB ? (j * K + k) : (k * N + j);
                sum += A[aIdx] * B[bIdx];
            }
            C[i * N + j] = sum;
        }
    }
}
```

**SIMD 优化版本（未来）**:

```cpp
void matmul_avx2(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += 8) {  // 8个float = 256位
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t k = 0; k < K; ++k) {
                __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                sum = _mm256_fmadd_ps(a, b, sum);  // FMA: a * b + sum
            }
            
            _mm256_storeu_ps(&C[i * N + j], sum);
        }
    }
}
```

**优化策略**:
1. **向量化**：使用 AVX2/AVX-512 一次处理 8/16 个浮点数
2. **分块 (Tiling)**：提高缓存命中率
3. **数据预取 (Prefetching)**：减少内存延迟
4. **循环展开 (Loop Unrolling)**：减少循环开销

#### 2.4.2 Softmax

**文件**: `include/cllm/inference/kernels/softmax.h`

**实现状态**: ⏳ 待开发（当前使用朴素实现）

```cpp
namespace cllm {
namespace inference {
namespace kernels {

/**
 * @brief Softmax 激活函数
 * 
 * 输入: [outer_dim, inner_dim]
 * 对最后一个维度应用 softmax
 */
void softmax(
    const float* input,
    float* output,
    size_t outerDim,
    size_t innerDim
);

/**
 * @brief 数值稳定版本
 */
void softmax_stable(
    const float* input,
    float* output,
    size_t outerDim,
    size_t innerDim
);

} // namespace kernels
} // namespace inference
} // namespace cllm
```

**数值稳定实现**:

```cpp
void softmax_stable(
    const float* input,
    float* output,
    size_t outerDim,
    size_t innerDim
) {
    for (size_t i = 0; i < outerDim; ++i) {
        const float* inRow = input + i * innerDim;
        float* outRow = output + i * innerDim;
        
        // 1. 找最大值（避免溢出）
        float maxVal = inRow[0];
        for (size_t j = 1; j < innerDim; ++j) {
            maxVal = std::max(maxVal, inRow[j]);
        }
        
        // 2. 计算 exp(x - max) 和 sum
        float sum = 0.0f;
        for (size_t j = 0; j < innerDim; ++j) {
            outRow[j] = std::exp(inRow[j] - maxVal);
            sum += outRow[j];
        }
        
        // 3. 归一化
        for (size_t j = 0; j < innerDim; ++j) {
            outRow[j] /= sum;
        }
    }
}
```

### 2.5 量化支持

#### 2.5.1 量化方案

**文件**: `include/cllm/inference/quantization.h`

**实现状态**: ⏳ 待开发

```cpp
namespace cllm {
namespace inference {

enum class QuantizationType {
    NONE,
    INT8,    // 8位整数量化
    INT4,    // 4位整数量化
    FP16     // 半精度浮点
};

/**
 * @brief 量化器
 */
class Quantizer {
public:
    explicit Quantizer(QuantizationType type);
    ~Quantizer();
    
    /**
     * @brief 对称量化
     * 
     * scale = max(abs(x)) / 127
     * x_quant = round(x / scale)
     */
    void quantize_symmetric(
        const float* input,
        int8_t* output,
        float* scale,
        size_t size
    );
    
    /**
     * @brief 非对称量化
     * 
     * scale = (max(x) - min(x)) / 255
     * zero_point = round(-min(x) / scale)
     * x_quant = round(x / scale) + zero_point
     */
    void quantize_asymmetric(
        const float* input,
        int8_t* output,
        float* scale,
        int8_t* zeroPoint,
        size_t size
    );
    
    /**
     * @brief 反量化
     */
    void dequantize(
        const int8_t* input,
        float* output,
        const float* scale,
        size_t size
    );
    
    /**
     * @brief 量化权重张量
     */
    Tensor quantizeWeights(const Tensor& weights);
    
private:
    QuantizationType type_;
    
    float computeScale(const float* data, size_t size);
    int8_t computeZeroPoint(const float* data, size_t size, float scale);
};

} // namespace inference
} // namespace cllm
```

## 3. 性能优化路线图

### 3.1 当前状态（MVP）

| 组件 | 实现状态 | 性能 |
|------|---------|------|
| Tensor | ✅ FP32, CPU | 基准 |
| MatMul | 朴素实现 | 慢 (~20 GFLOPS) |
| Softmax | 朴素实现 | 中等 |
| RMSNorm | 朴素实现 | 中等 |
| Attention | 朴素实现 | 慢 |
| 内存管理 | std::vector | 中等 |

### 3.2 短期优化（2-4周）

1. **SIMD 优化**:
   - ✅ 检测 CPU 指令集（AVX2/AVX-512）
   - 🚧 MatMul AVX2 实现
   - 🚧 Softmax AVX2 实现
   - 🚧 RMSNorm AVX2 实现

2. **内存优化**:
   - 🚧 内存池管理
   - 🚧 64字节内存对齐
   - 🚧 减少临时分配

3. **算子融合**:
   - 🚧 RMSNorm + MatMul 融合
   - 🚧 MatMul + Add 融合

**预期提升**：2-3x 加速

### 3.3 中期优化（1-2个月）

1. **量化支持**:
   - INT8 量化推理
   - 混合精度（FP16/INT8）
   - 权重量化 + 激活量化

2. **高级算子**:
   - Flash Attention 实现
   - 分块 MatMul（Tiling）
   - 预取优化

3. **并行优化**:
   - OpenMP 多线程
   - Token-level 并行
   - Layer-level 流水线

**预期提升**：5-8x 加速（相对MVP）

### 3.4 长期优化（3-6个月）

1. **极致性能**:
   - AVX-512 全覆盖
   - INT4 量化
   - 自定义算子库

2. **新硬件支持**:
   - ARM NEON（移动端）
   - AMD ZEN 优化
   - 探索 GPU 支持

3. **高级特性**:
   - 模型并行
   - Pipeline 并行
   - 动态形状支持

**预期提升**：10-15x 加速（相对 MVP），接近 llama.cpp 性能

## 4. 与 LibTorch 后端对比

| 特性 | Kylin Backend | LibTorch Backend |
|------|---------------|------------------|
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **开发速度** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPU 性能（未优化）** | ⭐⭐⭐ | ⭐⭐⭐ |
| **CPU 性能（优化后）** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **量化支持** | ⭐⭐⭐⭐⭐（未来） | ⭐⭐⭐⭐ |
| **可定制性** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **内存占用（未优化）** | ⭐⭐⭐ | ⭐⭐⭐ |
| **内存占用（优化后）** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **二进制大小** | ⭐⭐⭐⭐⭐（小） | ⭐⭐（大） |
| **GPU 支持** | ⏳ 待开发 | ⭐⭐⭐⭐⭐ |

## 5. 开发指南

### 5.1 编译和测试

```bash
# 编译 Kylin 后端
cd cpp/cLLM
make clean
make

# 运行测试
./build/tests/test_tensor
./build/tests/test_attention
./build/tests/test_transformer

# 性能测试
./build/bin/cllm_benchmark --backend kylin
```

### 5.2 使用示例

```cpp
#include "cllm/inference/inference_engine.h"

using namespace cllm::inference;

// 创建 Kylin 后端引擎
ModelConfig config;
config.loadFromJson("/path/to/config.json");

InferenceEngine engine(
    config,
    "/path/to/model.bin",  // 扁平 .bin 权重
    false  // 使用 Kylin 后端
);

if (!engine.initialize()) {
    std::cerr << "Failed to initialize Kylin engine" << std::endl;
    return -1;
}

// 推理
std::vector<int> inputIds = {1, 72, 105, 2};  // "Hi"
Tensor logits = engine.forward(inputIds);

std::cout << "Logits shape: [" << logits.shape()[0] 
          << ", " << logits.shape()[1] << "]" << std::endl;
```

## 6. 参考文档

- [推理引擎接口设计.md](推理引擎接口设计.md) - 统一接口层定义
- [LibTorch后端设计.md](LibTorch后端设计.md) - LibTorch 后端实现
- [C++编程规范.md](C++编程规范.md) - 编码规范
- [lesson/5.模型执行器的原理.md](/lesson/5.模型执行器的原理.md) - 执行器原理
- [lesson/7.前向传播优化技术.md](/lesson/7.前向传播优化技术.md) - 优化技术

## 7. 总结

**Kylin (麒麟) 推理引擎**是 cLLM 的自研高性能后端，专注于：

✅ **纯 C++ 实现**：无外部依赖，易于部署  
✅ **极致 CPU 性能**：SIMD 优化、量化支持  
✅ **模块化设计**：易于扩展和定制  
✅ **完全可控**：从算子到优化策略全掌控  

🚧 **当前状态**：MVP 阶段，基础功能完成  
🎯 **未来目标**：通过 SIMD 和量化优化，达到 llama.cpp 级别性能  
🎨 **设计理念**：先实现、后优化，逐步演进

通过 Kylin 后端，我们实现了：
- **技术自主**：完全掌握推理引擎核心技术
- **性能可控**：根据需求定制优化策略
- **长期价值**：为未来的硬件和算法演进打下基础
