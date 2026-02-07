# HF 格式性能优化指南

## 概述

本文档针对 cLLM 的 Kylin 引擎中 HuggingFace (HF) 原生格式（safetensors）的 CPU/GPU 推理性能优化提供系统性分析和优化方案。

**目标**：在不增加内存开销的前提下，实现 HF 格式的极致性能优化。

---

## 1. 现状分析

### 1.1 当前架构

```
HF Safetensors → BF16 权重 → 运行时转换 → FP32/FP16/INT8 → 推理
```

### 1.2 性能瓶颈识别

#### 1.2.1 CPU 推理瓶颈

| 瓶颈点 | 影响 | 严重程度 | 代码位置 |
|--------|------|----------|----------|
| **BF16→FP32 转换** | 每次推理都要转换，耗时 10-20% | 🔴 高 | `ggml_kernels.cpp:200-220` |
| **矩阵乘法未分块** | 缓存命中率低，未用 BLAS | � 中 | `ggml_kernels.cpp:300+` |
| **重复内存分配** | 推理时动态分配缓冲区 | 🟡 中 | `transformer.cpp:100-125` |
| **SIMD 已实现但可优化** | AVX2/NEON 已实现，但可用 BLAS 替代 | � 中 | `ggml_kernels.cpp:250+` |
| **OpenMP 并行已启用** | 已使用，但阈值设置可能不合理 | � 低 | `ggml_kernels.cpp:30-35` |

#### 1.2.2 GPU 推理瓶颈

| 瓶颈点 | 影响 | 严重程度 |
|--------|------|----------|
| **CPU-GPU 数据传输** | 每 token 都要传输，带宽瓶颈 | 🔴 高 |
| **计算图重建** | 每次推理重建 GGML 图 | 🔴 高 |
| **同步等待** | CPU 等待 GPU 完成 | 🟡 中 |
| **小批量利用率低** | batch=1 时 GPU 利用率不足 | 🟡 中 |

#### 1.2.3 内存使用问题

```cpp
// 当前内存分配模式（问题示例）
// src/kylin/hf/transformer.cpp:81-125
ropeFreqsCos_.resize(kMaxSeqLen * headDim / 2);      // 预分配 ✅
kCache_.resize(kvSize, 0.0f);                        // 预分配 ✅
hiddenStates_.resize(config_.hiddenSize);            // 单线程 ❌
qkvBuffer_.resize(qSize + 2 * kvSize2);              // 单线程 ❌
// ... 更多单线程缓冲区
```

**问题**：
- 工作缓冲区是成员变量，多线程需要竞争或复制
- 每个请求需要独立的 KV Cache，但预分配策略不够灵活
- 权重存储多份（BF16 + FP32/FP16/INT8）

---

## 2. 优化策略

### 2.1 核心原则

1. **零额外内存**：不增加内存占用，通过算法优化提升性能
2. **延迟最小化**：减少 CPU-GPU 数据传输和同步
3. **计算并行化**：充分利用多核 CPU 和 GPU 并行能力
4. **内存局部性**：优化数据布局，提高缓存命中率

### 2.2 CPU 优化方案

#### 2.2.1 矩阵乘法优化（最高优先级）

**现状分析**：

```cpp
// 当前实现（ggml_kernels.cpp:300+）
// 已实现 SIMD + OpenMP，但仍有优化空间：
// 1. 使用 Apple Accelerate BLAS（已启用）
// 2. 但未使用分块（Tiling）策略
// 3. 矩阵-向量乘法为主，未优化矩阵-矩阵乘法

void matmul_f32(const float* weight, const float* input,
                float* output, int M, int K) {
    // 当前：行主序遍历 + SIMD
    #pragma omp parallel for if(useParallel)
    for (int m = 0; m < M; ++m) {
        // AVX2/NEON 点积计算
        output[m] = dot_product_simd(weight + m*K, input, K);
    }
}
```

**优化方案 1: 使用 BLAS 库（推荐）**

```cpp
// 利用 Apple Accelerate (vecLib) 或 OpenBLAS
// ggml_kernels.cpp 修改：

#if USE_BLAS
void matmul_f32_blas(const float* A, const float* B, float* C, 
                     int M, int N, int K) {
    // cblas_sgemm: C = A * B
    // 对于矩阵-向量: cblas_sgemv
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                M, K, 1.0f, A, K, B, 1, 0.0f, C, 1);
}
#endif
```

**优化方案 2: 分块 + SIMD（如果不用 BLAS）**

```cpp
// 分块矩阵乘法，提高 L1/L2 缓存命中率
void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    constexpr int BM = 64;  // L1 缓存可容纳
    constexpr int BN = 64;
    constexpr int BK = 256;
    
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += BM) {
        for (int j0 = 0; j0 < N; j0 += BN) {
            // 处理 BM x BN 块
            for (int k0 = 0; k0 < K; k0 += BK) {
                simd_block_mul(A, B, C, i0, j0, k0, 
                              std::min(BM, M-i0),
                              std::min(BN, N-j0),
                              std::min(BK, K-k0));
            }
        }
    }
}
```
```

**实现步骤**：
1. 使用 OpenMP 并行化外层循环
2. 实现分块（Tiling）策略，优化 L1/L2 缓存使用
3. 使用 SIMD 指令（AVX2/AVX-512/NEON）加速内层计算
4. 考虑使用 BLAS 库（OpenBLAS/MKL）替代自研实现

#### 2.2.2 BF16 快速转换优化

**现状**：已实现 SIMD 优化 ✅

```cpp
// 当前实现（ggml_kernels.cpp:200-220）
// 已实现 AVX2/NEON SIMD 优化

void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t count) {
#if USE_AVX2
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i bf16 = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i bf16_32 = _mm256_cvtepu16_epi32(bf16);
        __m256 f32 = _mm256_castsi256_ps(_mm256_slli_epi32(bf16_32, 16));
        _mm256_storeu_ps(dst + i, f32);
    }
    // 处理剩余...
#elif USE_NEON
    // NEON 实现...
#endif
}
```

**状态**：✅ **已完成** - 已实现 AVX2/NEON SIMD 优化

**进一步优化建议**：
- 考虑使用 Intel AMX 指令（BF16 原生支持）
- 对于大批量转换，可使用多线程并行

#### 2.2.3 注意力机制优化

**现状**：标准 Attention 实现

```cpp
// Q @ K^T / sqrt(d_k)
// softmax
// @ V
```

**优化方案**：

1. **Flash Attention 算法**
   - 分块计算，减少 HBM 访问
   - 在线 softmax，避免存储完整 attention 矩阵
   - 适合长序列（>1024 tokens）

2. **GQA (Grouped Query Attention) 优化**
   - 共享 K/V，减少内存带宽
   - 已部分实现，可进一步优化

```cpp
// Flash Attention 伪代码
void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     int seq_len, int head_dim, int block_size) {
    for (int i = 0; i < seq_len; i += block_size) {
        for (int j = 0; j < seq_len; j += block_size) {
            // 加载 Q[i:i+block], K[j:j+block], V[j:j+block] 到 SRAM
            // 计算局部 attention
            // 在线 softmax 更新
        }
    }
}
```

#### 2.2.4 内存池优化

**现状**：KV Cache Pool 和 WorkBuffer Pool 已实现，但可进一步优化

```cpp
// 当前：每个请求独立分配
KVCacheSlot* slot = kvCachePool_->allocate(requestId);
WorkBufferSlot* workBuf = workBufferPool_->allocate();
```

**优化方案**：

1. **内存对齐**：确保所有缓冲区 64 字节对齐（缓存行大小）
2. **NUMA 感知**：在多路 CPU 上，将内存分配到使用它的 NUMA 节点
3. **Huge Pages**：使用大页减少 TLB miss

```cpp
// 对齐分配
void* aligned_malloc(size_t size, size_t alignment = 64) {
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}
```

### 2.3 GPU 优化方案

#### 2.3.1 计算图缓存（最高优先级）

**现状**：当前实现使用 CPU 回退，未充分利用 GPU

```cpp
// 当前实现（ggml_backend.cpp:380+）
// 实际是 CPU 路径，权重已缓存到 CPU
std::vector<float> GGMLGPUBackend::forward(int tokenId, int position) {
    // 1. 从 CPU 缓存获取 embedding
    // 2. 使用 cpuMatmul (BLAS/SIMD) 进行矩阵乘法
    // 3. 所有计算在 CPU 上完成
}
```

**问题分析**：
1. ✅ 权重已缓存到 CPU (`weightsCached_`)
2. ✅ 使用 BLAS/SIMD 优化的 `cpuMatmul`
3. ❌ **未真正使用 GPU** - 当前实现是 CPU 路径
4. ❌ 每次推理都有内存拷贝开销

**优化方案 1: 真正的 GPU 计算图（Metal/CUDA）**

```cpp
// 创建持久化的 GGML 计算图
class GGMLGPUBackend {
    // 当前已有：
    ggml_backend_t backend_;      // Metal/CUDA backend
    ggml_backend_sched_t graphSched_;  // 调度器
    
    // 需要添加：
    struct CachedGraph {
        ggml_cgraph* graph = nullptr;
        ggml_context* ctx = nullptr;
        int max_seq_len = 0;
        bool initialized = false;
    };
    std::vector<CachedGraph> layer_graphs_;  // 每层一个图
    
    // 预构建 Transformer Layer 的计算图
    bool buildLayerGraph(int layer_idx, int max_seq_len) {
        CachedGraph& cg = layer_graphs_[layer_idx];
        
        // 创建上下文
        struct ggml_init_params params = {
            .mem_size = 1024 * 1024 * 10,  // 10MB 图内存
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        cg.ctx = ggml_init(params);
        
        // 创建输入张量（占位符）
        ggml_tensor* input = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_F32, hiddenSize);
        ggml_tensor* position = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, 1);
        
        // 构建计算图：RMSNorm -> QKV Proj -> RoPE -> Attention -> FFN
        ggml_tensor* norm_out = ggml_rms_norm(cg.ctx, input, eps);
        ggml_tensor* q = ggml_mul_mat(cg.ctx, q_proj_weight, norm_out);
        ggml_tensor* k = ggml_mul_mat(cg.ctx, k_proj_weight, norm_out);
        ggml_tensor* v = ggml_mul_mat(cg.ctx, v_proj_weight, norm_out);
        
        // RoPE 位置编码
        q = ggml_rope(cg.ctx, q, position, head_dim, 10000.0f);
        k = ggml_rope(cg.ctx, k, position, head_dim, 10000.0f);
        
        // Attention: Q @ K^T / sqrt(d_k)
        ggml_tensor* attn_weights = ggml_soft_max(cg.ctx,
            ggml_scale(cg.ctx, ggml_mul_mat(cg.ctx, q, k), 1.0f / sqrt(head_dim)));
        
        // @ V
        ggml_tensor* attn_out = ggml_mul_mat(cg.ctx, attn_weights, v);
        
        // O Projection
        ggml_tensor* output = ggml_mul_mat(cg.ctx, o_proj_weight, attn_out);
        
        // 构建图
        cg.graph = ggml_new_graph(cg.ctx);
        ggml_build_forward_expand(cg.graph, output);
        
        // 分配后端缓冲区
        ggml_backend_alloc_ctx_tensors(cg.ctx, backend_);
        
        cg.max_seq_len = max_seq_len;
        cg.initialized = true;
        
        return true;
    }
};
```

**优化方案 2: 使用 GGML 的图调度器**

```cpp
// 利用 GGML 的自动调度功能
bool forwardGPU(int tokenId, int position) {
    // 1. 准备输入张量
    ggml_tensor* input = ggml_new_tensor_1d(computeCtx_, GGML_TYPE_F32, hiddenSize);
    
    // 2. 复制 embedding 到 GPU
    const float* embed = weightsGPU_["embed_tokens"] + tokenId * hiddenSize;
    ggml_backend_tensor_set(backend_, input, embed, hiddenSize * sizeof(float));
    
    // 3. 使用调度器自动分配计算到 GPU/CPU
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends_.data(), bufts_.data(), backends_.size(), 
        1024 * 1024 * 100  // 100MB 调度缓冲区
    );
    
    // 4. 设置评估回调，监控 GPU 利用率
    ggml_backend_sched_set_eval_callback(sched, sched_eval_cb, &stats);
    
    // 5. 执行计算
    ggml_backend_sched_graph_compute(sched, gf);
    
    // 6. 获取结果
    ggml_backend_tensor_get(backend_, output, result.data(), output_size);
}
```

**预期提升**：
- 真正的 GPU 加速：5-10x 速度提升（vs CPU）
- 图缓存：减少 30-50% 的准备时间
- 批处理：多序列并行时效率更高

#### 2.3.2 异步数据传输

**现状**：同步传输，CPU 等待 GPU

```cpp
// 当前：同步传输
memcpy(cpu_buffer, gpu_buffer, size);  // 阻塞
```

**优化方案**：

```cpp
// 方案：双缓冲 + 异步传输
class AsyncTransfer {
    void* buffer_[2];  // 双缓冲
    int current_ = 0;
    
    void* get_buffer() { return buffer_[current_]; }
    
    void swap_and_transfer() {
        int next = 1 - current_;
        // 异步启动传输到 buffer_[next]
        cudaMemcpyAsync(buffer_[next], gpu_buffer, size, cudaMemcpyDeviceToHost, stream_);
        current_ = next;
    }
};
```

#### 2.3.3 批处理优化

**现状**：单序列推理，GPU 利用率低

**优化方案**：

1. **动态批处理**：将多个请求合并成一个 batch
2. **连续批处理（Continuous Batching）**：新请求随时加入当前 batch

```cpp
// 连续批处理
void continuous_batching() {
    while (has_active_requests()) {
        // 收集所有需要 forward 的请求
        std::vector<Request*> batch = collect_ready_requests(max_batch_size);
        
        // 打包成 batch tensor
        BatchTensors tensors = pack_batch(batch);
        
        // 单次 GPU 推理
        forward_batch(tensors);
        
        // 分发结果
        for (auto* req : batch) {
            req->update_state();
        }
    }
}
```

#### 2.3.4 混合精度推理

**现状**：FP32 推理，未利用 Tensor Core

**优化方案**：

```cpp
// 使用 FP16/BF16 混合精度
// 权重：FP16
// 激活：FP16
// 累加：FP32（避免精度损失）

#ifdef GGML_USE_METAL
// Metal 自动使用混合精度
#endif
```

### 2.4 内存优化方案

#### 2.4.1 权重共享策略

**现状**：同时存储 BF16（原始）+ FP32/FP16/INT8（转换后）

**优化方案**：

```cpp
// 方案：按需加载，不保留原始 BF16
class WeightManager {
    enum class Format { BF16, FP32, FP16, INT8 };
    
    // 只保留一种格式
    Format current_format_;
    std::vector<uint8_t> weight_data_;
    
    void convert_to(Format target) {
        if (current_format_ == target) return;
        
        // 原地转换，不分配新内存
        convert_inplace(weight_data_, current_format_, target);
        current_format_ = target;
    }
};
```

#### 2.4.2 内存映射加载

**现状**：将整个模型加载到内存

**优化方案**：

```cpp
// 使用 mmap 延迟加载
class MmappedWeights {
    int fd_;
    void* mapped_;
    size_t size_;
    
    void* get_weight(const std::string& name) {
        // 按需从磁盘加载，OS 自动管理缓存
        return (char*)mapped_ + offset_table_[name];
    }
};
```

#### 2.4.3 KV Cache 压缩

**方案 1：量化 KV Cache**

```cpp
// KV Cache INT8 量化
struct QuantizedKVCache {
    std::vector<int8_t> k_cache_q;
    std::vector<int8_t> v_cache_q;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
    
    void quantize(const float* k, const float* v, int size) {
        // 动态计算 scale
        float k_max = max_abs(k, size);
        float k_scale = k_max / 127.0f;
        
        for (int i = 0; i < size; ++i) {
            k_cache_q[i] = round(k[i] / k_scale);
        }
        k_scales.push_back(k_scale);
    }
};
```

**方案 2：滑动窗口 Attention**

```cpp
// 只保留最近 N 个 token 的 KV
constexpr int SLIDING_WINDOW = 2048;

void sliding_window_kv_cache(float* k_cache, float* v_cache, 
                              int seq_len, int head_dim) {
    if (seq_len > SLIDING_WINDOW) {
        // 丢弃最早的 token
        memmove(k_cache, k_cache + head_dim, 
                (SLIDING_WINDOW - 1) * head_dim * sizeof(float));
    }
}
```

---

## 3. 实施路线图

### Phase 1: CPU 核心优化（2-3 周）

| 任务 | 优先级 | 预期收益 | 工作量 |
|------|--------|----------|--------|
| 矩阵乘法 SIMD 优化 | 🔴 高 | 3-5x 加速 | 1 周 |
| BF16 SIMD 转换 | 🔴 高 | 5-10x 加速 | 3 天 |
| OpenMP 并行化 | 🔴 高 | 线性扩展 | 3 天 |
| 内存对齐优化 | 🟡 中 | 10-20% 提升 | 2 天 |

### Phase 2: GPU 核心优化（2-3 周）

| 任务 | 优先级 | 预期收益 | 工作量 |
|------|--------|----------|--------|
| 计算图缓存 | 🔴 高 | 30-50% 提升 | 1 周 |
| 异步数据传输 | 🔴 高 | 减少延迟 | 3 天 |
| 混合精度推理 | 🟡 中 | 2x 吞吐 | 3 天 |
| 批处理优化 | 🟡 中 | 提升利用率 | 4 天 |

### Phase 3: 内存优化（1-2 周）

| 任务 | 优先级 | 预期收益 | 工作量 |
|------|--------|----------|--------|
| 权重原地转换 | 🟡 中 | 减少 50% 内存 | 3 天 |
| KV Cache 量化 | 🟢 低 | 减少 75% 内存 | 4 天 |
| 内存映射加载 | 🟢 低 | 快速启动 | 3 天 |

### Phase 4: 高级优化（2-3 周）

| 任务 | 优先级 | 预期收益 | 工作量 |
|------|--------|----------|--------|
| Flash Attention | 🔴 高 | 长序列 2-4x | 2 周 |
| 连续批处理 | 🟡 中 | 吞吐提升 | 1 周 |
| NUMA 优化 | 🟢 低 | 多路 CPU 优化 | 3 天 |

---

## 4. 性能基准

### 4.1 目标性能指标

#### CPU 目标（Apple M3 Pro）

| 模型 | 当前 | 目标 | 优化后 |
|------|------|------|--------|
| Qwen3-0.6B | 20 t/s | 60 t/s | 3x |
| Qwen3-1.7B | 10 t/s | 30 t/s | 3x |
| Qwen3-7B | 3 t/s | 10 t/s | 3x |

#### GPU 目标（Metal）

| 模型 | 当前 | 目标 | 优化后 |
|------|------|------|--------|
| Qwen3-0.6B | 40 t/s | 100 t/s | 2.5x |
| Qwen3-1.7B | 20 t/s | 60 t/s | 3x |
| Qwen3-7B | 8 t/s | 25 t/s | 3x |

### 4.2 内存目标

| 指标 | 当前 | 目标 |
|------|------|------|
| 权重内存 | 2x (BF16+FP32) | 1x (仅 FP16) |
| KV Cache | FP32 | INT8 (75% 减少) |
| 工作缓冲 | 每请求独立 | 内存池复用 |

---

## 5. 代码实现建议

### 5.1 矩阵乘法优化示例

```cpp
// include/cllm/kylin/core/optimized_matmul.h
#pragma once

#include <cstddef>

namespace cllm {
namespace kylin {

// 平台检测
#if defined(__AVX512F__)
    #define KYLIN_USE_AVX512
#elif defined(__AVX2__)
    #define KYLIN_USE_AVX2
#elif defined(__ARM_NEON)
    #define KYLIN_USE_NEON
#endif

// 优化的矩阵乘法接口
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, bool transB = false);

// BF16 快速转换
void convert_bf16_to_f32_fast(const uint16_t* src, float* dst, size_t count);
void convert_f32_to_bf16_fast(const float* src, uint16_t* dst, size_t count);

} // namespace kylin
} // namespace cllm
```

### 5.2 GPU 计算图缓存示例

```cpp
// src/kylin/hf/ggml_backend.cpp

class GraphCache {
public:
    struct GraphKey {
        int seq_len;
        int batch_size;
        
        bool operator==(const GraphKey& other) const {
            return seq_len == other.seq_len && batch_size == other.batch_size;
        }
    };
    
    struct GraphKeyHash {
        size_t operator()(const GraphKey& k) const {
            return std::hash<int>()(k.seq_len) ^ 
                   (std::hash<int>()(k.batch_size) << 1);
        }
    };
    
    ggml_cgraph* get_or_create(const GraphKey& key, 
                                std::function<ggml_cgraph*()> creator);
    void clear();
    
private:
    std::unordered_map<GraphKey, std::unique_ptr<ggml_cgraph>, GraphKeyHash> cache_;
};
```

---

## 6. 测试与验证

### 6.1 性能测试

```bash
# 基准测试
./bin/cllm_benchmark --backend kylin --model Qwen3-1.7B --device cpu
./bin/cllm_benchmark --backend kylin --model Qwen3-1.7B --device gpu

# 对比测试
./bin/cllm_benchmark --backend llama_cpp --model Qwen3-1.7B --device cpu
```

### 6.2 正确性验证

```cpp
// tests/test_kylin_optimized.cpp
TEST(KylinOptimized, MatmulCorrectness) {
    // 对比优化前后的结果
    std::vector<float> A(1024*1024), B(1024*1024), C_ref(1024*1024), C_opt(1024*1024);
    
    // 填充随机数据
    fill_random(A.data(), A.size());
    fill_random(B.data(), B.size());
    
    // 参考实现
    matmul_reference(A.data(), B.data(), C_ref.data(), 1024, 1024, 1024);
    
    // 优化实现
    matmul_optimized(A.data(), B.data(), C_opt.data(), 1024, 1024, 1024);
    
    // 验证误差
    EXPECT_LT(max_relative_error(C_ref, C_opt), 1e-5);
}
```

---

## 7. 总结

### 关键优化点

1. **CPU 核心**：矩阵乘法 SIMD + OpenMP 并行
2. **GPU 核心**：计算图缓存 + 异步传输
3. **内存优化**：原地转换 + KV Cache 量化
4. **算法优化**：Flash Attention + 连续批处理

### 预期成果

- **性能提升**：CPU 3x，GPU 2.5-3x
- **内存优化**：权重 50%，KV Cache 75%
- **延迟降低**：首 token 时间减少 30-50%

### 下一步行动

1. 立即开始 Phase 1（CPU 矩阵乘法优化）
2. 并行准备 Phase 2（GPU 计算图缓存）
3. 建立性能基准测试框架
4. 每周进行性能回归测试

---

**文档版本**: 1.0  
**作者**: cLLM Team  
**日期**: 2026-02-05
