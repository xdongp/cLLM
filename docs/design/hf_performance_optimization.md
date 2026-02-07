# HF 后端性能优化方案

> **状态**: ✅ Phase 1-3 已完成 (2026-01-24)

## 0. 优化结果总结

| 阶段 | 优化内容 | tokens/s | 提升 |
|------|----------|----------|------|
| 原始 | 朴素实现 | ~0.5 | - |
| Phase 1 | SIMD (NEON) | ~0.5 | +0% |
| Phase 2 | 预转换权重 | ~1.0 | **+100%** |
| **Phase 3** | **OpenMP 8线程** | **~4.0** | **+700%** |

### 配置日志
```
[ggml_kernels] Using ARM NEON optimizations
[ggml_kernels] OpenMP enabled: 8 threads available
[HFTransformer] Pre-conversion complete: 2161.72 MB F32 weights
```

## 1. 当前性能瓶颈分析

### 1.1 主要瓶颈

| 瓶颈 | 位置 | 影响 | 占比 |
|------|------|------|------|
| **BF16 矩阵乘法** | `matmulBF16()` | 每层 6 次调用，逐元素转换 | ~70% |
| **单线程执行** | 全局 | CPU 利用率低 | ~20% |
| **LM Head** | `lmHead()` | 151936 × 1024 = 1.56 亿次浮点运算 | ~10% |

### 1.2 `matmulBF16` 分析

当前实现（朴素双重循环）：
```cpp
for (int i = 0; i < outFeatures; ++i) {
    float sum = 0.0f;
    const uint16_t* row = weight + i * inFeatures;
    for (int j = 0; j < inFeatures; ++j) {
        float w = bf16ToF32(row[j]);  // 逐元素 BF16→F32
        sum += w * input[j];
    }
    output[i] = sum;
}
```

**问题**：
1. 每次循环都调用 `bf16ToF32()` - 无法 SIMD 向量化
2. 无缓存优化 - 内存访问不友好
3. 单线程 - 无法利用多核

## 2. 优化方案

### 方案 A: 使用 GGML 内核（推荐）

GGML 已经有高度优化的 BF16 计算：
- 支持 AVX/AVX2/AVX512/NEON
- 内置 BF16→F32 SIMD 转换
- 多线程支持

```cpp
// 使用 GGML 的 ggml_mul_mat
#include <ggml.h>

void matmulBF16_ggml(const uint16_t* weight, const float* input, 
                     float* output, int outFeatures, int inFeatures) {
    // 创建 ggml_tensor 并调用 ggml_mul_mat
    // GGML 会自动选择最优实现
}
```

**优势**：
- 项目已集成 GGML（llama.cpp 依赖）
- 成熟稳定，广泛验证
- 支持多种 SIMD 指令集

### 方案 B: 使用平台原生 BLAS

**macOS (Accelerate)**:
```cpp
#include <Accelerate/Accelerate.h>

// 先预转换 BF16→F32，然后调用 cblas_sgemv
cblas_sgemv(CblasRowMajor, CblasNoTrans, 
            outFeatures, inFeatures, 
            1.0f, weightF32, inFeatures, 
            input, 1, 
            0.0f, output, 1);
```

**Linux (OpenBLAS)**:
```cpp
#include <cblas.h>
// 同上
```

**优势**：
- 平台深度优化
- 矩阵乘法极快

**劣势**：
- 需要预转换权重到 F32（增加内存 2x）

### 方案 C: 手写 SIMD 优化

```cpp
#include <immintrin.h>  // AVX2

void matmulBF16_avx2(const uint16_t* weight, const float* input, 
                     float* output, int outFeatures, int inFeatures) {
    for (int i = 0; i < outFeatures; ++i) {
        __m256 sum = _mm256_setzero_ps();
        
        for (int j = 0; j < inFeatures; j += 8) {
            // 加载 8 个 BF16 并转换为 F32
            __m128i bf16 = _mm_loadu_si128((const __m128i*)(weight + i * inFeatures + j));
            __m256i bf16_32 = _mm256_cvtepu16_epi32(bf16);
            __m256 w = _mm256_castsi256_ps(_mm256_slli_epi32(bf16_32, 16));
            
            // 加载输入
            __m256 inp = _mm256_loadu_ps(input + j);
            
            // FMA
            sum = _mm256_fmadd_ps(w, inp, sum);
        }
        
        // 水平求和
        output[i] = hsum256_ps(sum);
    }
}
```

### 方案 D: 预转换权重 + OpenMP 多线程

```cpp
// 加载时预转换
std::vector<float> weightF32(outFeatures * inFeatures);
bf16ToF32Array(weightBF16, weightF32.data(), outFeatures * inFeatures);

// 运行时使用多线程
#pragma omp parallel for
for (int i = 0; i < outFeatures; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < inFeatures; ++j) {
        sum += weightF32[i * inFeatures + j] * input[j];
    }
    output[i] = sum;
}
```

## 3. 推荐实施计划

### Phase 1: GGML 集成（快速收益）

1. 创建 `core/ggml_kernels.cpp`
2. 封装 GGML 矩阵乘法
3. 替换 `matmulBF16()` 调用

预期提升：**5-10x**

### Phase 2: 预转换权重（减少运行时开销）

1. 在 `loadWeights()` 时将 BF16 转为 F32
2. 存储 F32 权重指针
3. 内存增加约 1GB（对于 0.6B 模型）

预期提升：**额外 1.5-2x**

### Phase 3: 多线程（充分利用 CPU）

1. 添加 OpenMP 支持
2. 层间和算子间并行

预期提升：**额外 2-4x**（取决于核心数）

## 4. 预期性能对比

| 配置 | 单 Token 推理 | 吞吐量 |
|------|--------------|--------|
| 当前（朴素） | ~1.5s | 0.6 tok/s |
| Phase 1 (GGML) | ~150ms | 6 tok/s |
| Phase 2 (+预转换) | ~100ms | 10 tok/s |
| Phase 3 (+多线程) | ~30ms | 30+ tok/s |

## 5. 代码修改清单

### 5.1 新增文件

```
src/kylin/core/
├── ggml_kernels.cpp    # GGML 封装
└── ggml_kernels.h

include/cllm/kylin/core/
└── ggml_kernels.h
```

### 5.2 修改文件

- `src/kylin/hf/transformer.cpp` - 替换 matmulBF16
- `include/cllm/kylin/hf/hf_transformer_model.h` - 添加 F32 权重成员
- `CMakeLists.txt` - 添加 OpenMP 支持

## 6. 实现示例

### 6.1 GGML 内核封装

```cpp
// core/ggml_kernels.h
#pragma once

namespace cllm {
namespace kylin {
namespace ggml_kernels {

/**
 * @brief BF16 矩阵向量乘法（使用 GGML 优化）
 */
void matmul_bf16_f32(
    const uint16_t* weight,  // [M, K] BF16
    const float* input,      // [K] F32
    float* output,           // [M] F32
    int M, int K
);

/**
 * @brief 批量 BF16→F32 转换（SIMD 优化）
 */
void convert_bf16_to_f32(
    const uint16_t* src,
    float* dst,
    size_t count
);

} // namespace ggml_kernels
} // namespace kylin
} // namespace cllm
```
