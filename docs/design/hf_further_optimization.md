# HF 后端进一步优化方案

## 1. Apple Accelerate BLAS（推荐，Phase 4）

### 原理
Apple Accelerate 框架包含高度优化的 BLAS 实现，专为 Apple Silicon 优化。

### 实现
```cpp
#include <Accelerate/Accelerate.h>

void matmul_accelerate(const float* A, const float* B, float* C, 
                       int M, int N, int K) {
    // C = A * B
    // A: [M, K], B: [K, N], C: [M, N]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,           // alpha
                A, K,           // A, lda
                B, N,           // B, ldb
                0.0f,           // beta
                C, N);          // C, ldc
}

// 对于向量情况 (N=1)
void matvec_accelerate(const float* A, const float* x, float* y,
                       int M, int K) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                M, K,
                1.0f,           // alpha
                A, K,           // A, lda
                x, 1,           // x, incx
                0.0f,           // beta
                y, 1);          // y, incy
}
```

### CMakeLists.txt 修改
```cmake
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(cllm_core PRIVATE ${ACCELERATE_FRAMEWORK})
        add_compile_definitions(USE_ACCELERATE)
        message(STATUS "✅ Using Apple Accelerate framework")
    endif()
endif()
```

### 预期收益
- **2-3x 性能提升**（8-12 tok/s）
- 零额外依赖（macOS 内置）

---

## 2. LM Head Top-K 优化（Phase 5）

### 问题
LM Head 是最大的瓶颈：
- 输出: 151936 × 1024 = 1.56 亿次 FMA
- 但采样只需要 Top-K（通常 K=50-100）

### 方案 A: 近似 Top-K（简单）
```cpp
void lmHead_topk(const float* input, float* output, int vocabSize, 
                 int hiddenSize, int topK) {
    // 1. 随机采样 10% 的词表作为候选
    std::vector<int> candidates = sampleCandidates(vocabSize, vocabSize / 10);
    
    // 2. 只计算候选的 logits
    for (int c : candidates) {
        output[c] = dot_product(lmHead[c], input, hiddenSize);
    }
    
    // 3. 找 Top-K 候选
    auto topK_ids = findTopK(output, candidates, topK * 2);
    
    // 4. 精确计算 Top-K 区域
    for (int id : topK_ids) {
        output[id] = dot_product(lmHead[id], input, hiddenSize);
    }
}
```

### 方案 B: 分层词表（复杂但精确）
- 将词表按使用频率分组
- 先计算高频组，再按需计算低频组

### 预期收益
- **1.5-2x**（如果采样 10% 候选）
- **5-10x**（如果使用分层词表）

---

## 3. 分块矩阵乘法（Phase 6）

### 问题
当前实现不考虑 CPU 缓存，大矩阵会产生大量 cache miss。

### 实现
```cpp
constexpr int BLOCK_SIZE = 64;  // 适应 L1 cache

void matmul_blocked(const float* A, const float* x, float* y,
                    int M, int K) {
    std::fill(y, y + M, 0.0f);
    
    for (int m0 = 0; m0 < M; m0 += BLOCK_SIZE) {
        int m1 = std::min(m0 + BLOCK_SIZE, M);
        
        for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            int k1 = std::min(k0 + BLOCK_SIZE, K);
            
            // 处理块 [m0:m1, k0:k1]
            for (int m = m0; m < m1; ++m) {
                float sum = 0.0f;
                for (int k = k0; k < k1; ++k) {
                    sum += A[m * K + k] * x[k];
                }
                y[m] += sum;
            }
        }
    }
}
```

### 预期收益
- **1.2-1.5x**（减少 cache miss）

---

## 4. Metal GPU 加速（Phase 7）

### 概述
使用 Apple Metal 将矩阵乘法卸载到 GPU。

### 架构
```
CPU                          GPU
 │                            │
 ├─ Embedding ────────────────┤
 ├─ RMSNorm ──────────────────┤
 │                            │
 │  ┌───────────────────────┐ │
 │  │ Q/K/V Projection      │◄┤ Metal shader
 │  │ Attention Scores      │◄┤
 │  │ Output Projection     │◄┤
 │  │ FFN (Gate, Up, Down)  │◄┤
 │  │ LM Head               │◄┤
 │  └───────────────────────┘ │
 │                            │
 └────── Sampling ────────────┘
```

### 实现复杂度
- 高：需要 Metal shader、内存管理、同步
- 建议使用 MPS (Metal Performance Shaders) 或 ggml-metal

### 预期收益
- **3-5x**（M1/M2 GPU 非常强）

---

## 5. INT8 动态量化（Phase 8）

### 概述
运行时将权重量化为 INT8，减少内存带宽。

### 实现
```cpp
struct QuantizedWeight {
    std::vector<int8_t> data;
    std::vector<float> scales;  // 每 128 元素一个 scale
    int rows, cols;
};

void quantize_f32_to_int8(const float* src, QuantizedWeight& dst,
                          int rows, int cols) {
    constexpr int GROUP_SIZE = 128;
    dst.data.resize(rows * cols);
    dst.scales.resize((rows * cols + GROUP_SIZE - 1) / GROUP_SIZE);
    
    for (int g = 0; g < dst.scales.size(); ++g) {
        int start = g * GROUP_SIZE;
        int end = std::min(start + GROUP_SIZE, rows * cols);
        
        // 找最大绝对值
        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = std::max(amax, std::abs(src[i]));
        }
        
        float scale = amax / 127.0f;
        dst.scales[g] = scale;
        
        // 量化
        for (int i = start; i < end; ++i) {
            dst.data[i] = static_cast<int8_t>(std::round(src[i] / scale));
        }
    }
}
```

### 预期收益
- **1.5-2x**（内存带宽减半）
- 精度损失 < 1%

---

## 6. 实施优先级

1. **Phase 4: Apple Accelerate** ⭐⭐⭐ (1-2小时)
   - 最简单，收益最高
   
2. **Phase 5: LM Head Top-K** ⭐⭐ (2-4小时)
   - 针对最大瓶颈
   
3. **Phase 6: 分块矩阵乘法** ⭐ (1-2小时)
   - 增量改进

4. **Phase 7: Metal GPU** ⭐⭐⭐ (1-2天)
   - 大幅提升但复杂

5. **Phase 8: INT8 量化** ⭐⭐ (半天)
   - 中等收益

---

## 7. 目标性能

| 阶段 | 预期 tok/s |
|------|-----------|
| 当前 (Phase 1-3) | ~4 |
| Phase 4 (Accelerate) | ~10 |
| Phase 5 (LM Head) | ~15 |
| Phase 7 (Metal) | ~30+ |

与 llama.cpp 对比目标：达到 llama.cpp 的 50-80% 性能。
