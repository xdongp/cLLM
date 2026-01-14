# GGUF Q4_K_M 量化格式推理实现深度分析

> **更新日期**: 2025-01-XX  
> **分析范围**: llama.cpp中Q4_K_M格式的完整实现  
> **重点关注**: 量化格式、反量化算法、推理使用方式

## 目录

1. [Q4_K_M格式概述](#1-q4_k_m格式概述)
2. [数据结构详解](#2-数据结构详解)
3. [量化算法分析](#3-量化算法分析)
4. [反量化算法分析](#4-反量化算法分析)
5. [推理时的使用方式](#5-推理时的使用方式)
6. [性能优化技术](#6-性能优化技术)
7. [与cLLM实现的对比](#7-与cllm实现的对比)
8. [实现建议](#8-实现建议)

---

## 1. Q4_K_M格式概述

### 1.1 基本特性

Q4_K_M (Q4_K Medium) 是llama.cpp中一种高效的4位量化格式，属于K-quant系列：

- **位宽**: 平均约4.5 bits/weight
- **块大小**: 256个元素/块 (QK_K = 256)
- **子块结构**: 8个子块，每个32元素
- **压缩比**: 约5.3:1 (相比FP32)
- **质量**: 在质量和压缩比之间取得良好平衡

### 1.2 格式特点

1. **分层量化**: 
   - 超级块级别：全局scale (d) 和 min scale (dmin)
   - 子块级别：每个32元素子块有独立的scale和min

2. **自适应量化**:
   - 使用重要性权重(quant_weights)优化量化
   - 支持激活感知量化(AWQ)

3. **高效存储**:
   - scales使用6位量化
   - 4位值打包存储（2个值/字节）

### 1.3 与其他格式对比

| 格式 | 位宽 | 块大小 | 压缩比 | 质量 |
|------|------|--------|--------|------|
| Q4_0 | 4.0 | 32 | 8:1 | 基础 |
| Q4_1 | 4.0 | 32 | 8:1 | 基础+偏移 |
| **Q4_K_M** | **4.5** | **256** | **5.3:1** | **高** |
| Q4_K_S | 4.5 | 256 | 5.3:1 | 中等 |
| Q5_K_M | 5.5 | 256 | 4.4:1 | 更高 |
| Q8_0 | 8.0 | 32 | 4:1 | 最高 |

---

## 2. 数据结构详解

### 2.1 block_q4_K结构

```c
// ggml/src/ggml-common.h
#define QK_K 256          // 每个块256个元素
#define K_SCALE_SIZE 12   // scales数组大小

typedef struct {
    // 超级块级别的缩放因子
    union {
        struct {
            ggml_half d;    // 超级块scale（用于量化后的scales）
            ggml_half dmin; // 超级块scale（用于量化后的mins）
        };
        ggml_half2 dm;      // 打包存储
    };
    
    // 子块级别的scales和mins（6位量化）
    // 存储8个子块的scale和min，每个6位
    uint8_t scales[K_SCALE_SIZE];  // 12字节 = 96位，足够存储8*2*6=96位
    
    // 4位量化值（打包存储）
    // 256个元素，每个4位，共128字节
    // 每字节存储2个4位值：低4位和高4位
    uint8_t qs[QK_K/2];  // 128字节
} block_q4_K;

// 总大小验证
static_assert(sizeof(block_q4_K) == 
    2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, 
    "wrong q4_K block size/padding");
// = 4 + 12 + 128 = 144字节/块
```

### 2.2 内存布局

```
block_q4_K (144字节):
├── d (2字节): FP16格式的超级块scale
├── dmin (2字节): FP16格式的超级块min scale
├── scales[12] (12字节): 子块scales和mins的量化值
│   └── 编码方式：
│       - j < 4: scales[j] = ls (低6位), scales[j+4] = lm (低6位)
│       - j >= 4: 复杂打包，使用高2位
└── qs[128] (128字节): 256个4位量化值
    └── 打包方式：
        - qs[i]的低4位 = 元素[2*i]的量化值
        - qs[i]的高4位 = 元素[2*i+1]的量化值
```

### 2.3 Scale编码详解

```c
// ggml/src/ggml-quants.c
static inline void get_scale_min_k4(int j, const uint8_t * q, 
                                     uint8_t * d, uint8_t * m) {
    if (j < 4) {
        // 前4个子块：直接存储
        *d = q[j] & 63;        // scale，低6位
        *m = q[j + 4] & 63;    // min，低6位
    } else {
        // 后4个子块：打包存储
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);  // 组合低4位和高2位
        *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4);   // 组合低4位和高2位
    }
}
```

**编码示例**:
```
子块0-3:
  scales[0-3]: scale值 (6位)
  scales[4-7]: min值 (6位)

子块4-7:
  scales[8-11]: scale和min的低4位
  scales[0-3]的高2位: scale的高2位
  scales[4-7]的高2位: min的高2位
```

---

## 3. 量化算法分析

### 3.1 量化流程

```c
// ggml/src/ggml-quants.c
static void quantize_row_q4_K_impl(
    const float * x, 
    block_q4_K * y, 
    int64_t n_per_row, 
    const float * quant_weights) {
    
    const int64_t nb = n_per_row / QK_K;  // 块数
    
    for (int i = 0; i < nb; i++) {
        // 步骤1: 计算统计信息
        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) {
            sum_x2 += x[l] * x[l];
        }
        float sigma2 = 2*sum_x2/QK_K;
        float av_x = sqrtf(sigma2);
        
        // 步骤2: 对每个32元素子块进行量化
        for (int j = 0; j < QK_K/32; ++j) {
            // 计算重要性权重
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) {
                    weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
                }
            } else {
                for (int l = 0; l < 32; ++l) {
                    weights[l] = av_x + fabsf(x[32*j + l]);
                }
            }
            
            // 量化子块（使用make_qkx3_quants）
            scales[j] = make_qkx3_quants(
                32, 15,                    // 32元素，15级量化
                x + 32*j, weights, 
                L + 32*j, &mins[j], Laux,
                -0.9f, 0.05f, 36, false
            );
        }
        
        // 步骤3: 量化超级块的scales和mins
        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K/32, 63, mins, Lm, sw);
        
        // 步骤4: 编码scales
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];  // 量化后的scale
            uint8_t lm = Lm[j];  // 量化后的min
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                // 打包编码
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        
        // 步骤5: 存储超级块scale
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);
        
        // 步骤6: 重新量化并打包4位值
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            
            // 量化到0-15
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        
        // 步骤7: 打包4位值到字节
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }
        
        x += QK_K;
    }
}
```

### 3.2 关键量化函数

#### 3.2.1 make_qkx3_quants

```c
// 对32个元素进行3级量化（实际是4位，0-15）
static float make_qkx3_quants(
    int n,                    // 元素数 (32)
    int nmax,                 // 最大量化值 (15)
    const float * x,           // 输入数据
    const float * weights,    // 重要性权重
    uint8_t * L,              // 输出量化值
    float * the_min,          // 输出min值
    uint8_t * Laux,           // 辅助数组
    float rmin,               // -0.9f
    float rdelta,             // 0.05f
    int nstep,                // 36
    bool use_mad) {           // false
    
    // 1. 计算min和max
    float min = x[0], max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    
    if (min > 0) min = 0;  // 确保min <= 0
    
    // 2. 迭代优化量化参数
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = ...;
    
    // 3. 尝试不同的scale范围
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        // ... 计算误差并选择最佳scale
    }
    
    return scale;
}
```

#### 3.2.2 make_qp_quants

```c
// 量化scales和mins本身（使用6位，0-63）
static float make_qp_quants(
    int n,                    // 子块数 (8)
    int nmax,                 // 最大量化值 (63)
    const float * x,          // scales或mins数组
    uint8_t * L,             // 输出量化值
    const float * quant_weights) {
    
    // 找到最大值
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    
    // 量化
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        L[i] = MAX(0, MIN(nmax, l));
    }
    
    return max / nmax;  // 返回反量化scale
}
```

### 3.3 量化参数说明

- **nmax = 15**: 4位量化，值范围0-15
- **nmax = 63**: 6位量化scales，值范围0-63
- **rmin = -0.9f, rdelta = 0.05f, nstep = 36**: 优化scale的搜索范围
- **重要性权重**: 支持激活感知量化，提高重要权重的精度

---

## 4. 反量化算法分析

### 4.1 反量化流程

```c
// ggml/src/ggml-quants.c
void dequantize_row_q4_K(
    const block_q4_K * x, 
    float * y, 
    int64_t k) {
    
    assert(k % QK_K == 0);
    const int nb = k / QK_K;  // 块数

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;  // 量化值数组
        
        // 获取超级块级别的scale
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;  // scale索引
        uint8_t sc, m;
        
        // 处理每个64元素组（2个子块）
        for (int j = 0; j < QK_K; j += 64) {
            // 获取第一个子块的scale和min
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;   // 实际scale
            const float m1 = min * m;  // 实际min
            
            // 获取第二个子块的scale和min
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; 
            const float m2 = min * m;
            
            // 反量化前32个元素（使用d1和m1）
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;  // 低4位
            }
            
            // 反量化后32个元素（使用d2和m2）
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;   // 高4位
            }
            
            q += 32;  // 移动到下一组
            is += 2;  // 更新scale索引
        }
    }
}
```

### 4.2 反量化公式

对于每个元素：

```
value = d * sc * quantized_value - min * m

其中:
- d: 超级块scale (FP16)
- sc: 子块scale (6位量化，0-63)
- quantized_value: 4位量化值 (0-15)
- min: 超级块min scale (FP16)
- m: 子块min (6位量化，0-63)
```

### 4.3 详细步骤

1. **读取超级块参数**:
   ```c
   d = FP16_to_FP32(block.d)
   min = FP16_to_FP32(block.dmin)
   ```

2. **处理每个64元素组**:
   - 提取2个子块的scale和min
   - 计算实际scale: `d1 = d * sc1`, `d2 = d * sc2`
   - 计算实际min: `m1 = min * m1`, `m2 = min * m2`

3. **反量化32个元素**:
   ```c
   // 前32个元素（低4位）
   for (l = 0; l < 32; l++) {
       value = d1 * (q[l] & 0xF) - m1;
   }
   
   // 后32个元素（高4位）
   for (l = 0; l < 32; l++) {
       value = d2 * (q[l] >> 4) - m2;
   }
   ```

### 4.4 内存访问模式

```
块布局 (144字节):
[0-1]    d (FP16)
[2-3]    dmin (FP16)
[4-15]   scales[12] (编码的scales和mins)
[16-143] qs[128] (打包的4位值)

访问模式:
- 顺序读取块
- 顺序读取每个64元素组
- 顺序读取每个32元素子块
- 缓存友好
```

---

## 5. 推理时的使用方式

### 5.1 在计算图中的使用

llama.cpp在推理时**不进行完整反量化**，而是直接在量化数据上计算：

```c
// ggml/src/ggml.c
// Q4_K类型注册
[GGML_TYPE_Q4_K] = {
    .type_name      = "q4_K",
    .blck_size      = QK_K,           // 256
    .type_size      = sizeof(block_q4_K),  // 144
    .is_quantized   = true,
    .to_float       = dequantize_row_q4_K,  // 仅用于调试/导出
    .from_float_ref = quantize_row_q4_K_ref,
},
```

### 5.2 矩阵乘法优化

llama.cpp实现了专门的量化矩阵乘法内核：

#### 5.2.1 CPU后端

```c
// ggml/src/ggml-cpu/ggml-cpu-impl.h
// 专门的Q4_K矩阵乘法实现
// 在计算时动态反量化，避免完整反量化

// 伪代码示例
void mul_mat_q4_K_f32(
    const block_q4_K * A,    // Q4_K格式的权重
    const float * B,         // FP32输入
    float * C,               // FP32输出
    int64_t n,               // 行数
    int64_t k) {             // 列数
    
    for (int i = 0; i < n; i++) {
        float sum = 0;
        
        // 按块处理
        for (int j = 0; j < k; j += QK_K) {
            const block_q4_K * block = &A[i * (k/QK_K) + j/QK_K];
        
            // 动态反量化并计算点积
            for (int l = 0; l < QK_K; l++) {
                // 获取scale和min
                uint8_t sc, m;
                get_scale_min_k4(l/32, block->scales, &sc, &m);
                
                float d = GGML_FP16_TO_FP32(block->d) * sc;
                float min_val = GGML_FP16_TO_FP32(block->dmin) * m;
                
                // 提取量化值
                int q_idx = l / 2;
                int q_val = (l % 2 == 0) ? 
                    (block->qs[q_idx] & 0xF) : 
                    (block->qs[q_idx] >> 4);
                
                // 反量化并累加
                float val = d * q_val - min_val;
                sum += val * B[j + l];
            }
        }
        
        C[i] = sum;
    }
}
```

#### 5.2.2 GPU后端优化

llama.cpp为不同GPU后端实现了优化的内核：

**CUDA实现**:
```cuda
// ggml/src/ggml-cuda/convert.cu
__global__ void dequantize_block_q4_K(
    const void * vx, 
    dst_t * yy) {
    // GPU上的块级反量化
    // 利用共享内存和warp级操作
}
```

**Metal实现**:
```metal
// ggml/src/ggml-metal/ggml-metal.metal
kernel void dequantize_q4_K(
    device const block_q4_K * xb,
    short il,
    thread type4x4 & reg) {
    // Apple Silicon优化实现
    // 使用SIMD组操作
}
```

### 5.3 推理流程

```
输入 (FP32 tokens)
    │
    ▼
┌─────────────────┐
│ Embedding层     │  ← 通常不量化
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer层   │
│                 │
│ ┌─────────────┐ │
│ │ Attention   │ │  ← 权重Q4_K_M
│ │ - wq (Q4_K) │ │    直接使用量化数据
│ │ - wk (Q4_K) │ │    不完整反量化
│ │ - wv (Q4_K) │ │
│ │ - wo (Q4_K) │ │
│ └─────────────┘ │
│                 │
│ ┌─────────────┐ │
│ │ FFN         │ │  ← 权重Q4_K_M
│ │ - gate(Q4_K)│ │
│ │ - up (Q4_K) │ │
│ │ - down(Q4_K)│ │
│ └─────────────┘ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output层        │  ← 通常不量化
└─────────────────┘
```

### 5.4 关键优化点

1. **延迟反量化**: 只在需要时反量化，不预先反量化整个张量
2. **融合操作**: 反量化与矩阵乘法融合，减少内存访问
3. **SIMD优化**: 使用AVX/AVX2/NEON等SIMD指令加速
4. **缓存优化**: 块级处理，提高缓存命中率

---

## 6. 性能优化技术

### 6.1 SIMD优化

#### 6.1.1 AVX2优化示例

```c
#ifdef __AVX2__
void dequantize_row_q4_K_avx2(
    const block_q4_K * x,
    float * y,
    int64_t k) {
    
    const int nb = k / QK_K;
    
    for (int i = 0; i < nb; i++) {
        // 加载超级块scale
        __m256 d_vec = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));
        __m256 min_vec = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].dmin));
        
        // 处理32个元素批次
        for (int j = 0; j < QK_K; j += 32) {
            // 获取子块scale和min
            uint8_t sc, m;
            get_scale_min_k4(j/32, x[i].scales, &sc, &m);
            
            __m256 scale = _mm256_mul_ps(d_vec, _mm256_set1_ps(sc));
            __m256 min_val = _mm256_mul_ps(min_vec, _mm256_set1_ps(m));
            
            // 加载16个字节（32个4位值）
            __m128i q_bytes = _mm_loadu_si128((__m128i*)(x[i].qs + j/2));
            
            // 提取低4位和高4位
            __m128i q_low = _mm_and_si128(q_bytes, _mm_set1_epi8(0x0F));
            __m128i q_high = _mm_srli_epi16(q_bytes, 4);
            q_high = _mm_and_si128(q_high, _mm_set1_epi8(0x0F));
            
            // 转换为float并应用scale
            // ... SIMD操作 ...
        }
    }
}
#endif
```

### 6.2 内存访问优化

1. **顺序访问**: 块和子块按顺序处理
2. **缓存对齐**: 块结构对齐到缓存行
3. **预取**: 在处理当前块时预取下一个块

### 6.3 计算优化

1. **融合操作**: 反量化+矩阵乘法融合
2. **批量处理**: 一次处理多个块
3. **并行化**: 多线程处理不同行

---

## 7. 与cLLM实现的对比

### 7.1 当前cLLM实现

```cpp
// src/model/gguf_dequantization.cpp
void dequantizeQ4KMF32(const void* input, float* output, 
                       const std::vector<size_t>& shape) {
    // 当前实现：完整反量化到FP32
    // 问题：格式理解不准确
}
```

**问题**:
1. ❌ 块结构理解错误（当前假设6字节/块，实际144字节/块）
2. ❌ 缺少子块scale/min处理
3. ❌ 缺少scale编码/解码逻辑
4. ❌ 没有实现推理时的融合计算

### 7.2 正确实现要点

#### 7.2.1 块大小计算

```cpp
// 正确计算
size_t getTensorByteSize(const GGULTensorInfo& tensorInfo) {
    uint64_t elementCount = 1;
    for (uint64_t dim : tensorInfo.shape) {
        elementCount *= dim;
    }
    
    if (tensorInfo.type == GGMLType::Q4_K) {
        // Q4_K: 256元素/块，144字节/块
        uint64_t blockCount = (elementCount + QK_K - 1) / QK_K;
        return blockCount * sizeof(block_q4_K);
    }
}
```

#### 7.2.2 正确的反量化实现

```cpp
void dequantize_row_q4_K(
    const block_q4_K * x,
    float * y,
    int64_t k) {
    
    const int nb = k / QK_K;
    
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d = FP16_to_FP32(x[i].d);
        const float min = FP16_to_FP32(x[i].dmin);
        
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            // 获取两个子块的scale和min
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, x[i].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, x[i].scales, &sc2, &m2);
            
            const float d1 = d * sc1;
            const float m1_val = min * m1;
            const float d2 = d * sc2;
            const float m2_val = min * m2;
            
            // 反量化前32个元素
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1_val;
            }
            
            // 反量化后32个元素
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2_val;
            }
            
            q += 32;
            is += 2;
        }
    }
}
```

### 7.3 实现差距

| 功能 | llama.cpp | cLLM当前 | 需要改进 |
|------|-----------|----------|----------|
| **块结构理解** | ✅ 正确 | ❌ 错误 | 需要修正 |
| **Scale编码** | ✅ 完整 | ❌ 缺失 | 需要实现 |
| **反量化算法** | ✅ 完整 | ⚠️ 部分 | 需要完善 |
| **推理融合** | ✅ 支持 | ❌ 不支持 | 需要实现 |
| **SIMD优化** | ✅ 多平台 | ⚠️ 部分 | 需要扩展 |

---

## 8. 实现建议

### 8.1 短期改进（必须）

1. **修正块结构定义**
   ```cpp
   // include/cllm/model/gguf_dequantization.h
   struct block_q4_K {
       ggml_half d;           // 2字节
       ggml_half dmin;        // 2字节
       uint8_t scales[12];    // 12字节
       uint8_t qs[128];       // 128字节
   };
   static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");
   ```

2. **实现scale解码函数**
   ```cpp
   static inline void get_scale_min_k4(
       int j, 
       const uint8_t * scales, 
       uint8_t * d, 
       uint8_t * m) {
       if (j < 4) {
           *d = scales[j] & 63;
           *m = scales[j + 4] & 63;
       } else {
           *d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
           *m = (scales[j+4] >> 4) | ((scales[j-0] >> 6) << 4);
       }
   }
   ```

3. **重写反量化函数**
   - 参考llama.cpp的`dequantize_row_q4_K`
   - 正确处理256元素块
   - 正确处理8个子块
   - 正确处理scale编码

### 8.2 中期改进（重要）

1. **实现推理时的融合计算**
   - 不完整反量化
   - 直接在量化数据上计算
   - 实现专门的矩阵乘法内核

2. **SIMD优化**
   - AVX2优化
   - NEON优化（ARM）
   - 批量处理优化

3. **内存优化**
   - 缓存对齐
   - 预取优化
   - 减少内存拷贝

### 8.3 长期改进（可选）

1. **GPU支持**
   - CUDA内核
   - Metal内核（macOS）
   - OpenCL支持

2. **更多量化格式**
   - Q5_K_M
   - Q6_K
   - IQ系列

3. **量化感知训练支持**
   - 支持重要性矩阵
   - 激活感知量化

---

## 附录

### A. 关键常量

```c
#define QK_K 256              // 每个块256个元素
#define K_SCALE_SIZE 12       // scales数组大小
#define sizeof(block_q4_K) 144 // 块大小（字节）
```

### B. 计算公式

**反量化公式**:
```
value = d * sc * q - min * m

其中:
- d: FP16超级块scale
- sc: 6位子块scale (0-63)
- q: 4位量化值 (0-15)
- min: FP16超级块min scale
- m: 6位子块min (0-63)
```

**块大小计算**:
```
block_count = ceil(element_count / 256)
byte_size = block_count * 144
```

### C. 参考实现

- **量化**: `ggml/src/ggml-quants.c:quantize_row_q4_K_impl`
- **反量化**: `ggml/src/ggml-quants.c:dequantize_row_q4_K`
- **结构定义**: `ggml/src/ggml-common.h:block_q4_K`
- **类型注册**: `ggml/src/ggml.c:type_traits[GGML_TYPE_Q4_K]`

### D. 测试建议

1. **单元测试**: 测试反量化正确性
2. **集成测试**: 测试完整推理流程
3. **性能测试**: 对比llama.cpp性能
4. **精度测试**: 验证量化误差

---

**报告结束**
