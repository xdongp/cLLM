# CPU Backend 内存分析报告

## 1. 内存使用现状

### 1.1 实测数据
```
最大驻留内存 (RSS): 8.49 GB
峰值内存占用: 11.6 GB
理论需求: 2.35 GB
开销倍数: 3.62x
```

### 1.2 内存分布分析

| 组件 | 理论内存 | 实际占用 | 倍数 |
|------|----------|----------|------|
| 权重 (F32) | 1.89 GB | ~3.8 GB | 2x |
| KV Cache | 0.44 GB | ~0.9 GB | 2x |
| 工作缓冲区 | 0.02 GB | ~0.1 GB | 5x |
| 其他开销 | - | ~3.7 GB | - |
| **总计** | **2.35 GB** | **8.49 GB** | **3.62x** |

## 2. 内存开销来源分析

### 2.1 权重存储 - 主要问题 ⚠️

**当前实现**:
```cpp
// cpu_backend.cpp
struct LayerWeights {
    std::vector<float> qProj;  // BF16 -> F32 转换后存储
    std::vector<float> kProj;
    // ... 所有权重都是F32
};
```

**问题**:
1. **双重存储**: SafetensorsLoader内存映射 (BF16) + CPUBackend (F32)
2. **内存映射开销**: mmap文件占用虚拟内存
3. **转换开销**: 加载时BF16→F32转换，额外分配临时内存

**内存占用**:
- 原始BF16权重: 0.95 GB (内存映射)
- 转换后F32权重: 1.89 GB (堆内存)
- **合计**: ~2.84 GB (理论1.89GB的1.5倍)

### 2.2 KV Cache - 次要问题

**当前实现**:
```cpp
struct KVCache {
    std::vector<float> kCache;  // 预分配max_seq_len
    std::vector<float> vCache;
};
// 按requestId存储在unordered_map中
```

**问题**:
1. **预分配过大**: 按max_seq_len=4096预分配，实际使用可能只有几百
2. **HashMap开销**: unordered_map每个entry有额外开销
3. **内存碎片**: 多次分配释放导致碎片

**内存占用**:
- 理论: 0.44 GB (max_seq=4096)
- 实际平均: 0.11 GB (avg_seq=1024)
- **浪费**: 75%的KV Cache内存未使用

### 2.3 工作缓冲区 - 轻微问题

**当前实现**:
```cpp
// 大量预分配缓冲区
std::vector<float> hiddenStates;    // 1024
std::vector<float> residual;        // 1024
std::vector<float> normOutput;      // 1024
std::vector<float> attnOutput;      // 1024
std::vector<float> ffnOutput;       // 1024
std::vector<float> bufferA;         // 1024 (双缓冲)
std::vector<float> bufferB;         // 1024
std::vector<float> qkvBuffer;       // 2048 + 1024 + 1024
std::vector<float> attnScores;      // 16 * 4096 = 65536
// ... 等等
```

**问题**:
1. **重复分配**: hiddenStates/bufferA/bufferB功能重复
2. **过度预分配**: attnScores按max_seq预分配
3. **未使用缓冲区**: gateUpBuffer等可能未使用

### 2.4 其他开销

1. **SafetensorsLoader内存映射**: ~1.2 GB
2. **Tokenizer数据**: ~200 MB
3. **RoPE预计算**: 2 × 4096 × 32 × 4 = 1 MB
4. **内存碎片和allocator开销**: ~500 MB
5. **系统库和运行时**: ~300 MB

## 3. 内存优化方案

### 3.1 方案A: 混合精度权重 (立即可实施) ⭐⭐⭐

**目标**: 减少50%权重内存

**实现**:
```cpp
struct LayerWeights {
    // 保持BF16存储，计算时即时转换
    std::vector<uint16_t> qProj_bf16;
    std::vector<uint16_t> kProj_bf16;
    // ...
    
    // 热点权重可缓存F32
    std::unique_ptr<std::vector<float>> qProj_f32_cache;
};

// 矩阵乘法时即时转换
void matmulBF16(const uint16_t* weight, const float* input, 
                float* output, int M, int K) {
    // 使用SIMD即时转换BF16->F32并计算
    // 避免存储完整的F32权重
}
```

**收益**:
- 权重内存: 1.89 GB → 0.95 GB (-50%)
- 总内存: 8.49 GB → ~6.5 GB

**风险**: 计算性能下降5-10%

### 3.2 方案B: 动态KV Cache (立即可实施) ⭐⭐⭐

**目标**: 减少75% KV Cache内存

**实现**:
```cpp
struct KVCache {
    // 动态增长，而非预分配
    std::vector<float> kCache;
    std::vector<float> vCache;
    int currentLen = 0;
    int allocatedLen = 0;
    
    void ensureCapacity(int requiredLen) {
        if (requiredLen > allocatedLen) {
            // 按2倍增长策略
            int newSize = std::max(requiredLen, allocatedLen * 2);
            kCache.resize(newSize * numKVHeads * headDim);
            vCache.resize(newSize * numKVHeads * headDim);
            allocatedLen = newSize;
        }
    }
};
```

**收益**:
- KV Cache: 0.44 GB → 0.11 GB (avg_seq=1024)
- 总内存: 8.49 GB → ~7.5 GB

**风险**: 无，动态扩容有轻微性能开销

### 3.3 方案C: 内存映射权重 (立即可实施) ⭐⭐

**目标**: 消除权重拷贝

**实现**:
```cpp
// 直接使用内存映射的BF16权重，不转换
class CPUBackend {
    const uint16_t* qProj_mapped;  // 指向mmap内存
    const uint16_t* kProj_mapped;
    // ...
};

// 加载时只保存指针，不拷贝数据
bool loadWeights(const ModelWeights& weights) {
    // weights已经是内存映射的指针
    qProj_mapped = static_cast<const uint16_t*>(weights.qProj);
    // ...
}
```

**收益**:
- 消除1.89 GB堆内存分配
- 总内存: 8.49 GB → ~6.6 GB

**风险**: 需要确保内存映射生命周期

### 3.4 方案D: 工作缓冲区合并 (立即可实施) ⭐

**目标**: 减少工作缓冲区数量

**实现**:
```cpp
// 合并功能重复的缓冲区
union Workspace {
    struct {
        float hidden[1024];
        float residual[1024];
    };
    struct {
        float bufferA[1024];
        float bufferB[1024];
    };
};
// 或使用std::variant动态管理
```

**收益**:
- 工作缓冲区: 50 MB → 20 MB
- 总内存: 微小减少

### 3.5 方案E: INT8量化 (中期实施) ⭐⭐⭐⭐

**目标**: 减少75%权重内存

**实现**:
```cpp
struct QuantizedWeights {
    std::vector<int8_t> qProj_q8;  // INT8量化
    std::vector<float> qProj_scale; // 每层一个scale
};

// 使用ARM NEON dotprod加速INT8矩阵乘法
void matmulQ8(const int8_t* weight, const float* input,
              float* output, int M, int K, float scale);
```

**收益**:
- 权重内存: 1.89 GB → 0.47 GB (-75%)
- 总内存: 8.49 GB → ~5.0 GB

**风险**: 需要验证精度损失

## 4. 优化实施优先级

### Phase 1: 立即实施 (预期节省 2-3 GB)
1. **动态KV Cache** (P0) - 节省 0.33 GB
2. **混合精度权重** (P0) - 节省 0.95 GB
3. **内存映射权重** (P1) - 节省 1.89 GB (与方案2冲突，二选一)

### Phase 2: 短期实施 (预期再节省 0.5 GB)
4. **工作缓冲区合并** (P2)
5. **内存池管理** (P2)

### Phase 3: 中期实施 (预期再节省 1.5 GB)
6. **INT8量化** (P3)
7. **权重共享** (P3) - 如果tieWordEmbeddings

## 5. 预期效果

| 阶段 | 优化措施 | 内存节省 | 累计内存 |
|------|----------|----------|----------|
| 当前 | - | - | 8.49 GB |
| Phase 1 | 动态KV + BF16权重 | 1.28 GB | 7.21 GB |
| Phase 2 | 缓冲区合并 | 0.30 GB | 6.91 GB |
| Phase 3 | INT8量化 | 1.42 GB | 5.49 GB |
| **目标** | **全部优化** | **3.0 GB** | **5.5 GB** |

## 6. 验证方法

```bash
# 基准测试
/usr/bin/time -l ./bin/show_model_output --device cpu

# 检查内存峰值
# 关注: maximum resident set size

# 多轮测试
for i in {1..5}; do
    /usr/bin/time -l ./bin/show_model_output --device cpu 2>&1 | grep "maximum resident"
done

# 同时监测输出正确性
./bin/compare_cpu_gpu_layers
```

## 7. 结论

当前CPU Backend内存使用存在**3.62倍**的理论开销，主要来自：
1. **权重双重存储** (BF16 mmap + F32 heap) - 最大问题
2. **KV Cache过度预分配** - 次要问题
3. **工作缓冲区冗余** - 轻微问题

通过实施**混合精度权重** + **动态KV Cache**，可立即将内存从8.49 GB降至约7.2 GB (-15%)。
通过**INT8量化**，可进一步降至约5.5 GB (-35%)。
