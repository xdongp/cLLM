# CPU Backend 性能优化方案

## 1. 现状分析

### 1.1 基准测试数据

| 指标 | 当前值 |
|------|--------|
| Token生成速度 | 25.13 tokens/s |
| 模型 | Qwen3-0.6B |
| 配置 | hidden=1024, layers=28, heads=16 |
| 硬件 | Apple Silicon (M系列) |

### 1.2 代码结构分析

```
cpu_backend.cpp (781行)
├── CPUBackendImpl 结构
│   ├── 权重存储 (LayerWeights + 全局权重)
│   ├── KV Cache管理 (unordered_map)
│   ├── 工作缓冲区 (预分配)
│   └── RoPE预计算
├── 核心计算函数
│   ├── rmsNorm() - RMS归一化
│   ├── matmulF32/BF16() - 矩阵乘法
│   ├── applyRoPE() - 位置编码
│   ├── attention() - 注意力计算
│   ├── ffn() - 前馈网络
│   └── softmax/silu() - 激活函数
└── forward() - 主推理流程

ggml_kernels.cpp (1275行)
├── SIMD优化 (NEON/AVX2)
├── BLAS加速 (Apple Accelerate)
├── OpenMP并行
└── BF16/F32转换
```

## 2. 性能瓶颈分析

### 2.1 内存使用问题

#### 问题1: 权重存储格式
```cpp
// 当前: 所有权重转为F32存储
std::vector<float> qProj;  // BF16 -> F32 转换，内存翻倍

// 问题:
// - 内存占用翻倍 (BF16权重转为F32)
// - 缓存利用率低
// - 带宽压力大
```

#### 问题2: 频繁的内存拷贝
```cpp
// forward() 中的拷贝操作
std::copy(hiddenStates.begin(), hiddenStates.end(), residual.begin());
// 每层2次拷贝 × 28层 = 56次拷贝/Token
// 每次拷贝: 1024 floats × 4 bytes = 4KB
// 总计: 224KB/Token 的内存拷贝
```

#### 问题3: KV Cache内存布局
```cpp
// 当前: 线性存储，非连续访问
size_t layerOffset = layerIdx * maxSeqLen * nKVHeads * headDim;
size_t posOffset = startPos * nKVHeads * headDim;
// Attention计算时跨层访问，缓存不友好
```

### 2.2 计算瓶颈

#### 瓶颈1: 矩阵乘法 (占70%+时间)
```cpp
// 每层3次QKV投影 + 1次Output投影 + 3次FFN
// 总计: 7次矩阵乘法/层 × 28层 = 196次/Token
// 矩阵大小: [hidden, hidden] × [hidden, 1]
```

#### 瓶颈2: Attention计算
```cpp
// Q×K^T: [nHeads, headDim] × [kvLen, headDim]
// 复杂度: O(nHeads × headDim × kvLen)
// 随序列长度增长，计算量剧增
```

#### 瓶颈3: BF16转换开销
```cpp
// 权重加载时转换
convert_bf16_to_f32(src, dst, size);  // 315个张量全部转换
// 启动时耗时，且增加内存占用
```

### 2.3 并行化问题

#### 问题1: OpenMP使用不当
```cpp
// 当前: 只在matmul和attention中使用
#pragma omp parallel for if(nHeads >= 4)
// 问题: 小矩阵并行开销大，线程同步成本高
```

#### 问题2: 批处理支持有限
```cpp
// forwardBatch只是循环调用forward
for (size_t i = 0; i < batchInputIds.size(); ++i) {
    results.push_back(forward(batchInputIds[i], requestIds[i]));
}
// 没有真正的并行批处理
```

## 3. 优化方案

### 3.1 内存优化 (优先级: P0)

#### 优化1: 混合精度权重存储
```cpp
// 方案: 保持BF16存储，计算时即时转换
struct LayerWeights {
    std::vector<uint16_t> qProj_bf16;  // BF16存储
    std::vector<float> qProj_f32_cache; // 可选: 热点权重缓存
};

// 计算时:
// 1. 使用SIMD即时转换BF16->F32
// 2. 对高频使用权重预转F32缓存
// 预期收益: 内存减少40-50%
```

#### 优化2: 消除残差拷贝
```cpp
// 方案: 使用指针交换替代拷贝
struct LayerBuffers {
    std::vector<float> buf1;  // 主缓冲区
    std::vector<float> buf2;  // 残差缓冲区
    float* current = buf1.data();
    float* residual = buf2.data();
    
    void swap() {
        std::swap(current, residual);
    }
};

// 每层结束: swap() 替代 copy()
// 预期收益: 消除56次拷贝/Token，节省224KB内存带宽
```

#### 优化3: KV Cache重排
```cpp
// 方案: 按访问模式重排内存布局
// 当前: [layer, seq, head, dim]
// 优化: [layer, head, seq, dim] 或 [layer, head, dim, seq]

// 连续访问模式，提高缓存命中率
// 预期收益: Attention计算加速10-20%
```

### 3.2 计算优化 (优先级: P0)

#### 优化4: 矩阵乘法优化
```cpp
// 方案A: 使用更高效的BLAS调用
// - 启用Accelerate的并行BLAS
// - 小矩阵使用自定义SIMD内核

// 方案B: 权重预打包
// - 矩阵按块重排，提高缓存局部性
// - 类似im2col的权重展开

// 预期收益: 矩阵乘法加速20-30%
```

#### 优化5: Attention优化
```cpp
// 方案A: FlashAttention风格分块
void flash_attention_blocked(...) {
    const int BLOCK_M = 64;  // Q分块
    const int BLOCK_N = 64;  // K/V分块
    
    for (int i = 0; i < nHeads; i += BLOCK_M) {
        for (int j = 0; j < kvLen; j += BLOCK_N) {
            // 小块计算，O(1)内存
            // 减少内存带宽压力
        }
    }
}

// 方案B: 在线Softmax
// - 避免存储完整attention矩阵
// - 流式计算softmax

// 预期收益: Attention加速30-50%，内存减少50%
```

#### 优化6: 融合算子
```cpp
// 方案: 融合多个小算子
// 当前: RMSNorm -> Matmul -> Add (3次遍历)
// 优化: FusedRMSNormMatmulAdd (1次遍历)

// 融合:
// - RMSNorm + Matmul
// - SiLU + Mul
// - Add + RMSNorm

// 预期收益: 减少内存遍历，加速10-15%
```

### 3.3 并行优化 (优先级: P1)

#### 优化7: 细粒度并行策略
```cpp
// 方案: 动态选择并行策略
void parallel_matmul(...) {
    const int MIN_PARALLEL_SIZE = 128;  // 阈值调优
    
    if (M < MIN_PARALLEL_SIZE) {
        // 小矩阵: 串行，避免并行开销
        matmul_serial(...);
    } else {
        // 大矩阵: 并行
        #pragma omp parallel for
        matmul_parallel(...);
    }
}
```

#### 优化8: 批处理并行
```cpp
// 方案: 真正的批处理并行
void forward_batch_parallel(const std::vector<std::vector<int>>& batch) {
    #pragma omp parallel for
    for (size_t b = 0; b < batch.size(); ++b) {
        // 每个batch独立计算
        // 共享权重，独立激活
    }
}
```

### 3.4 架构优化 (优先级: P2)

#### 优化9: 算子调度器
```cpp
// 方案: 引入计算图和调度器
class CPUGraphExecutor {
    std::vector<Op*> ops;
    MemoryPool pool;
    
    void execute() {
        // 拓扑排序
        // 内存复用规划
        // 并行调度
    }
};
```

#### 优化10: 量化支持
```cpp
// 方案: INT8/INT4权重支持
struct QuantizedWeights {
    std::vector<int8_t> qProj_q8;  // INT8量化
    std::vector<float> scales;      // 缩放因子
};

// 使用dotprod指令加速INT8矩阵乘法
// 预期收益: 内存减少75%，速度提升2-4x
```

## 4. 实施计划

### Phase 1: 内存优化 (Week 1-2)
- [ ] 实现混合精度权重存储
- [ ] 消除残差拷贝
- [ ] 优化KV Cache布局
- **预期收益**: 内存减少50%，速度提升15-20%

### Phase 2: 计算优化 (Week 3-4)
- [ ] 矩阵乘法优化
- [ ] FlashAttention实现
- [ ] 基础算子融合
- **预期收益**: 速度提升30-50%

### Phase 3: 并行优化 (Week 5)
- [ ] 细粒度并行策略
- [ ] 批处理并行
- **预期收益**: 多请求场景提升2-4x

### Phase 4: 高级优化 (Week 6-8)
- [ ] 计算图调度器
- [ ] INT8量化支持
- **预期收益**: 内存减少75%，极限速度提升

## 5. 验证方法

### 5.1 性能测试
```bash
# 基准测试
./bin/show_model_output --device cpu --max_tokens 100

# 多轮测试取平均
for i in {1..5}; do
    ./bin/show_model_output --device cpu 2>&1 | grep "生成吞吐量"
done

# 内存分析
/usr/bin/time -l ./bin/show_model_output --device cpu
```

### 5.2 正确性验证
```bash
# CPU vs GPU输出对比
./bin/compare_cpu_gpu_layers

# Token序列一致性检查
# 数值精度检查 (max_diff < 1e-4)
```

### 5.3 性能指标

| 指标 | 当前 | Phase1目标 | Phase2目标 | 最终目标 |
|------|------|------------|------------|----------|
| Tokens/s | 25.13 | 30 | 35 | 50+ |
| 内存占用 | 100% | 60% | 60% | 25% |
| 首Token延迟 | - | -20% | -30% | -40% |
| 批处理效率 | 1x | 1x | 2x | 4x |

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| BF16转换精度损失 | 输出质量下降 | 充分测试，设置精度阈值 |
| 内存优化引入Bug | 崩溃/错误 | 渐进式修改，充分测试 |
| 并行优化线程竞争 | 性能下降 | 使用线程安全数据结构 |
| 融合算子维护成本 | 代码复杂度 | 模块化设计，文档完善 |

## 7. 参考资源

- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU优化参考
- [Apple Accelerate](https://developer.apple.com/documentation/accelerate) - BLAS优化
- [OpenMP最佳实践](https://www.openmp.org/resources/) - 并行优化
