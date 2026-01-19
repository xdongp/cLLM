# cLLM CPU vs GPU 性能对比测试报告

## 测试环境

### 硬件信息
- **设备**: MacBook Air
- **芯片**: Apple M3
- **内存**: 8GB 统一内存
- **操作系统**: macOS

### 软件环境
- **cLLM版本**: 最新版本
- **llama.cpp版本**: 重新编译版本（启用Metal GPU）
- **编译器**: Clang
- **CMake版本**: 最新版本
- **Python版本**: 3.x

### 编译配置
```bash
cmake -DLLAMA_METAL=ON \
      -DLLAMA_ACCELERATE=ON \
      -DLLAMA_NEON=ON \
      -DLLAMA_F16C=ON \
      -DLLAMA_AVX=OFF \
      -DLLAMA_AVX2=OFF \
      -DLLAMA_AVX512=OFF \
      -DLLAMA_BLAS=OFF \
      -DLLAMA_LAPACK=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=../install \
      -DLLAMA_STATIC=OFF \
      -DLLAMA_SHARED=ON \
      ..
```

### 启用的优化
- ✅ **ARM NEON SIMD**: 启用（针对Apple M3芯片优化）
- ✅ **F16C指令集**: 启用（半精度浮点运算）
- ✅ **Accelerate框架**: 启用（使用Apple的BLAS/LAPACK实现）
- ✅ **Metal GPU**: 启用（使用Apple Metal API）
- ❌ **x86 SIMD**: 未启用（ARM架构不适用）

### 模型配置
- **模型**: Qwen3 0.6B (Q4_K_M量化)
- **模型文件**: qwen3-0.6b-q4_k_m.gguf
- **词汇表大小**: 151,936
- **隐藏层大小**: 1024
- **层数**: 28
- **注意力头数**: 8
- **KV头数**: 8
- **最大序列长度**: 40,960

### 测试参数
- **请求数**: 16
- **并发度**: 8
- **每个请求token数**: 50
- **测试类型**: 顺序测试 + 并发测试

## 测试配置

### CPU配置 (config/config_cpu.yaml)
```yaml
backend:
  llama_cpp:
    n_batch: 512
    n_threads: 8
    n_gpu_layers: 0  # 纯CPU模式
    n_seq_max: 8
    use_mmap: true
    use_mlock: false
```

### GPU配置 (config/config_gpu.yaml)
```yaml
backend:
  llama_cpp:
    n_batch: 512
    n_threads: 8
    n_gpu_layers: 99  # 全部使用GPU层
    n_seq_max: 8
    use_mmap: true
    use_mlock: false
```

## 测试结果

### CPU性能测试结果

#### 顺序测试（16个请求，每个50 tokens）

| 指标 | 值 |
|------|-----|
| 总请求数 | 16 |
| 成功请求 | 16 |
| 失败请求 | 0 |
| 平均响应时间 | 1.42s |
| 最小响应时间 | 1.17s |
| 最大响应时间 | 1.77s |
| 平均吞吐量 | 35.19 tokens/sec |
| 平均每秒token数 | 35.55 tokens/sec |
| 总处理token数 | 1,498 |
| 平均生成token数 | 50.00 |

**总时间**: 22.73s

#### 并发测试（16个请求，并发度8，每个50 tokens）

| 指标 | 值 |
|------|-----|
| 总请求数 | 16 |
| 成功请求 | 16 |
| 失败请求 | 0 |
| 平均响应时间 | 9.47s |
| 最小响应时间 | 1.68s |
| 最大响应时间 | 11.32s |
| 平均吞吐量 | 56.97 tokens/sec |
| 平均每秒token数 | 8.53 tokens/sec |
| 总处理token数 | 1,827 |
| 平均生成token数 | 70.56 |

**总时间**: 19.82s

### GPU性能测试结果

#### 顺序测试（16个请求，每个50 tokens）

| 指标 | 值 |
|------|-----|
| 总请求数 | 16 |
| 成功请求 | 16 |
| 失败请求 | 0 |
| 平均响应时间 | 1.21s |
| 最小响应时间 | 1.12s |
| 最大响应时间 | 1.26s |
| 平均吞吐量 | 41.45 tokens/sec |
| 平均每秒token数 | 41.56 tokens/sec |
| 总处理token数 | 1,498 |
| 平均生成token数 | 50.00 |

**总时间**: 19.30s

#### 并发测试（16个请求，并发度8，每个50 tokens）

| 指标 | 值 |
|------|-----|
| 总请求数 | 16 |
| 成功请求 | 16 |
| 失败请求 | 0 |
| 平均响应时间 | 7.12s |
| 最小响应时间 | 1.17s |
| 最大响应时间 | 8.15s |
| 平均吞吐量 | 74.40 tokens/sec |
| 平均每秒token数 | 11.38 tokens/sec |
| 总处理token数 | 1,810 |
| 平均生成token数 | 69.50 |

**总时间**: 14.95s

## 性能对比分析

### 顺序测试性能对比

| 指标 | CPU | GPU | 提升 |
|------|-----|-----|------|
| 平均响应时间 | 1.42s | 1.21s | **14.79%** |
| 平均吞吐量 | 35.19 tokens/sec | 41.45 tokens/sec | **17.79%** |
| 平均每秒token数 | 35.55 tokens/sec | 41.56 tokens/sec | **16.91%** |
| 总时间 | 22.73s | 19.30s | **15.09%** |

**分析**:
- GPU在顺序测试中表现优于CPU
- 平均响应时间减少了14.79%
- 吞吐量提升了17.79%
- 总处理时间减少了15.09%

### 并发测试性能对比

| 指标 | CPU | GPU | 提升 |
|------|-----|-----|------|
| 平均响应时间 | 9.47s | 7.12s | **24.82%** |
| 平均吞吐量 | 56.97 tokens/sec | 74.40 tokens/sec | **30.59%** |
| 平均每秒token数 | 8.53 tokens/sec | 11.38 tokens/sec | **33.41%** |
| 总时间 | 19.82s | 14.95s | **24.57%** |

**分析**:
- GPU在并发测试中表现显著优于CPU
- 平均响应时间减少了24.82%
- 吞吐量提升了30.59%
- 总处理时间减少了24.57%
- 并发场景下GPU加速效果更明显

### 综合性能对比

| 测试类型 | 平均响应时间 | 吞吐量 | 总时间 |
|----------|--------------|--------|--------|
| CPU顺序 | 1.42s | 35.19 tokens/sec | 22.73s |
| GPU顺序 | 1.21s | 41.45 tokens/sec | 19.30s |
| CPU并发 | 9.47s | 56.97 tokens/sec | 19.82s |
| GPU并发 | 7.12s | 74.40 tokens/sec | 14.95s |

**关键发现**:
1. **GPU加速有效**: 在所有测试场景中，GPU都比CPU表现更好
2. **并发场景优势更明显**: GPU在并发测试中的性能提升（30.59%）高于顺序测试（17.79%）
3. **响应时间优化**: GPU显著减少了响应时间，特别是在高并发场景下
4. **吞吐量提升**: GPU显著提升了整体吞吐量，适合处理大量并发请求

## Metal GPU加速效果分析

### 为什么GPU加速有效

1. **并行计算能力**: Metal GPU可以并行处理多个神经网络层的计算
2. **内存带宽优势**: GPU具有更高的内存带宽，适合处理大量数据传输
3. **统一内存架构**: Apple M3的统一内存架构减少了CPU-GPU数据传输开销
4. **批处理优化**: GPU可以更高效地处理批处理请求

### 性能提升来源

1. **矩阵运算加速**: GPU擅长处理神经网络中的矩阵乘法运算
2. **注意力机制优化**: GPU可以并行计算注意力权重
3. **KV缓存管理**: GPU可以更高效地管理键值缓存
4. **量化计算**: GPU对量化后的模型计算更高效

## 实际应用建议

### 推荐使用场景

1. **高并发场景**: 如果需要处理大量并发请求，强烈推荐使用GPU
2. **批量处理**: 如果需要批量生成文本，GPU可以显著提升吞吐量
3. **实时服务**: 如果需要快速响应用户请求，GPU可以减少响应时间

### 配置建议

1. **n_gpu_layers**: 根据模型层数和内存情况调整
   - 对于Qwen3 0.6B（28层），推荐使用99层（全部GPU）
   - 对于更大的模型，可能需要调整以适应内存限制

2. **n_threads**: 根据CPU核心数调整
   - 对于Apple M3（8核心），推荐使用8线程

3. **n_batch**: 根据内存和并发需求调整
   - 推荐使用512以平衡内存和性能

4. **并发度**: 根据硬件能力调整
   - 对于Apple M3，推荐使用8-16的并发度

## 结论

### 测试总结

1. **Metal GPU加速成功**: 成功编译并启用了Metal GPU加速
2. **性能提升显著**: GPU比CPU平均提升约20-30%的性能
3. **并发场景优势**: GPU在高并发场景下表现更优
4. **稳定性良好**: 所有测试都达到了100%的成功率

### 性能指标

- **顺序测试**: GPU比CPU快14.79%
- **并发测试**: GPU比CPU快24.82%
- **吞吐量**: GPU比CPU高30.59%
- **总时间**: GPU比CPU少24.57%

### 建议

1. **生产环境**: 推荐使用GPU加速以获得最佳性能
2. **开发环境**: 可以使用CPU以节省资源
3. **混合配置**: 可以根据实际需求调整GPU层数，平衡性能和内存使用
4. **持续优化**: 可以进一步调整配置参数以获得更好的性能

## 附录

### 测试命令
```bash
# CPU测试
python3 tools/cllm_optimized_benchmark.py \
  --server-url http://localhost:18084 \
  --test-type all \
  --requests 16 \
  --concurrency 8 \
  --max-tokens 50

# GPU测试
python3 tools/cllm_optimized_benchmark.py \
  --server-url http://localhost:18085 \
  --test-type all \
  --requests 16 \
  --concurrency 8 \
  --max-tokens 50
```

### 配置文件
- **CPU配置**: config/config_cpu.yaml
- **GPU配置**: config/config_gpu.yaml

### 日志文件
- **CPU日志**: logs/cllm_server_cpu_18084.log
- **GPU日志**: logs/cllm_server_gpu_18085.log

---

**报告生成时间**: 2026-01-19
**测试人员**: cLLM Team
**报告版本**: 1.0
