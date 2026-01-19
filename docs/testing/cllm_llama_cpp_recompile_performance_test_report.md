# cLLM llama.cpp 重新编译后性能测试报告

## 测试环境

### 硬件信息
- **设备**: MacBook Air
- **芯片**: Apple M3
- **内存**: 8GB 统一内存
- **操作系统**: macOS

### 软件环境
- **cLLM版本**: 最新版本
- **llama.cpp版本**: 重新编译版本（启用SIMD优化）
- **编译器**: Clang
- **CMake版本**: 最新版本
- **Python版本**: 3.x

### 编译配置
```bash
cmake -DLLAMA_METAL=OFF \
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
- ✅ **Accelerate框架**: 启用（使用Apple的BLAS/LAPACK实现）
- ✅ **F16C指令集**: 启用（半精度浮点运算）
- ❌ **Metal GPU**: 未启用（编译时未启用）
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

### 服务器配置
```yaml
resources:
  max_batch_size: 8
  max_context_length: 2048
  kv_cache_max_size: 100
  kv_cache_max_memory_mb: 4096
  memory_limit_mb: 8192

backend:
  llama_cpp:
    n_batch: 512
    n_threads: 8
    n_gpu_layers: 0  # 纯CPU模式
    n_seq_max: 8
    use_mmap: true
    use_mlock: false
```

## 测试方法

### 测试工具
- **测试脚本**: `tools/cllm_optimized_benchmark.py`
- **测试类型**: API顺序测试、API并发测试
- **测试参数**:
  - 请求数: 5
  - 并发度: 3
  - 最大token数: 100

### 测试流程
1. 启动cLLM服务器
2. 等待服务器完全加载模型
3. 执行顺序测试（5个请求，每个100 tokens）
4. 执行并发测试（5个请求，并发度3，每个100 tokens）
5. 收集性能指标

## 测试结果

### API顺序测试（100 tokens）

| 指标 | 值 |
|------|-----|
| 总请求数 | 5 |
| 成功请求 | 5 |
| 失败请求 | 0 |
| 平均响应时间 | 3.10s |
| 最小响应时间 | 2.80s |
| 最大响应时间 | 3.48s |
| 平均吞吐量 | 32.25 tokens/sec |
| 平均每秒token数 | 32.49 tokens/sec |
| 总处理token数 | 714 |
| 平均生成token数 | 100.00 |

**详细结果**:
- Request 1/5: ✓ 3.32s - Generated: 100 tokens
- Request 2/5: ✓ 3.03s - Generated: 100 tokens
- Request 3/5: ✓ 3.48s - Generated: 100 tokens
- Request 4/5: ✓ 2.80s - Generated: 100 tokens
- Request 5/5: ✓ 2.88s - Generated: 100 tokens

**总时间**: 15.50s

### API并发测试（100 tokens，并发度3）

| 指标 | 值 |
|------|-----|
| 总请求数 | 5 |
| 成功请求 | 5 |
| 失败请求 | 0 |
| 平均响应时间 | 7.28s |
| 最小响应时间 | 3.10s |
| 最大响应时间 | 9.25s |
| 平均吞吐量 | 33.23 tokens/sec |
| 平均每秒token数 | 16.46 tokens/sec |
| 总处理token数 | 714 |
| 平均生成token数 | 100.00 |

**详细结果**:
- Request 1/5: ✓ 9.24s - Generated: 100 tokens
- Request 2/5: ✓ 9.25s - Generated: 100 tokens
- Request 3/5: ✓ 3.10s - Generated: 100 tokens
- Request 4/5: ✓ 8.99s - Generated: 100 tokens
- Request 5/5: ✓ 5.81s - Generated: 100 tokens

**总时间**: 15.05s

## 性能分析

### 顺序测试性能
- **平均响应时间**: 3.10s
- **平均吞吐量**: 32.25 tokens/sec
- **平均每秒token数**: 32.49 tokens/sec
- **成功率**: 100%

### 并发测试性能
- **平均响应时间**: 7.28s
- **平均吞吐量**: 33.23 tokens/sec
- **平均每秒token数**: 16.46 tokens/sec
- **成功率**: 100%

### 性能对比

| 测试类型 | 平均响应时间 | 吞吐量 | 每秒token数 |
|----------|--------------|--------|-------------|
| 顺序测试 | 3.10s | 32.25 tokens/sec | 32.49 tokens/sec |
| 并发测试 | 7.28s | 33.23 tokens/sec | 16.46 tokens/sec |

**分析**:
- 并发测试的吞吐量略高于顺序测试（33.23 vs 32.25 tokens/sec）
- 但并发测试的每秒token数较低（16.46 vs 32.49 tokens/sec）
- 这是因为并发测试中多个请求同时进行，单个请求的响应时间更长
- 总体来看，并发处理能够提高整体吞吐量

## 编译优化效果

### SIMD优化
- **ARM NEON**: 启用，针对Apple M3芯片优化
- **F16C**: 启用，支持半精度浮点运算
- **Accelerate**: 启用，使用Apple的BLAS/LAPACK实现

### 内存优化
- **KV Cache**: 100个槽位，4096MB内存限制
- **批处理大小**: 512
- **最大序列数**: 8

### 线程优化
- **线程数**: 8（充分利用Apple M3的8个CPU核心）
- **批处理线程**: 8

## 问题与解决方案

### 问题1: 内存分配失败
**错误信息**: `decode: failed to find a memory slot for batch of size 1`

**原因**: 初始配置的KV cache和批处理大小过大，导致内存不足

**解决方案**:
- 减少KV cache大小从200到100
- 减少KV cache内存限制从8192MB到4096MB
- 减少批处理大小从1024到512
- 减少最大序列数从16到8

### 问题2: 大token数生成失败
**现象**: 当max_tokens设置为600时，请求失败

**原因**: 内存限制和批处理配置不支持大token数生成

**解决方案**:
- 使用较小的max_tokens值（如100）
- 进一步优化内存配置

## 结论

### 编译成功
- ✅ llama.cpp成功重新编译
- ✅ 启用了SIMD优化（ARM NEON, F16C）
- ✅ 启用了Accelerate框架
- ✅ cLLM服务器成功启动

### 性能表现
- ✅ 顺序测试成功率: 100%
- ✅ 并发测试成功率: 100%
- ✅ 平均吞吐量: 32-33 tokens/sec
- ✅ 响应时间稳定

### 优化建议

#### 短期优化
1. **启用Metal GPU加速**: 重新编译llama.cpp时启用Metal，可以显著提升性能
2. **调整内存配置**: 根据实际需求优化KV cache和批处理大小
3. **增加批处理大小**: 在内存允许的情况下，增加批处理大小以提高吞吐量

#### 长期优化
1. **模型量化**: 使用更高效的量化方法（如Q4_K_S）
2. **Flash Attention**: 集成Flash Attention优化
3. **KV Cache优化**: 实现更高效的KV cache管理
4. **动态批处理**: 实现动态批处理策略

## 附录

### 测试命令
```bash
# 启动服务器
./build/bin/cllm_server \
  --model-path /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf \
  --port 18082 \
  --host 0.0.0.0 \
  --log-level info \
  --max-batch-size 8 \
  --max-context-length 2048

# 运行基准测试
python3 tools/cllm_optimized_benchmark.py \
  --server-url http://localhost:18082 \
  --test-type all \
  --requests 5 \
  --concurrency 3 \
  --max-tokens 100
```

### 配置文件
详见 `config/config.yaml`

### 日志文件
详见 `logs/cllm_server_q4k_18082.log`

---

**报告生成时间**: 2026-01-19
**测试人员**: cLLM Team
**报告版本**: 1.0
