# cLLM llama_cpp 后端长 Token 生成性能测试报告

## 执行摘要

本报告测试了 cLLM 使用 llama_cpp 后端 + GGUF 模型 + GPU (Metal) 加速下，不同 token 长度的生成性能。

### 测试环境

| 项目 | 配置 |
|------|------|
| **CPU** | Apple M3 |
| **内存** | 24 GB |
| **GPU** | Apple M3 (Metal) |
| **操作系统** | macOS (darwin 24.5.0) |
| **模型** | Qwen3-0.6B (Q4_K_M 量化) |
| **模型文件** | qwen3-0.6b-q4_k_m.gguf (493 MB) |
| **后端** | llama_cpp + Metal GPU |
| **上下文长度** | 8192 tokens |
| **测试时间** | 2026-02-03 |

### 测试配置

- **并发数**: 2
- **每轮请求数**: 10
- **n_gpu_layers**: 99 (全部层使用 GPU)
- **n_batch**: 512
- **n_seq_max**: 2

## 测试结果汇总

| Token 长度 | 成功率 | 总吞吐量 (t/s) | 单请求速度 (t/s) | 平均响应时间 (s) | 总测试时间 (s) |
|------------|--------|----------------|------------------|------------------|----------------|
| **512** | 100% (10/10) | **102.06** | 60.12 | 9.47 | 50.17 |
| **1024** | 100% (10/10) | **54.90** | 29.78 | 37.30 | 186.52 |
| **2048** | 100% (10/10) | **51.43** | 33.04 | 75.38 | 398.18 |
| **4096** | N/A | - | - | - | KV cache 限制 |
| **8192** | N/A | - | - | - | KV cache 限制 |

### 详细测试数据

#### 512 Tokens 测试

```
测试时间: 2026-02-03 14:18:10
请求数: 10, 并发: 2

结果:
- 成功请求: 10/10 (100%)
- 失败请求: 0
- 平均响应时间: 9.47s
- 最小响应时间: 5.66s
- 最大响应时间: 15.38s
- 总吞吐量: 102.06 tokens/sec
- 单请求平均速度: 60.12 tokens/sec
- 总生成 tokens: 5,120
- 总测试时间: 50.17s
```

#### 1024 Tokens 测试

```
测试时间: 2026-02-03 14:21:24
请求数: 10, 并发: 2

结果:
- 成功请求: 10/10 (100%)
- 失败请求: 0
- 平均响应时间: 37.30s
- 最小响应时间: 20.51s
- 最大响应时间: 43.54s
- 总吞吐量: 54.90 tokens/sec
- 单请求平均速度: 29.78 tokens/sec
- 总生成 tokens: 10,240
- 总测试时间: 186.52s
```

#### 2048 Tokens 测试

```
测试时间: 2026-02-03 14:28:14
请求数: 10, 并发: 2

结果:
- 成功请求: 10/10 (100%)
- 失败请求: 0
- 平均响应时间: 75.38s
- 最小响应时间: 27.31s
- 最大响应时间: 94.00s
- 总吞吐量: 51.43 tokens/sec
- 单请求平均速度: 33.04 tokens/sec
- 总生成 tokens: 20,480
- 总测试时间: 398.18s
```

#### 4096/8192 Tokens 测试

由于 KV cache 内存限制，在当前配置下（n_seq_max=2, max_context_length=8192）无法完成 2 并发的 4096+ tokens 测试。

**限制原因**: 
- 2 个并发请求 × 4096 tokens = 8192 tokens（刚好等于 max_context_length）
- 加上 prompt tokens，总量超过 KV cache 容量
- 服务器报错: `decode: failed to find a memory slot for batch of size 2`

## 资源使用情况

### 内存使用

| 测试阶段 | RSS 内存 (MB) | 内存占比 |
|----------|---------------|----------|
| 初始状态 | ~1,650 | 6.7% |
| 512 tokens 后 | ~1,650 | 6.7% |
| 1024 tokens 后 | ~1,735 | 7.1% |
| 2048 tokens 后 | ~1,740 | 7.1% |
| 4096 tokens 前 | ~1,745 | 7.1% |

### CPU 使用

- 测试期间 CPU 使用率稳定在 **~20%**
- Metal GPU 承担主要计算任务

### GPU 使用

- 所有 28 层模型全部加载到 Metal GPU
- Metal compute buffer: ~310 MB
- 推荐最大工作集: 17,179 MB

## 性能分析

### 吞吐量趋势

```
Token 长度    总吞吐量 (t/s)
   512  ████████████████████████████████████████████ 102.06
  1024  ██████████████████████████ 54.90
  2048  █████████████████████████ 51.43
```

### 关键发现

1. **短 token 高效**: 512 tokens 时吞吐量达到 **102.06 t/s**，性能最优
2. **长 token 稳定**: 1024-2048 tokens 范围内吞吐量稳定在 **51-55 t/s**
3. **并发支持有限**: 当前配置下，2 并发最多支持到 2048 tokens
4. **100% 成功率**: 在支持范围内，所有测试均 100% 成功

### 性能瓶颈

1. **KV Cache 限制**: 长 token + 高并发需要更大的 KV cache 空间
2. **上下文长度**: max_context_length=8192 限制了单请求最大 token 数
3. **并发数与 token 长度的权衡**: 需要根据使用场景调整 n_seq_max

## 配置建议

### 短 token 高并发场景 (≤512 tokens)

```yaml
llama_cpp:
  n_seq_max: 8      # 支持更多并发
  n_gpu_layers: 99
```

预期性能: ~100+ t/s，支持 8 并发

### 中等 token 场景 (512-2048 tokens)

```yaml
llama_cpp:
  n_seq_max: 2-4    # 平衡并发与 token 长度
  n_gpu_layers: 99
```

预期性能: ~50-60 t/s，2-4 并发

### 长 token 场景 (4096+ tokens)

```yaml
llama_cpp:
  n_seq_max: 1      # 单序列以支持最长 token
  n_gpu_layers: 99
```

预期性能: ~30-40 t/s，单并发

## 与目标对比

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 512 tokens 吞吐量 | 80+ t/s | 102.06 t/s | ✅ **超越目标 27.6%** |
| 1024 tokens 吞吐量 | - | 54.90 t/s | ✅ 稳定 |
| 2048 tokens 吞吐量 | - | 51.43 t/s | ✅ 稳定 |
| 成功率 | 100% | 100% | ✅ 达标 |

## 结论

1. **llama_cpp + GGUF + Metal GPU 组合性能优秀**
   - 短 token (512) 吞吐量超过 100 t/s
   - 中长 token (1024-2048) 稳定在 50+ t/s

2. **稳定性良好**
   - 在支持范围内，100% 成功率
   - 内存使用稳定，无明显增长

3. **存在长 token + 高并发限制**
   - 受 KV cache 容量限制
   - 需要根据实际需求调整配置

## 附录

### 测试命令

```bash
# 512 tokens
python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8085 \
  --test-type api-concurrent --requests 10 --concurrency 2 --max-tokens 512

# 1024 tokens
python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8085 \
  --test-type api-concurrent --requests 10 --concurrency 2 --max-tokens 1024

# 2048 tokens
python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8085 \
  --test-type api-concurrent --requests 10 --concurrency 2 --max-tokens 2048
```

### 配置文件

使用配置: `config/config_llama_cpp.yaml`

### 测试结果文件

- `results/llama_cpp_long_tokens_20260203/tokens_512.json`
- `results/llama_cpp_long_tokens_20260203/tokens_1024.json`
- `results/llama_cpp_long_tokens_20260203/tokens_2048.json`
- `results/llama_cpp_long_tokens_20260203/resource_log.txt`
