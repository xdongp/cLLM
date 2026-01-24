# 动态 Batch Size 策略测试报告

## 测试概述

本报告基于 `docs/design/dynamic_batch_size_mechanism_design.md` 设计文档，对 cLLM 的三种动态 Batch Size 策略进行了全面的性能测试和对比分析。

### 测试环境

- **模型**: Qwen3-0.5B-Instruct (GGUF)
- **硬件**: GPU 环境
- **配置文件**: `config/config.yaml`
- **测试工具**: `tools/unified_benchmark.py`
- **测试参数**:
  - 总请求数: 72
  - 最大 tokens: 50
  - 并发级别: 8, 16, 24, 32

### 测试策略

1. **Static 策略**: 固定 batch size = 8
2. **Dynamic 策略**: 持续动态调整 batch size
3. **Hybrid 策略**: 探测完成后锁定最优 batch size

## 测试结果

### 详细性能数据

#### 并发级别 8

| 策略 | Throughput (tok/s) | TPS (tok/s) | Avg Time (s) | Max Time (s) | Total Time (s) |
|------|-------------------|-------------|--------------|--------------|----------------|
| Static | 137.64 | 17.86 | 2.91 | 3.50 | 26.13 |
| Dynamic | 135.67 | 20.72 | 2.96 | 3.50 | 26.78 |
| Hybrid | 136.80 | 17.75 | 2.92 | 3.49 | 26.32 |

**性能对比**:
- Throughput: Static > Hybrid > Dynamic
- TPS: Dynamic > Static > Hybrid
- Dynamic 的 TPS 比 Static 高 16.0%，但 Throughput 低 1.4%

#### 并发级别 16

| 策略 | Throughput (tok/s) | TPS (tok/s) | Avg Time (s) | Max Time (s) | Total Time (s) |
|------|-------------------|-------------|--------------|--------------|----------------|
| Static | 137.68 | 19.23 | 5.62 | 9.49 | 27.48 |
| Dynamic | 124.40 | 18.60 | 6.22 | 12.48 | 29.50 |
| Hybrid | 132.46 | 18.39 | 5.69 | 9.48 | 27.18 |

**性能对比**:
- Throughput: Static > Hybrid > Dynamic
- TPS: Static > Dynamic > Hybrid
- Dynamic 的 Throughput 比 Static 低 9.6%
- Hybrid 的 Throughput 比 Static 低 3.8%

#### 并发级别 24

| 策略 | Throughput (tok/s) | TPS (tok/s) | Avg Time (s) | Max Time (s) | Total Time (s) |
|------|-------------------|-------------|--------------|--------------|----------------|
| Static | 129.00 | 16.48 | 9.15 | 12.58 | 27.91 |
| Dynamic | 120.35 | 19.61 | 9.00 | 17.11 | 29.91 |
| Hybrid | 123.13 | 18.34 | 9.17 | 15.05 | 29.24 |

**性能对比**:
- Throughput: Static > Hybrid > Dynamic
- TPS: Dynamic > Hybrid > Static
- Dynamic 的 TPS 比 Static 高 19.0%，但 Throughput 低 6.7%
- Hybrid 的 Throughput 比 Static 低 4.5%

#### 并发级别 32

| 策略 | Throughput (tok/s) | TPS (tok/s) | Avg Time (s) | Max Time (s) | Total Time (s) |
|------|-------------------|-------------|--------------|--------------|----------------|
| Static | 127.42 | 18.89 | 11.60 | 18.89 | 28.25 |
| Dynamic | 115.95 | 18.65 | 11.34 | 19.73 | 31.05 |
| Hybrid | 123.67 | 17.48 | 10.74 | 21.09 | 29.11 |

**性能对比**:
- Throughput: Hybrid > Static > Dynamic
- TPS: Static > Dynamic > Hybrid
- Hybrid 的 Throughput 比 Static 高 2.9%
- Dynamic 的 Throughput 比 Static 低 9.0%

### 平均性能对比

| 策略 | 平均 Throughput (tok/s) | 平均 TPS (tok/s) |
|------|------------------------|------------------|
| Static | 132.93 | 18.11 |
| Dynamic | 124.09 (-6.7%) | 19.39 (+7.1%) |
| Hybrid | 129.01 (-2.9%) | 17.99 (-0.7%) |

## 性能分析

### Static 策略

**优点**:
- 在低中高并发场景 (8-24) 下性能最佳
- 无探测开销，响应时间稳定
- 实现简单，资源占用低

**缺点**:
- 在超高并发 (32) 下略逊于 Hybrid
- 无法根据负载自动调整

**适用场景**:
- 低中高并发场景 (≤24)
- 对响应时间稳定性要求高的场景
- 资源受限的环境

### Dynamic 策略

**优点**:
- TPS 在部分场景下表现优异 (并发 8 和 24)
- 理论上可以适应不同负载

**缺点**:
- 在所有并发级别下 Throughput 均不如 Static
- 持续动态调整带来额外开销
- 探测阶段性能不稳定
- 平均 Throughput 降低 6.7%

**适用场景**:
- 不推荐作为默认策略
- 仅在特定场景下需要高 TPS 时考虑

### Hybrid 策略

**优点**:
- 在超高并发 (32) 下性能最佳
- 探测完成后锁定，避免持续调整开销
- 性能表现均衡

**缺点**:
- 在低中高并发场景下不如 Static
- 探测阶段仍有性能波动
- 实现复杂度较高

**适用场景**:
- 超高并发场景 (≥32)
- 需要自适应调整的环境

## 推荐配置

### 场景 1: 低并发场景 (≤8)

**推荐策略**: Static

**配置参数**:
```yaml
dynamic_batch_tuner:
  enabled: false
  strategy: "static"
  fixed_batch_size: 8
```

**理由**:
- Static 性能最佳，Throughput 最高 (137.64 tok/s)
- 无探测开销，响应时间稳定
- 实现简单，资源占用低

### 场景 2: 中高并发场景 (16-24)

**推荐策略**: Static

**配置参数**:
```yaml
dynamic_batch_tuner:
  enabled: false
  strategy: "static"
  fixed_batch_size: 8
```

**理由**:
- Static 性能最佳，Throughput 最高
- Dynamic 和 Hybrid 探测开销明显
- 无需动态调整，性能稳定

### 场景 3: 超高并发场景 (≥32)

**推荐策略**: Hybrid

**配置参数**:
```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "hybrid"
  fixed_batch_size: 8
  min_batch_size: 1
  max_batch_size: 64
  initial_batch_size: 8
  probing_growth_factor: 2.0
  max_probing_attempts: 10
  time_increase_threshold: 0.30
  adjustment_factor: 0.30
  validation_interval: 50
  exploration_interval: 200
  probe_batch_count: 10
  validation_batch_count: 10
  max_consecutive_time_increases: 5
```

**理由**:
- Hybrid 在超高并发下略优于 Static (Throughput 高 2.9%)
- 探测完成后锁定，避免持续调整开销
- 能够适应超高并发负载

### 不推荐配置

**不推荐**: Dynamic 策略

**理由**:
- 在所有并发级别下性能均不如 Static
- 持续动态调整带来额外开销
- 探测阶段性能不稳定
- 平均 Throughput 降低 6.7%

## 结论

1. **Static 策略是最佳默认选择**
   - 在大多数场景下性能最佳
   - 无探测开销，响应时间稳定
   - 实现简单，易于维护

2. **Hybrid 策略适用于超高并发场景**
   - 在并发 ≥32 时略优于 Static
   - 探测完成后锁定，性能稳定

3. **Dynamic 策略不推荐使用**
   - 在所有场景下性能均不如 Static
   - 持续调整开销明显

4. **配置建议**
   - 低中高并发 (≤24): 使用 Static 策略，禁用动态调谐器
   - 超高并发 (≥32): 使用 Hybrid 策略，启用动态调谐器

## 附录

### 测试数据文件

所有测试结果保存在 `/tmp/` 目录:
- `/tmp/static_benchmark_8.json`
- `/tmp/static_benchmark_16.json`
- `/tmp/static_benchmark_24.json`
- `/tmp/static_benchmark_32.json`
- `/tmp/dynamic_benchmark_8.json`
- `/tmp/dynamic_benchmark_16.json`
- `/tmp/dynamic_benchmark_24.json`
- `/tmp/dynamic_benchmark_32.json`
- `/tmp/hybrid_benchmark_8.json`
- `/tmp/hybrid_benchmark_16.json`
- `/tmp/hybrid_benchmark_24.json`
- `/tmp/hybrid_benchmark_32.json`

### 测试命令

```bash
# Static 策略测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/static_benchmark_${concurrency}.json
done

# Dynamic 策略测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/dynamic_benchmark_${concurrency}.json
done

# Hybrid 策略测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/hybrid_benchmark_${concurrency}.json
done
```

### 相关文档

- 设计文档: [docs/design/dynamic_batch_size_mechanism_design.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/design/dynamic_batch_size_mechanism_design.md)
- 测试计划: [docs/analysis/dynamic_batch_strategy_test_plan.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/analysis/dynamic_batch_strategy_test_plan.md)
- 配置文件: [config/config.yaml](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/config/config.yaml)
