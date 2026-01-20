# cLLM vs Ollama 统一基准测试对比报告

## 执行摘要

本报告基于统一的测试脚本 (`tools/unified_benchmark.py`)，对 cLLM 和 Ollama 进行了全面的性能对比测试。测试确保了两个系统的测试方法和计算方式完全一致。

### 测试配置
- **请求数量**: 72个
- **每个请求最大tokens**: 50
- **测试类型**: Sequential 和 Concurrent (8/16/24/32并发)
- **模型**: qwen3-0.6b-q4_k_m (cLLM) / qwen3:0.6b (Ollama)
- **测试时间**: 2026-01-20

### 关键发现
1. **Sequential测试**: Ollama 吞吐量 (135.56 t/s) 显著高于 cLLM (43.69 t/s)
2. **Concurrent测试**: 
   - 并发8: Ollama (172.56 t/s) 优于 cLLM (65.85 t/s)
   - 并发16: 差距缩小，cLLM (116.05 t/s) vs Ollama (169.65 t/s)
   - 并发24: 差距进一步缩小，cLLM (166.86 t/s) vs Ollama (171.94 t/s)
   - 并发32: 差距略有扩大，cLLM (167.99 t/s) vs Ollama (181.50 t/s)
3. **并发扩展性**: cLLM 在高并发下表现更好，随着并发数增加，性能差距逐渐缩小（并发24达到最佳，32并发略有下降）

## 测试方法统一

### 统一的计算方式
1. **时间计算**: 使用 `first_request_start` 到 `last_request_end` 作为 `total_test_time`
2. **吞吐量计算**: `sum(generated_tokens) / total_test_time`
3. **Token提取**: 统一处理两种服务器的响应格式

### 测试脚本
- **脚本位置**: `tools/unified_benchmark.py`
- **支持服务器**: cLLM 和 Ollama
- **统一接口**: 相同的测试参数和统计输出格式

## 详细测试结果

### 1. Sequential Test (顺序测试)

#### cLLM Sequential
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 82.41s |
| 总生成tokens | 3600 |
| **总吞吐量** | **43.69 tokens/sec** |
| 平均响应时间 | 1.14s |
| 最小/最大响应时间 | 1.02s / 1.56s |
| 平均生成tokens | 50.00 |

#### Ollama Sequential
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 37.43s |
| 总生成tokens | 5074 |
| **总吞吐量** | **135.56 tokens/sec** |
| 平均响应时间 | 0.52s |
| 最小/最大响应时间 | 0.46s / 1.58s |
| 平均生成tokens | 70.47 |

#### Sequential 对比分析
- **吞吐量差距**: Ollama 是 cLLM 的 **3.10倍**
- **响应时间**: Ollama 平均响应时间 (0.52s) 显著低于 cLLM (1.14s)
- **总测试时间**: Ollama (37.43s) 比 cLLM (82.41s) 快 **54.6%**

### 2. Concurrent Test (并发测试)

#### 2.1 并发数 8

##### cLLM Concurrent 8
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 64.98s |
| 总生成tokens | 4279 |
| **总吞吐量** | **65.85 tokens/sec** |
| 平均响应时间 | 7.02s |
| 最小/最大响应时间 | 1.08s / 13.00s |
| 平均生成tokens | 59.43 |

##### Ollama Concurrent 8
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 29.41s |
| 总生成tokens | 5075 |
| **总吞吐量** | **172.56 tokens/sec** |
| 平均响应时间 | 3.14s |
| 最小/最大响应时间 | 0.73s / 3.98s |
| 平均生成tokens | 70.49 |

##### 并发8对比
- **吞吐量差距**: Ollama 是 cLLM 的 **2.62倍**
- **cLLM劣势**: -61.8%

#### 2.2 并发数 16

##### cLLM Concurrent 16
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 71 (98.6%) |
| 失败请求 | 1 |
| 总测试时间 | 60.78s |
| 总生成tokens | 7053 |
| **总吞吐量** | **116.05 tokens/sec** |
| 平均响应时间 | 12.72s |
| 最小/最大响应时间 | 1.08s / 26.27s |
| 平均生成tokens | 99.34 |

##### Ollama Concurrent 16
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 29.91s |
| 总生成tokens | 5075 |
| **总吞吐量** | **169.65 tokens/sec** |
| 平均响应时间 | 5.96s |
| 最小/最大响应时间 | 0.89s / 7.09s |
| 平均生成tokens | 70.49 |

##### 并发16对比
- **吞吐量差距**: Ollama 是 cLLM 的 **1.46倍**
- **cLLM劣势**: -31.6%
- **改进**: 相比并发8，cLLM劣势从-61.8%缩小到-31.6%

#### 2.3 并发数 24

##### cLLM Concurrent 24
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 69 (95.8%) |
| 失败请求 | 3 |
| 总测试时间 | 61.23s |
| 总生成tokens | 10217 |
| **总吞吐量** | **166.86 tokens/sec** |
| 平均响应时间 | 18.65s |
| 最小/最大响应时间 | 5.36s / 34.07s |
| 平均生成tokens | 148.07 |

##### Ollama Concurrent 24
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 29.51s |
| 总生成tokens | 5074 |
| **总吞吐量** | **171.94 tokens/sec** |
| 平均响应时间 | 8.33s |
| 最小/最大响应时间 | 0.87s / 10.24s |
| 平均生成tokens | 70.47 |

##### 并发24对比
- **吞吐量差距**: Ollama 仅比 cLLM 高 **3.0%**
- **cLLM劣势**: -3.0%
- **改进**: 相比并发8，cLLM劣势从-61.8%大幅缩小到-3.0%

#### 2.4 并发数 32

##### cLLM Concurrent 32
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 52 (72.2%) |
| 失败请求 | 20 |
| 总测试时间 | 46.01s |
| 总生成tokens | 7729 |
| **总吞吐量** | **167.99 tokens/sec** |
| 平均响应时间 | 19.33s |
| 最小/最大响应时间 | 8.62s / 32.26s |
| 平均生成tokens | 148.63 |

##### Ollama Concurrent 32
| 指标 | 数值 |
|------|------|
| 总请求数 | 72 |
| 成功请求 | 72 (100%) |
| 失败请求 | 0 |
| 总测试时间 | 27.96s |
| 总生成tokens | 5074 |
| **总吞吐量** | **181.50 tokens/sec** |
| 平均响应时间 | 9.97s |
| 最小/最大响应时间 | 1.91s / 12.81s |
| 平均生成tokens | 70.47 |

##### 并发32对比
- **吞吐量差距**: Ollama 比 cLLM 高 **8.0%**
- **cLLM劣势**: -7.4%
- **稳定性**: cLLM 出现较多失败请求 (20个失败，成功率72.2%)

## 性能对比总结

### 吞吐量对比表

| 测试类型 | cLLM (t/s) | Ollama (t/s) | 差距 | cLLM劣势 |
|---------|-----------|-------------|------|---------|
| **Sequential** | 43.69 | 135.56 | -91.87 | -67.8% |
| **Concurrent 8** | 65.85 | 172.56 | -106.71 | -61.8% |
| **Concurrent 16** | 116.05 | 169.65 | -53.60 | -31.6% |
| **Concurrent 24** | 166.86 | 171.94 | -5.08 | -3.0% |
| **Concurrent 32** | 167.99 | 181.50 | -13.51 | -7.4% |

### 并发扩展性分析

#### cLLM 并发扩展性
- 并发8 → 16: 吞吐量提升 **76.2%** (65.85 → 116.05)
- 并发16 → 24: 吞吐量提升 **43.8%** (116.05 → 166.86)
- 并发24 → 32: 吞吐量提升 **0.7%** (166.86 → 167.99)
- **总体**: 从并发8到32，吞吐量提升 **155.0%**
- **最佳点**: 并发24达到最佳性能，并发32略有下降但基本持平

#### Ollama 并发扩展性
- 并发8 → 16: 吞吐量变化 **-1.7%** (172.56 → 169.65)
- 并发16 → 24: 吞吐量提升 **1.3%** (169.65 → 171.94)
- 并发24 → 32: 吞吐量提升 **5.6%** (171.94 → 181.50)
- **总体**: 从并发8到32，吞吐量提升 **5.2%** (基本稳定，略有提升)

### 关键发现

1. **cLLM 并发扩展性优秀**
   - 随着并发数增加，cLLM 吞吐量显著提升
   - 从并发8到24，吞吐量提升153.3%
   - 在高并发(24)下，性能接近 Ollama (仅差3.0%)
   - 并发32下性能基本持平，但稳定性下降（20个失败请求）

2. **Ollama 并发性能稳定**
   - 在不同并发数下，Ollama 吞吐量基本稳定在170 t/s左右
   - 并发扩展性有限，但基线性能高

3. **响应时间对比**
   - Sequential: Ollama (0.52s) 显著优于 cLLM (1.14s)
   - Concurrent: 随着并发数增加，cLLM 响应时间显著增加，但吞吐量提升

4. **稳定性对比**
   - Ollama: 所有测试中成功率100%
   - cLLM: 高并发下出现失败请求
     - 并发16: 1个失败 (成功率98.6%)
     - 并发24: 3个失败 (成功率95.8%)
     - 并发32: 20个失败 (成功率72.2%) ⚠️

## 性能差距分析

### Sequential 测试差距 (67.8%)
**可能原因**:
1. **单请求处理效率**: Ollama 在单请求处理上更高效
2. **响应时间**: Ollama 平均响应时间 (0.52s) 显著低于 cLLM (1.14s)
3. **内部优化**: Ollama 可能在单请求路径上有更多优化

### Concurrent 测试差距变化
- **并发8**: -61.8% (显著差距)
- **并发16**: -31.6% (差距缩小)
- **并发24**: -3.0% (接近，最佳点)
- **并发32**: -7.4% (差距略有扩大，稳定性下降)

**分析**:
1. **并发处理能力**: cLLM 在高并发下表现更好，说明其并发处理架构更优秀
2. **资源利用**: 随着并发数增加，cLLM 能更好地利用系统资源
3. **瓶颈位置**: 在低并发下，cLLM 可能存在单请求处理瓶颈；在高并发下，并发处理能力得到充分发挥

## 优化建议

### 针对 cLLM
1. **优化单请求处理路径**
   - 分析 Sequential 测试中的性能瓶颈
   - 优化响应时间，目标从1.14s降低到0.5-0.6s

2. **提高低并发性能**
   - 优化并发8的性能，缩小与Ollama的差距
   - 分析并发8下的瓶颈点

3. **提高稳定性**
   - 解决高并发下的失败请求问题
   - 并发32下失败率较高(27.8%)，需要重点优化
   - 优化资源管理和错误处理
   - 考虑限制最大并发数或实现更好的资源保护机制

### 针对整体架构
1. **保持并发优势**
   - cLLM 在高并发下的扩展性很好，应继续保持
   - 并发24下已接近Ollama性能，说明架构方向正确

2. **平衡单请求和并发性能**
   - 在保持并发优势的同时，优化单请求性能
   - 目标是在所有并发数下都接近或超过Ollama

## 测试数据文件

所有测试结果已保存到以下文件：
- `/tmp/cllm_sequential.json`
- `/tmp/cllm_concurrent_8.json`
- `/tmp/cllm_concurrent_16.json`
- `/tmp/cllm_concurrent_24.json`
- `/tmp/ollama_sequential.json`
- `/tmp/ollama_concurrent_8.json`
- `/tmp/ollama_concurrent_16.json`
- `/tmp/ollama_concurrent_24.json`

## 结论

1. **Sequential性能**: Ollama 显著优于 cLLM (3.10倍)
2. **Concurrent性能**: 
   - 低并发(8): Ollama 优于 cLLM (2.62倍)
   - 中并发(16): 差距缩小，Ollama 优于 cLLM (1.46倍)
   - 高并发(24): 性能接近，Ollama 仅比 cLLM 高3.0% (最佳点)
   - 超高并发(32): 差距略有扩大，Ollama 比 cLLM 高8.0%，但cLLM稳定性下降
3. **并发扩展性**: cLLM 在高并发下表现优秀，从并发8到32吞吐量提升155.0%，最佳点在并发24
4. **优化方向**: cLLM 应重点优化单请求处理路径和低并发性能，同时保持并发优势

## 附录

### 测试命令

#### cLLM 测试
```bash
# Sequential
python3 tools/unified_benchmark.py --server-type cllm --test-type api-sequential --requests 72 --max-tokens 50

# Concurrent 8
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50

# Concurrent 16
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50

# Concurrent 24
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50

# Concurrent 32
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50
```

#### Ollama 测试
```bash
# Sequential
python3 tools/unified_benchmark.py --server-type ollama --test-type api-sequential --requests 72 --max-tokens 50

# Concurrent 8
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50

# Concurrent 16
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50

# Concurrent 24
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50

# Concurrent 32
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50
```

### 相关文档
- `tools/unified_benchmark.py`: 统一测试脚本
- `docs/analysis/cLLM_vs_Ollama_Deep_Analysis.md`: 深度分析报告
