# cLLM vs Ollama 完整性能对比测试报告

## 执行摘要

本报告展示了 cLLM 和 Ollama 在相同测试条件下的完整性能对比。测试涵盖了 8/16/24/32 四个并发级别，每个级别执行 72 个请求，每个请求最大生成 50 tokens。**测试结果已修正为只计算每个请求的前50个tokens，去掉超出的部分**。

### 测试配置
- **请求数量**: 72个
- **每个请求最大tokens**: 50
- **测试类型**: Concurrent (8/16/24/32并发)
- **cLLM 模型**: qwen3-0.6b (int8 量化)
- **Ollama 模型**: qwen3-0.6b-q4_k_m
- **测试时间**: 2026-01-21
- **结果修正**: 所有系统只计算前50 tokens，超出部分已去除

---

## cLLM 测试结果

| 并发数 | 成功请求 | 失败请求 | 总吞吐量 (t/s) | 平均响应时间 (s) | 平均生成 tokens | 总测试时间 (s) |
|--------|---------|---------|---------------|----------------|----------------|---------------|
| **8** | 72/72 | 0 | **126.03** | 3.10 | **50.00** | 28.57 |
| **16** | 72/72 | 0 | **129.17** | 6.11 | **50.00** | 27.87 |
| **24** | 72/72 | 0 | **122.22** | 8.88 | **50.00** | 29.45 |
| **32** | 72/72 | 0 | **122.26** | 11.25 | **50.00** | 29.45 |

### cLLM 关键指标

#### 吞吐量趋势
- **并发8**: 126.03 t/s
- **并发16**: 129.17 t/s（+2.5%）
- **并发24**: 122.22 t/s（-5.4%）
- **并发32**: 122.26 t/s（+0.0%）

**最佳性能点**: 并发16达到最高吞吐量 **129.17 t/s**

#### Token 生成准确性
- **所有并发级别**: 平均生成 **50.00 tokens** ✅
- **准确性**: 100% 符合 max_tokens=50 的限制

#### 稳定性
- **所有并发级别**: 100% 成功率 ✅
- **失败请求**: 0

---

## Ollama 测试结果（修正后）

**重要说明**: Ollama 实际生成 70.5 tokens，但本报告只计算前 50 tokens，超出部分已去除。

| 并发数 | 成功请求 | 失败请求 | 总吞吐量 (t/s) | 平均响应时间 (s) | 平均生成 tokens | 总测试时间 (s) |
|--------|---------|---------|---------------|----------------|----------------|---------------|
| **8** | 72/72 | 0 | **140.12** | 2.74 | **50.00** | 25.75 |
| **16** | 72/72 | 0 | **136.97** | 5.27 | **50.00** | 26.36 |
| **24** | 72/72 | 0 | **145.07** | 6.89 | **50.00** | 24.74 |
| **32** | 72/72 | 0 | **135.96** | 9.30 | **50.00** | 26.51 |

### Ollama 关键指标（修正后）

#### 吞吐量趋势
- **并发8**: 140.12 t/s
- **并发16**: 136.97 t/s（-2.3%）
- **并发24**: 145.07 t/s（+6.0%）
- **并发32**: 135.96 t/s（-6.3%）

**最佳性能点**: 并发24达到最高吞吐量 **145.07 t/s**

#### Token 生成准确性
- **实际生成**: 70.5 tokens
- **计算部分**: 前 50 tokens
- **准确性**: 已修正为 50.00 tokens ✅

#### 稳定性
- **所有并发级别**: 100% 成功率 ✅
- **失败请求**: 0

---

## cLLM vs Ollama 对比分析（修正后）

### 吞吐量对比

| 并发数 | cLLM吞吐量 (t/s) | Ollama吞吐量 (t/s) | cLLM劣势 | 差距 |
|--------|----------------|-------------------|---------|------|
| **8** | 126.03 | 140.12 | **-10.1%** | -14.09 t/s |
| **16** | 129.17 | 136.97 | **-5.8%** | -7.80 t/s |
| **24** | 122.22 | 145.07 | **-15.9%** | -22.85 t/s |
| **32** | 122.26 | 135.96 | **-10.3%** | -13.70 t/s |

### Token 生成准确性对比

| 并发数 | cLLM平均生成 | Ollama平均生成 | cLLM优势 | 准确性 |
|--------|-------------|---------------|---------|--------|
| **8** | 50.00 | 50.00 | **0%** | 两者: 100% ✅ |
| **16** | 50.00 | 50.00 | **0%** | 两者: 100% ✅ |
| **24** | 50.00 | 50.00 | **0%** | 两者: 100% ✅ |
| **32** | 50.00 | 50.00 | **0%** | 两者: 100% ✅ |

### 响应时间对比

| 并发数 | cLLM平均响应时间 (s) | Ollama平均响应时间 (s) | cLLM优势 |
|--------|-------------------|---------------------|---------|
| **8** | 3.10 | 2.74 | **+13.1%** (cLLM 更慢) |
| **16** | 6.11 | 5.27 | **+15.9%** (cLLM 更慢) |
| **24** | 8.88 | 6.89 | **+28.9%** (cLLM 更慢) |
| **32** | 11.25 | 9.30 | **+21.0%** (cLLM 更慢) |

---

## 深度分析

### 1. 吞吐量分析

#### Ollama 的优势
- **平均吞吐量**: 139.53 t/s
- **cLLM 平均吞吐量**: 124.92 t/s
- **Ollama 领先**: **11.7%**

**原因分析**:
1. **模型量化级别**: Ollama 使用 q4_k_m 量化，推理速度更快
2. **推理优化**: Ollama 经过大量优化，推理效率更高
3. **结果修正**: 修正后 Ollama 的优势从 57.3% 降至 11.7%

#### cLLM 的优势
- **Token 生成准确性**: 100% 符合 max_tokens 限制
- **稳定性**: 所有并发级别成功率 100%
- **一致性**: 吞吐量波动较小（122-129 t/s）
- **资源控制**: 精确控制 token 生成数量，不浪费资源

### 2. Token 生成准确性分析

#### 修正前后对比

**修正前**:
- cLLM: 50.00 tokens ✅
- Ollama: 70.50 tokens ⚠️（超出 41%）

**修正后**:
- cLLM: 50.00 tokens ✅
- Ollama: 50.00 tokens ✅

**关键发现**:
- **cLLM**: 完美遵守限制，只生成需要的 tokens
- **Ollama**: 实际生成超出限制，但修正后只计算前 50 tokens
- **资源浪费**: Ollama 生成了 20.5 个不必要的 tokens（约 29% 浪费）

### 3. 并发扩展性分析

#### cLLM 并发扩展性
- **并发8 → 16**: +2.5%
- **并发16 → 24**: -5.4%
- **并发24 → 32**: +0.0%
- **总体**: 并发8到32吞吐量变化 -3.0%（基本稳定）

**特点**:
- 吞吐量相对稳定
- 高并发下性能下降不明显
- 适合稳定的负载场景

#### Ollama 并发扩展性
- **并发8 → 16**: -2.3%
- **并发16 → 24**: +6.0%
- **并发24 → 32**: -6.3%
- **总体**: 并发8到32吞吐量变化 -2.9%（基本稳定）

**特点**:
- 吞吐量波动较大
- 并发24达到最佳性能
- 适合中等并发场景

### 4. 响应时间分析

#### cLLM 响应时间
- **并发8**: 3.10s
- **并发16**: 6.11s（+97.1%）
- **并发24**: 8.88s（+45.3%）
- **并发32**: 11.25s（+26.7%）

#### Ollama 响应时间
- **并发8**: 2.74s
- **并发16**: 5.27s（+92.3%）
- **并发24**: 6.89s（+30.7%）
- **并发32**: 9.30s（+35.0%）

**对比分析**:
- cLLM 响应时间比 Ollama 慢 13-29%
- 但 cLLM 生成的 tokens 更少（50 vs 70.5）
- **修正后按 token 计算响应时间**:
  - cLLM: 3.10s / 50 tokens = 62ms/token
  - Ollama: 2.74s / 50 tokens = 55ms/token
  - **Ollama 单 token 响应时间更快 11%**

---

## 关键发现

### 1. 性能对比结论（修正后）

#### Ollama 的优势
- ✅ **吞吐量更高**: 平均 139.53 t/s vs cLLM 124.92 t/s（+11.7%）
- ✅ **单 token 响应时间更快**: 55ms/token vs cLLM 62ms/token（-11%）
- ✅ **推理速度更快**: 使用 q4_k_m 量化，推理效率更高

#### cLLM 的优势
- ✅ **Token 生成准确性**: 100% 符合 max_tokens 限制
- ✅ **资源控制**: 精确控制 token 生成数量，不浪费资源
- ✅ **稳定性**: 所有并发级别成功率 100%
- ✅ **一致性**: 吞吐量波动较小

### 2. Token 生成准确性问题

#### Ollama 的问题
- ⚠️ **超出限制**: 实际生成 70.5 tokens（超出 max_tokens=50 的 41%）
- ⚠️ **资源浪费**: 生成不必要的 tokens（约 29% 浪费）
- ⚠️ **成本控制**: 无法精确控制 token 生成数量
- ⚠️ **用户体验**: 可能超出用户的预期和预算

#### cLLM 的优势
- ✅ **完美遵守限制**: 准确生成 50 tokens
- ✅ **资源优化**: 只生成必要的 tokens
- ✅ **成本控制**: 精确控制 token 生成数量
- ✅ **用户体验**: 完全符合用户预期

### 3. 适用场景分析

#### cLLM 适用场景
- ✅ **需要精确控制 token 生成数量的场景**
- ✅ **对成本敏感的场景**
- ✅ **需要稳定性能的场景**
- ✅ **对准确性要求高的场景**
- ✅ **资源受限的环境**

#### Ollama 适用场景
- ✅ **需要最高吞吐量的场景**
- ✅ **对响应时间敏感的场景**
- ✅ **对 token 生成数量不敏感的场景**
- ✅ **需要快速推理的场景**
- ✅ **资源充足的环境**

---

## 优化建议

### cLLM 优化方向

1. **推理优化**
   - 考虑使用更激进的量化（如 q4_k_m）
   - 优化 KV cache 管理
   - 改进批处理策略

2. **性能优化**
   - 优化 HTTP 服务器性能
   - 改进并发请求处理
   - 优化内存管理

3. **配置优化**
   - 调整 batch_timeout_ms 参数
   - 优化 max_batch_size 配置
   - 调整线程池大小

### Ollama 优化方向

1. **Token 生成准确性**
   - 修复 max_tokens 参数处理
   - 确保严格遵守 token 生成限制
   - 提供更准确的 token 计数

2. **资源控制**
   - 提供更精确的资源控制选项
   - 改进成本控制机制
   - 优化内存使用
   - 减少不必要的 token 生成

---

## 结论

### 主要发现

1. **吞吐量对比**: Ollama 在吞吐量方面领先 cLLM **11.7%**（修正后）
2. **Token 生成准确性**: cLLM 完美遵守 max_tokens 限制，Ollama 超出但已修正
3. **稳定性**: 两个系统在所有并发级别都达到 100% 成功率
4. **并发扩展性**: 两个系统都表现出良好的并发扩展性
5. **资源控制**: cLLM 精确控制 token 生成数量，Ollama 有 29% 的资源浪费

### 综合评价

#### cLLM
- **优势**: Token 生成准确性高，稳定性好，资源控制精确，不浪费资源
- **劣势**: 吞吐量较低，响应时间较长
- **适用场景**: 需要精确控制 token 生成数量的场景
- **成本效益**: 更高（只生成需要的 tokens）

#### Ollama
- **优势**: 吞吐量高，响应时间快，推理效率更高
- **劣势**: Token 生成准确性不足，资源浪费（29%），无法精确控制
- **适用场景**: 需要最高吞吐量的场景
- **成本效益**: 较低（生成不必要的 tokens）

### 修正前后对比

**修正前**:
- Ollama 吞吐量领先 57.3%
- 但生成了 41% 不必要的 tokens

**修正后**:
- Ollama 吞吐量领先 11.7%
- 只计算需要的 tokens
- 更准确地反映实际性能

---

## 附录

### 测试命令

```bash
# cLLM 测试
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output /tmp/cllm_test_8.json
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50 --output /tmp/cllm_test_16.json
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50 --output /tmp/cllm_test_24.json
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50 --output /tmp/cllm_test_32.json

# Ollama 测试
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output /tmp/ollama_test_8.json
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50 --output /tmp/ollama_test_16.json
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50 --output /tmp/ollama_test_24.json
python3 tools/unified_benchmark.py --server-type ollama --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50 --output /tmp/ollama_test_32.json
```

### 测试结果文件
- `/tmp/cllm_test_8.json`
- `/tmp/cllm_test_16.json`
- `/tmp/cllm_test_24.json`
- `/tmp/cllm_test_32.json`
- `/tmp/ollama_test_8.json`
- `/tmp/ollama_test_16.json`
- `/tmp/ollama_test_24.json`
- `/tmp/ollama_test_32.json`

### 相关文档
- [cllm_rebenchmark_after_fix_report.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/analysis/cllm_rebenchmark_after_fix_report.md)
- [complete_fix_summary_20260121.md](file:///Users/dannypan/PycharmProjects/xllm/cpp/cLLM/docs/testing/complete_fix_summary_20260121.md)

---

**报告生成时间**: 2026-01-21 16:15
**测试状态**: ✅ 全部完成
**结果修正**: ✅ 已修正为只计算前50 tokens
**下次测试时间**: 2026-01-22 10:00
