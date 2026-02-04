# cLLM + llama_cpp + GPU + GGUF 性能测试方案

## 1. 测试目标

### 1.1 主要目标
- 验证 cLLM 在 llama_cpp 后端 + GPU 加速 + GGUF 模型格式下的推理性能
- 测试系统在**不同并发级别**下的吞吐量(TPS)和延迟表现
- 评估动态批处理策略的有效性
- 建立性能基准线，为后续优化提供数据支撑

### 1.2 预期性能指标
| 指标 | 目标值 | 说明 |
|------|--------|------|
| 单请求吞吐量 | ≥ 20 tokens/s | 单用户场景下的生成速度 |
| 并发吞吐量 | ≥ 50 tokens/s | 8并发下的总吞吐量 |
| 平均延迟 | < 5s | 50 tokens生成的总耗时 |
| 成功率 | ≥ 99% | 成功完成的请求比例 |

## 2. 测试环境

### 2.1 硬件配置
- **CPU**: Apple Silicon M系列 (具体型号待测)
- **GPU**: 集成GPU (Metal加速)
- **内存**: 16GB+ (系统自动管理)
- **存储**: SSD (模型文件存储)

### 2.2 软件环境
- **操作系统**: macOS
- **cLLM版本**: 当前main分支
- **llama.cpp版本**: 集成版本
- **模型格式**: GGUF (Q4_K_M量化)
- **模型大小**: Qwen3-0.6B

### 2.3 架构配置
```
HTTP Server (cLLM)
    ↓
Scheduler (动态批处理)
    ↓
Model Executor (统一接口)
    ↓
llama.cpp Backend (GGUF + Metal GPU)
```

### 2.4 llama.cpp 后端关键参数
```yaml
backend:
  type: "llama_cpp"
  llama_cpp:
    n_gpu_layers: 99    # 全部层在GPU
    n_batch: 512         # 批处理大小
    n_seq_max: 16        # 最大并发序列
    use_mmap: true      # 内存映射加载
```

## 3. 测试场景

### 3.1 场景1：顺序请求测试 (Sequential)
**目的**: 测试单请求下的稳定性能和延迟表现

**配置**:
- 并发数: 1 (顺序执行)
- 请求数: 72
- 最大生成长度: 50 tokens
- 测试prompts: 5个中文文本

**测试内容**:
1. 冷启动性能 (前3个请求)
2. 稳定状态性能 (剩余请求)
3. 延迟分布统计

### 3.2 场景2：并发请求测试 (Concurrent)
**目的**: 测试系统在多用户场景下的吞吐量和资源利用

**配置**:
- 并发数: 8 (模拟8个同时用户)
- 请求数: 72
- 最大生成长度: 50 tokens
- 测试prompts: 5个中文文本

**测试内容**:
1. 系统吞吐量 (总tokens/总时间)
2. 单请求TPS分布
3. 并发稳定性
4. 失败请求统计

### 3.3 场景3：混合负载测试 (可选)
**目的**: 模拟真实生产环境的复杂负载

**配置**:
- 短请求: max_tokens=20, 占比40%
- 中等请求: max_tokens=50, 占比40%
- 长请求: max_tokens=100, 占比20%
- 并发数: 8
- 总请求数: 100

## 4. 测试数据

### 4.1 测试 Prompts
```
1. 人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

2. 机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。

3. 深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。

4. 自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。

5. 计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。
```

### 4.2 采样参数
```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

## 5. 测试工具

### 5.1 统一基准测试工具
**位置**: `tools/unified_benchmark.py`

**功能**:
- 支持 cLLM 和 Ollama 对比测试
- 顺序和并发测试模式
- 自动统计计算
- JSON结果输出

### 5.2 使用方法
```bash
# 顺序测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 72 \
  --concurrency 1 \
  --max-tokens 50 \
  --test-type api-sequential

# 并发测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --test-type api-concurrent

# 完整测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --test-type all \
  --output-file test_reports/cllm_llama_cpp_gpu_test.json
```

## 6. 性能指标定义

### 6.1 核心指标
| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **吞吐量 (Throughput)** | 每秒生成的总token数 | 总生成tokens / 总测试时间 |
| **单请求TPS** | 单个请求的生成速度 | generated_tokens / response_time |
| **平均响应时间** | 所有请求的平均耗时 | sum(response_time) / 成功请求数 |
| **P95响应时间** | 95%请求的响应时间 | 排序后取95分位数 |
| **P99响应时间** | 99%请求的响应时间 | 排序后取99分位数 |
| **成功率** | 成功完成请求的比例 | 成功请求数 / 总请求数 |

### 6.2 计算公式
```python
# 吞吐量 (Throughput)
throughput = total_generated_tokens / total_test_time

# 单请求TPS
tps_per_request = generated_tokens / response_time

# 平均吞吐量
avg_throughput = sum(tps_list) / len(tps_list)

# 总吞吐量 (另一种计算方式)
actual_throughput = sum(generated_tokens) / total_test_time
```

## 7. 测试流程

### 7.1 准备阶段
1. ✅ 确保 cLLM 服务器已编译
2. ✅ 确认 GGUF 模型文件存在
3. ✅ 清理系统缓存 (可选)
4. ✅ 记录系统状态

### 7.2 执行阶段
```bash
# Step 1: 启动 cLLM 服务器 (使用 llama_cpp + GPU 配置)
./build/bin/cllm_server --config config/config_gpu.yaml

# Step 2: 等待服务器就绪 (检查 /health 端点)
curl http://localhost:8080/health

# Step 3: 执行顺序测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --requests 72 \
  --concurrency 1 \
  --max-tokens 50 \
  --test-type api-sequential

# Step 4: 执行并发测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --test-type api-concurrent
```

### 7.3 收尾阶段
1. 停止 cLLM 服务器
2. 收集测试日志
3. 保存测试结果
4. 清理临时文件

## 8. 结果分析

### 8.1 成功标准
- ✅ 成功率 ≥ 99%
- ✅ 平均延迟 < 10s
- ✅ 无内存溢出
- ✅ 无崩溃

### 8.2 性能评估标准
| 等级 | 单请求TPS | 并发吞吐量 | 说明 |
|------|----------|-----------|------|
| ⭐优秀 | ≥ 30 | ≥ 100 | 超越预期 |
| ✅良好 | 20-30 | 50-100 | 符合预期 |
| ⚠️一般 | 10-20 | 20-50 | 需要优化 |
| ❌较差 | < 10 | < 20 | 需重大优化 |

### 8.3 关注指标
1. **冷启动 vs 热状态**: 对比前3个请求与后续请求的性能差异
2. **并发扩展性**: 并发8x时性能提升比例 (理想值: 4-6x)
3. **延迟稳定性**: P95/P99 与平均值差异
4. **错误模式**: 失败请求的原因分析

## 9. 预期问题与解决方案

### 9.1 常见问题
| 问题 | 可能原因 | 解决方案 |
|------|---------|----------|
| GPU未使用 | n_gpu_layers=0 | 设置为99 |
| 内存不足 | 批处理过大 | 减小 batch_size |
| 延迟波动 | 系统负载高 | 错峰测试 |
| 连接超时 | 请求处理慢 | 增加 timeout |

### 9.2 监控命令
```bash
# 查看GPU使用率 (macOS)
sudo powermetrics --gpu

# 查看内存使用
top -l 1 | head -n 10

# 查看cLLM服务器日志
tail -f logs/cllm_server.log
```

## 10. 输出文档

### 10.1 测试报告结构
1. **测试概述**: 时间、环境、配置
2. **测试结果**: 各场景的详细数据
3. **性能分析**: 图表、统计、对比
4. **问题发现**: 性能瓶颈、错误分析
5. **优化建议**: 改进方向、参数调整

### 10.2 结果文件
- `test_reports/cllm_llama_cpp_gpu_test.json`: 原始数据
- `test_reports/cllm_performance_report.md`: 详细报告

---

**测试方案版本**: v1.0
**创建日期**: 2025-02-04
**适用版本**: cLLM main分支
