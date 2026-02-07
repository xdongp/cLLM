# cLLM + Kylin + HuggingFace + GPU 性能测试方案

## 1. 测试目标

### 1.1 主要目标
- 验证 cLLM 在 Kylin 自研推理引擎 + HuggingFace safetensors 格式 + GPU 加速下的推理性能
- 测试系统在**不同并发级别**下的吞吐量(TPS)和延迟表现
- 评估 GGML GPU 后端（Metal）的计算效率
- 对比 HF + GPU 与 llama_cpp + GPU 的性能差异
- 建立性能基准线，为后续优化提供数据支撑

### 1.2 预期性能指标
| 指标 | 目标值 | 说明 |
|------|--------|------|
| 单请求吞吐量 | ≥ 25 tokens/s | 单用户场景下的生成速度 |
| 并发吞吐量 | ≥ 60 tokens/s | 8并发下的总吞吐量 |
| 平均延迟 | < 4s | 50 tokens生成的总耗时 |
| 冷启动时间 | < 3s | 模型加载到首次响应 |
| 成功率 | ≥ 99% | 成功完成的请求比例 |

### 1.3 与 llama_cpp 对比目标
| 对比项 | Kylin + HF + GPU | llama_cpp + GPU | 预期差异 |
|--------|-----------------|-----------------|---------|
| 单请求 TPS | ≥ 25 | ≥ 20 | +25% |
| 并发 TPS | ≥ 60 | ≥ 50 | +20% |
| 内存占用 | < 2GB | < 2.5GB | -20% |

## 2. 测试环境

### 2.1 硬件配置
- **CPU**: Apple Silicon M系列 (具体型号待测)
- **GPU**: 集成GPU (Metal加速, ~10 TFLOPS)
- **内存**: 16GB+ (系统自动管理)
- **存储**: SSD (模型文件存储)

### 2.2 软件环境
- **操作系统**: macOS
- **cLLM版本**: 当前main分支
- **推理引擎**: Kylin 自研引擎 (src/kylin/)
- **模型格式**: HuggingFace safetensors (FP16)
- **模型**: Qwen3-0.6B
- **量化类型**: FP16 (原生 BF16 -> FP16)

### 2.3 架构配置
```
HTTP Server (cLLM)
    ↓
Scheduler (动态批处理)
    ↓
Kylin Backend (统一推理接口)
    ↓
HF Transformer (HuggingFace safetensors)
    ↓
GGML GPU Backend (Metal加速)
    ↓
SIMD优化内核 (ARM NEON)
```

### 2.4 Kylin 后端关键参数
```yaml
backend:
  type: "kylin"
  kylin:
    device_backend: "metal"    # 使用 Metal GPU 加速
    operator_backend: "ggml"    # GGML 计算后端
    n_threads: 0                # GPU 模式线程数（自动）

model:
  path: "/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B"
  quantization: "fp16"          # FP16 推理

resources:
  max_batch_size: 32            # GPU 模式可以使用更大批次
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
   - 首次请求延迟
   - 模型预热时间
2. 稳定状态性能 (剩余请求)
   - 平均TPS
   - 延迟分布
3. 内存使用监控
   - 峰值内存
   - 稳定后内存

### 3.2 场景2：并发请求测试 (Concurrent)
**目的**: 测试系统在多用户场景下的吞吐量和资源利用

**配置**:
- 并发数: 8 (模拟8个同时用户)
- 请求数: 72
- 最大生成长度: 50 tokens
- 测试prompts: 5个中文文本

**测试内容**:
1. 系统吞吐量
   - 总tokens/总时间
   - 平均TPS per request
2. 延迟分布
   - P50, P90, P95, P99
3. 并发稳定性
   - 无崩溃/无超时
4. 失败请求统计

### 3.3 场景3：混合负载测试 (Hybrid)
**目的**: 模拟真实生产环境的复杂负载

**配置**:
- 短请求: max_tokens=20, 占比40%
- 中等请求: max_tokens=50, 占比40%
- 长请求: max_tokens=100, 占比20%
- 并发数: 8
- 总请求数: 100

**测试内容**:
1. 动态批处理效果
2. KV Cache 复用率
3. GPU 内存稳定性

### 3.4 场景4：长时间压力测试 (Stress)
**目的**: 测试系统在持续负载下的稳定性

**配置**:
- 并发数: 4
- 请求数: 500
- 最大生成长度: 50 tokens
- 测试时间: 30分钟

**测试内容**:
1. 内存泄漏检测
2. GPU 内存稳定性
3. 性能衰减曲线
4. 系统稳定性

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
  "top_k": 50,
  "max_tokens": 50
}
```

## 5. 测试工具

### 5.1 统一基准测试工具
**位置**: `tools/unified_benchmark.py`

**功能**:
- 支持 Kylin 和 llama_cpp 对比测试
- 顺序和并发测试模式
- 自动统计计算
- JSON结果输出
- 性能图表生成

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

# 完整测试 (推荐)
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --test-type all \
  --output-file test_reports/cllm_kylin_hf_gpu_test.json

# 长时间压力测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 500 \
  --concurrency 4 \
  --max-tokens 50 \
  --test-type stress \
  --duration 1800 \
  --output-file test_reports/cllm_kylin_hf_gpu_stress.json
```

### 5.3 手动 curl 测试
```bash
# 健康检查
curl http://localhost:8080/health

# 单次生成测试
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "人工智能是计算机科学的一个分支",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# 流式生成测试
curl -X POST http://localhost:8080/generate_stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "prompt": "机器学习是人工智能的一个分支",
    "max_tokens": 50,
    "temperature": 0.7
  }'
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
| **冷启动时间** | 首个请求的完整延迟 | 模型加载 + 首次推理 |
| **成功率** | 成功完成请求的比例 | 成功请求数 / 总请求数 |

### 6.2 GPU 特有指标
| 指标 | 定义 | 监控方式 |
|------|------|----------|
| **GPU 利用率** | Metal GPU 计算占用率 | powermetrics |
| **GPU 内存** | Metal 分配的显存 | instruments |
| **计算节点数** | GGML 调度器执行的 GPU 节点数 | 日志统计 |

### 6.3 计算公式
```python
# 吞吐量 (Throughput)
throughput = total_generated_tokens / total_test_time

# 单请求TPS
tps_per_request = generated_tokens / response_time

# 平均吞吐量
avg_throughput = sum(tps_list) / len(tps_list)

# 总吞吐量 (另一种计算方式)
actual_throughput = sum(generated_tokens) / total_test_time

# 并发扩展性
concurrency_scalability = throughput_concurrent / throughput_sequential
```

## 7. 测试流程

### 7.1 准备阶段
```bash
# Step 1: 确保 cLLM 服务器已编译 (Kylin + GPU)
cd /Users/dannypan/PycharmProjects/cLLM
mkdir -p build && cd build
cmake .. -DKYLIN_BACKEND=ON -DGGML_METAL=ON
make -j$(nproc)

# Step 2: 确认 HF 模型文件存在
ls -la /Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B/

# Step 3: 创建配置文件
cp config/config_kylin_gpu.yaml config_kylin_hf_gpu.yaml

# Step 4: 清理系统缓存 (可选)
sudo purge

# Step 5: 记录系统状态
sysctl hw.memsize | head -n 1
system_profiler SPMemoryType
```

### 7.2 执行阶段
```bash
# Step 1: 启动 cLLM 服务器 (使用 Kylin + HF + GPU 配置)
./build/bin/cllm_server --config config_kylin_hf_gpu.yaml

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

# Step 5: 执行压力测试 (可选)
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --requests 500 \
  --concurrency 4 \
  --max-tokens 50 \
  --test-type stress \
  --duration 1800
```

### 7.3 监控命令
```bash
# 实时监控 GPU 使用率 (需要 sudo)
sudo powermetrics --gpu -n 5 -u 1000

# 查看内存使用
top -l 1 | head -n 10

# 查看进程详情
ps aux | grep cllm_server

# 监控 Metal GPU
sudo powermetrics --gpu -n 100

# 查看 cLLM 服务器日志
tail -f logs/cllm_server.log
```

### 7.4 收尾阶段
```bash
# Step 1: 停止 cLLM 服务器
pkill -f cllm_server

# Step 2: 收集测试日志
cp logs/cllm_server.log test_reports/

# Step 3: 保存测试结果
ls -la test_reports/

# Step 4: 清理临时文件
rm -f /tmp/cllm_*
```

## 8. 结果分析

### 8.1 成功标准
- ✅ 成功率 ≥ 99%
- ✅ 平均延迟 < 10s
- ✅ 无内存溢出 (内存稳定在 3GB 以内)
- ✅ 无崩溃

### 8.2 性能评估标准
| 等级 | 单请求TPS | 并发吞吐量 | 说明 |
|------|----------|-----------|------|
| ⭐优秀 | ≥ 35 | ≥ 100 | 超越预期，显著优于 llama_cpp |
| ✅良好 | 25-35 | 60-100 | 符合预期，达到目标 |
| ⚠️一般 | 15-25 | 30-60 | 需要优化，有提升空间 |
| ❌较差 | < 15 | < 30 | 需重大优化，可能存在瓶颈 |

### 8.3 对比分析维度
1. **冷启动 vs 热状态**: 对比前3个请求与后续请求的性能差异
2. **并发扩展性**: 并发8x时性能提升比例 (理想值: 4-6x)
3. **延迟稳定性**: P95/P99 与平均值差异
4. **GPU 利用率**: 对比 CPU vs GPU 模式的性能提升
5. **错误模式**: 失败请求的原因分析

### 8.4 关注指标
```
关键指标优先级:
1. 并发吞吐量 (最高优先级 - 衡量整体性能)
2. 单请求 TPS (衡量单用户体验)
3. P99 延迟 (衡量尾部延迟)
4. 冷启动时间 (衡量首次体验)
5. 内存使用 (衡量资源效率)
```

## 9. 预期问题与解决方案

### 9.1 常见问题
| 问题 | 可能原因 | 解决方案 |
|------|---------|----------|
| GPU 未使用 | device_backend=cpu | 设置 device_backend=metal |
| Metal 初始化失败 | ggml-metal 未编译 | 重新编译启用 -DGGML_METAL=ON |
| 内存不足 | 批处理过大 | 减小 max_batch_size |
| 延迟波动 | 系统负载高 | 错峰测试，关闭其他应用 |
| 连接超时 | 请求处理慢 | 增加 timeout_ms |
| 模型加载失败 | safetensors 路径错误 | 检查 model.path 配置 |

### 9.2 调试命令
```bash
# 检查 Metal 是否可用
./build/bin/cllm_server --help 2>&1 | grep -i metal

# 启用详细日志
./build/bin/cllm_server --config config_kylin_hf_gpu.yaml --log-level debug

# 测试模型加载
time ./build/bin/cllm_server --config config_kylin_hf_gpu.yaml &
sleep 5
curl http://localhost:8080/health
pkill -f cllm_server
```

## 10. 输出文档

### 10.1 测试报告结构
1. **测试概述**: 时间、环境、配置
2. **测试结果**: 各场景的详细数据
3. **性能分析**: 图表、统计、对比
4. **问题发现**: 性能瓶颈、错误分析
5. **优化建议**: 改进方向、参数调整
6. **与 llama_cpp 对比**: 性能差异分析

### 10.2 结果文件
- `test_reports/cllm_kylin_hf_gpu_test.json`: 原始数据
- `test_reports/cllm_kylin_hf_gpu_stress.json`: 压力测试数据
- `test_reports/cllm_performance_report.md`: 详细报告

### 10.3 报告模板
```markdown
# Kylin + HF + GPU 性能测试报告

## 测试概述
- **测试日期**: YYYY-MM-DD HH:MM
- **测试环境**: macOS + Apple Silicon
- **推理引擎**: Kylin + HF + Metal
- **模型**: Qwen3-0.6B (FP16)
- **测试配置**: [链接到配置文件]

## 测试结果

### 顺序测试 (72 请求, 1 并发)
- 平均 TPS: XX.XX
- 平均延迟: XX.XXs
- P95 延迟: XX.XXs
- P99 延迟: XX.XXs
- 成功率: XX%

### 并发测试 (72 请求, 8 并发)
- 总吞吐量: XX.XX tokens/s
- 平均 TPS: XX.XX
- 平均延迟: XX.XXs
- P95 延迟: XX.XXs
- P99 延迟: XX.XXs
- 成功率: XX%

## 性能对比
| 指标 | Kylin + HF + GPU | llama_cpp + GPU | 差异 |
|------|-----------------|-----------------|------|
| 单请求 TPS | XX | XX | +XX% |
| 并发 TPS | XX | XX | +XX% |
| 冷启动 | XXs | XXs | +XX% |

## 问题与优化
- [问题描述]
- [根本原因]
- [解决方案]
- [预期效果]

## 结论
[总结测试结果和优化建议]
```

## 11. 与 llama_cpp 对比测试

### 11.1 对比测试流程
```bash
# Step 1: 测试 Kylin + HF + GPU
./build/bin/cllm_server --config config_kylin_hf_gpu.yaml &
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --output-file test_reports/kylin_hf_gpu.json
pkill -f cllm_server

# Step 2: 等待 30 秒冷却
sleep 30

# Step 3: 测试 llama_cpp + GPU
./build/bin/cllm_server --config config_llama_cpp_gpu.yaml &
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --requests 72 \
  --concurrency 8 \
  --max-tokens 50 \
  --output-file test_reports/llama_cpp_gpu.json
pkill -f cllm_server

# Step 4: 生成对比报告
python3 tools/compare_results.py \
  test_reports/kylin_hf_gpu.json \
  test_reports/llama_cpp_gpu.json \
  --output test_reports/compare_report.md
```

### 11.2 对比指标
```
核心对比指标:
1. 吞吐量提升: (Kylin_TPS - llama_TPS) / llama_TPS * 100%
2. 延迟降低: (llama_latency - Kylin_latency) / llama_latency * 100%
3. 内存效率: llama_memory / Kylin_memory
4. 启动速度: llama_startup / Kylin_startup
```

---

**测试方案版本**: v1.0
**创建日期**: 2026-02-05
**适用版本**: cLLM main分支
**基于方案**: cllm_llama_cpp_gpu_test_plan.md v1.0
