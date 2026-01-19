# cLLM vs Ollama 自动化测试指南

本文档介绍如何使用自动化测试脚本进行cLLM和Ollama的性能对比测试。

---

## 测试脚本说明

我们提供了两个自动化测试脚本：

1. **Bash版本**: `tools/run_cllm_ollama_comparison.sh`
   - 简单易用，适合快速测试
   - 依赖bash环境

2. **Python版本**: `tools/run_cllm_ollama_comparison.py`
   - 功能强大，支持更多参数
   - 自动生成详细的Markdown报告
   - 支持保存JSON测试结果

---

## 前置条件

### 1. 安装依赖

```bash
# 安装Python依赖
pip3 install requests argparse
```

### 2. 启动cLLM服务器

```bash
# 方式1: 使用GPU配置
./build/bin/cllm_server --config config/config_gpu.yaml

# 方式2: 使用CPU配置
./build/bin/cllm_server --config config/config_cpu.yaml
```

### 3. 启动Ollama服务器

```bash
# 启动Ollama服务
ollama serve

# 确保已下载模型
ollama pull qwen3:0.6b
```

---

## 快速开始

### 使用Bash脚本（简单快速）

```bash
# 默认测试: 160请求，5并发，50 tokens
bash tools/run_cllm_ollama_comparison.sh
```

### 使用Python脚本（功能强大）

```bash
# 默认测试: 160请求，5并发，50 tokens
python3 tools/run_cllm_ollama_comparison.py

# 保存测试结果为JSON
python3 tools/run_cllm_ollama_comparison.py --save-results

# 自定义测试参数
python3 tools/run_cllm_ollama_comparison.py \
    --requests 100 \
    --concurrency 10 \
    --max-tokens 100

# 测试特定模型
python3 tools/run_cllm_ollama_comparison.py \
    --model qwen3:0.6b \
    --requests 160 \
    --concurrency 5
```

---

## 命令行参数

### Python脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cllm-url` | cLLM服务器URL | `http://localhost:18085` |
| `--ollama-url` | Ollama服务器URL | `http://localhost:11434` |
| `--model` | Ollama模型名称 | `qwen3:0.6b` |
| `--requests` | 测试请求数 | `160` |
| `--concurrency` | 并发数 | `5` |
| `--max-tokens` | 最大生成token数 | `50` |
| `--save-results` | 保存测试结果为JSON文件 | 不保存 |
| `--skip-check` | 跳过服务器状态检查 | 不跳过 |

---

## 测试方案

### 默认测试方案

- **请求数**: 160
- **并发数**: 5
- **最大token数**: 50
- **测试类型**: 顺序测试 + 并发测试
- **Prompt**: "Hello, how are you?"

### 推荐的测试场景

#### 场景1: 大规模性能测试

```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 160 \
    --concurrency 5 \
    --max-tokens 50 \
    --save-results
```

**适用**: 全面评估系统在高负载下的性能表现

#### 场景2: 高并发测试

```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 100 \
    --concurrency 20 \
    --max-tokens 50
```

**适用**: 评估系统在高并发场景下的性能

#### 场景3: 长文本生成测试

```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 50 \
    --concurrency 5 \
    --max-tokens 200
```

**适用**: 评估系统在生成长文本时的性能

#### 场景4: 快速验证测试

```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 10 \
    --concurrency 2 \
    --max-tokens 50
```

**适用**: 快速验证系统功能和基本性能

---

## 测试结果

### 控制台输出

测试过程中会实时显示每个请求的进度和结果：

```
2026-01-19 16:57:25,123 - cllm-optimized-benchmark - INFO -   Request 1/160: ✓ 1.37s - Generated: 50 tokens
2026-01-19 16:57:26,394 - cllm-optimized-benchmark - INFO -   Request 2/160: ✓ 1.27s - Generated: 50 tokens
...
```

### JSON结果文件

使用`--save-results`参数会生成JSON格式的测试结果：

```
results/
├── cllm_test_results_20260119_165725.json
└── ollama_test_results_20260119_165725.json
```

JSON文件包含详细的测试统计数据：

```json
{
  "sequential": {
    "total_requests": 160,
    "successful_requests": 160,
    "failed_requests": 0,
    "success_rate": 100.0,
    "avg_response_time": 1.26,
    "min_response_time": 1.13,
    "max_response_time": 1.46,
    "total_time": 201.20,
    "avg_throughput": 39.76,
    "avg_tokens_per_second": 39.82,
    "total_tokens": 14848,
    "avg_generated_tokens": 50.00
  },
  "concurrent": {
    ...
  }
}
```

### Markdown测试报告

Python脚本会自动生成详细的Markdown测试报告：

```
docs/testing/
└── cllm_vs_ollama_comparison_20260119_165725.md
```

报告包含：
- 测试结果摘要
- 详细的性能对比表格
- 性能分析和建议
- 相关文件链接

---

## 测试报告示例

### 测试结果摘要

| 指标 | cLLM | Ollama | 优势方 |
|------|-------|---------|--------|
| 总分 | 8.1/10 | 8.5/10 | Ollama |
| 成功率 | 95% | 100% | Ollama |
| 总测试时间 | 212.72s | 58.09s | Ollama |
| 平均响应时间 | 6.57s | 1.80s | Ollama |
| 吞吐量 | 35.73 t/s | 102.53 t/s | Ollama |

### 关键发现

1. **Ollama在并发测试中表现更优**
   - 吞吐量比cLLm高186.9%
   - 响应时间比cLLm短72.6%
   - 零失败率

2. **cLLM在顺序测试中表现更优**
   - 响应时间比Ollama短42.1%
   - 吞吐量比Ollama高48.0%
   - token生成准确率100%

3. **cLLM在高负载下稳定性下降**
   - 出现5%的失败率
   - 响应时间增加41.6%

---

## 常见问题

### Q1: 服务器未运行

**错误信息**: "cLLM服务器未运行" 或 "Ollama服务器未运行"

**解决方案**:

```bash
# 启动cLLM服务器
./build/bin/cllm_server --config config/config_gpu.yaml

# 启动Ollama服务器
ollama serve
```

### Q2: 端口被占用

**错误信息**: "端口已被占用"

**解决方案**:

```bash
# 查找占用端口的进程
lsof -i :18085  # cLLM默认端口
lsof -i :11434  # Ollama默认端口

# 杀死进程
kill -9 <PID>
```

### Q3: 测试失败率高

**可能原因**:
- 服务器资源不足
- 并发数设置过高
- 模型加载失败

**解决方案**:

```bash
# 降低并发数
python3 tools/run_cllm_ollama_comparison.py --concurrency 3

# 减少请求数
python3 tools/run_cllm_ollama_comparison.py --requests 50

# 检查服务器日志
```

### Q4: 测试时间过长

**可能原因**:
- 请求数过多
- 并发数过低
- 服务器性能不足

**解决方案**:

```bash
# 减少请求数
python3 tools/run_cllm_ollama_comparison.py --requests 50

# 增加并发数
python3 tools/run_cllm_ollama_comparison.py --concurrency 10
```

### Q5: 找不到Python模块

**错误信息**: "ModuleNotFoundError: No module named 'requests'"

**解决方案**:

```bash
# 安装依赖
pip3 install requests argparse

# 或使用系统Python
python3 -m pip install requests argparse
```

---

## 高级用法

### 批量测试

创建一个脚本来运行多个测试场景：

```bash
#!/bin/bash

# 测试不同并发数
for concurrency in 5 10 15 20; do
    echo "测试并发数: $concurrency"
    python3 tools/run_cllm_ollama_comparison.py \
        --requests 100 \
        --concurrency $concurrency \
        --max-tokens 50 \
        --save-results
    echo ""
done
```

### 定时测试

使用cron定时运行测试：

```bash
# 每天凌晨2点运行测试
0 2 * * * cd /path/to/cLLM && python3 tools/run_cllm_ollama_comparison.py --save-results >> logs/daily_test.log 2>&1
```

### 结果分析

使用Python脚本分析历史测试结果：

```python
import json
import glob

# 加载所有测试结果
results_files = glob.glob('results/*.json')

for file in results_files:
    with open(file, 'r') as f:
        data = json.load(f)
        print(f"{file}: {data['concurrent']['avg_throughput']} t/s")
```

---

## 相关文件

### 测试脚本

- [tools/run_cllm_ollama_comparison.sh](tools/run_cllm_ollama_comparison.sh) - Bash自动化测试脚本
- [tools/run_cllm_ollama_comparison.py](tools/run_cllm_ollama_comparison.py) - Python自动化测试脚本
- [tools/cllm_optimized_benchmark.py](tools/cllm_optimized_benchmark.py) - cLLM测试脚本
- [tools/ollama_benchmark.py](tools/ollama_benchmark.py) - Ollama测试脚本

### 配置文件

- [config/config_gpu.yaml](config/config_gpu.yaml) - GPU配置
- [config/config_cpu.yaml](config/config_cpu.yaml) - CPU配置

### 测试报告

- [docs/testing/cllm_vs_ollama_comparison_report_v3.md](docs/testing/cllm_vs_ollama_comparison_report_v3.md) - 大规模测试报告
- [docs/testing/AUTOMATED_TESTING_GUIDE.md](docs/testing/AUTOMATED_TESTING_GUIDE.md) - 本指南

---

## 贡献指南

如果您发现bug或有改进建议，欢迎：

1. 提交Issue描述问题
2. 提交Pull Request修复bug
3. 提供测试结果和性能数据

---

## 许可证

本项目遵循MIT许可证。

---

**最后更新**: 2026-01-19  
**版本**: 1.0.0
