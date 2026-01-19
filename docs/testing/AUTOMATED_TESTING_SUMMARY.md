# cLLM vs Ollama 自动化测试总结

## 概述

本文档总结了cLLM和Ollama性能对比测试的自动化方案。

---

## 已完成的工作

### 1. 测试脚本

#### Bash版本
- **文件**: [tools/run_cllm_ollama_comparison.sh](tools/run_cllm_ollama_comparison.sh)
- **特点**: 简单易用，适合快速测试
- **功能**:
  - 自动检查服务器状态
  - 运行cLLM和Ollama测试
  - 支持自定义参数

#### Python版本
- **文件**: [tools/run_cllm_ollama_comparison.py](tools/run_cllm_ollama_comparison.py)
- **特点**: 功能强大，自动生成报告
- **功能**:
  - 自动检查服务器状态
  - 运行cLLM和Ollama测试
  - 生成详细的Markdown报告
  - 保存JSON格式的测试结果
  - 支持多种测试场景

### 2. Makefile集成

- **文件**: [Makefile](../Makefile)
- **新增目标**:
  - `make start` - 启动服务器（GPU模式）
  - `make start-gpu` - 启动服务器（GPU模式）
  - `make start-cpu` - 启动服务器（CPU模式）
  - `make start-bg` - 后台启动服务器
  - `make stop` - 停止服务器
  - `make status` - 查看服务器状态
  - `make tail-logs` - 查看日志
- **日志管理**:
  - 自动创建logs目录
  - 日志输出到 `logs/cllm_server.log`
  - 错误日志输出到 `logs/cllm_server_error.log`

### 3. 测试报告

- **V1报告**: [cllm_vs_ollama_comparison_report.md](cllm_vs_ollama_comparison_report.md)
  - 小规模测试（10请求）
  - 基础性能对比

- **V2报告**: [cllm_vs_ollama_comparison_report_v2.md](cllm_vs_ollama_comparison_report_v2.md)
  - 中等规模测试（10请求）
  - 详细的性能对比
  - 相同测试条件

- **V3报告**: [cllm_vs_ollama_comparison_report_v3.md](cllm_vs_ollama_comparison_report_v3.md)
  - 大规模测试（160请求）
  - 全面的性能分析
  - 高负载下的稳定性对比

### 4. 文档

- **自动化测试指南**: [AUTOMATED_TESTING_GUIDE.md](AUTOMATED_TESTING_GUIDE.md)
  - 详细的使用说明
  - 常见问题解答
  - 高级用法

- **Makefile使用指南**: [Makefile_USAGE.md](../Makefile_USAGE.md)
  - Makefile目标说明
  - 使用场景
  - 故障排除

---

## 测试方案

### 默认测试配置

- **请求数**: 160
- **并发数**: 5
- **最大token数**: 50
- **模型**: Qwen3 0.6B Q4_K_M
- **Prompt**: "Hello, how are you?"

### 测试场景

#### 场景1: 快速验证测试
```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 10 \
    --concurrency 2 \
    --max-tokens 50
```

#### 场景2: 标准性能测试
```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 160 \
    --concurrency 5 \
    --max-tokens 50 \
    --save-results
```

#### 场景3: 高并发测试
```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 100 \
    --concurrency 20 \
    --max-tokens 50
```

#### 场景4: 长文本生成测试
```bash
python3 tools/run_cllm_ollama_comparison.py \
    --requests 50 \
    --concurrency 5 \
    --max-tokens 200
```

---

## 关键发现

### 在大规模测试（160请求）中的表现

| 指标 | cLLM | Ollama | 优势方 |
|------|-------|---------|--------|
| 总分 | 8.1/10 | 8.5/10 | Ollama |
| 成功率 | 95% | 100% | Ollama |
| 总测试时间 | 212.72s | 58.09s | Ollama |
| 平均响应时间 | 6.57s | 1.80s | Ollama |
| 吞吐量 | 35.73 t/s | 102.53 t/s | Ollama |

### cLLM的优势

1. **token生成准确**: 严格按照要求生成50个token
2. **顺序测试稳定**: 响应时间变异系数仅为4.8%
3. **适合低负载**: 在低负载下性能稳定可靠

### cLLM的劣势

1. **高负载下稳定性下降**: 在160请求时出现5%的失败率
2. **响应时间增加**: 在高负载下响应时间增加41.6%
3. **吞吐量下降**: 在高负载下吞吐量下降28.7%

### Ollama的优势

1. **高并发性能优异**: 吞吐量比cLLm高186.9%
2. **零失败率**: 在所有测试中都保持100%成功率
3. **总测试时间短**: 比cLLm短72.7%
4. **适合高负载**: 能够快速处理大量并发请求

### Ollama的劣势

1. **token生成不准确**: 平均只生成37.23个token
2. **第一个请求预热慢**: 14.04s vs cLLm的1.37s

---

## 使用方法

### 快速开始

```bash
# 1. 启动cLLM服务器
make start-bg

# 2. 启动Ollama服务器
ollama serve

# 3. 运行测试
python3 tools/run_cllm_ollama_comparison.py

# 4. 查看报告
open docs/testing/cllm_vs_ollama_comparison_*.md

# 5. 停止服务器
make stop
```

### 后台自动化测试

```bash
#!/bin/bash

# 启动服务器
make start-bg

# 等待服务器启动
sleep 5

# 运行测试
python3 tools/run_cllm_ollama_comparison.py \
    --requests 160 \
    --concurrency 5 \
    --max-tokens 50 \
    --save-results

# 停止服务器
make stop

# 查看结果
ls -la results/
ls -la docs/testing/cllm_vs_ollama_comparison_*.md
```

---

## 文件结构

```
docs/testing/
├── AUTOMATED_TESTING_GUIDE.md          # 自动化测试指南
├── AUTOMATED_TESTING_SUMMARY.md        # 本总结文档
├── cllm_vs_ollama_comparison_report.md           # V1测试报告
├── cllm_vs_ollama_comparison_report_v2.md        # V2测试报告
├── cllm_vs_ollama_comparison_report_v3.md        # V3测试报告
└── *.md                                  # 其他测试报告

tools/
├── run_cllm_ollama_comparison.sh        # Bash测试脚本
├── run_cllm_ollama_comparison.py        # Python测试脚本
├── cllm_optimized_benchmark.py          # cLLM测试脚本
├── ollama_benchmark.py                  # Ollama测试脚本
└── examples/
    └── test_example.sh                  # 测试示例

results/                                  # 测试结果目录
├── cllm_test_results_*.json
└── ollama_test_results_*.json

logs/                                     # 日志目录
├── cllm_server.log
├── cllm_server_error.log
├── cllm_server_cpu.log
├── cllm_server_cpu_error.log
└── cllm_server.pid
```

---

## 未来改进方向

### 1. cLLM优化

- [ ] 优化并发处理，减少高负载下的失败率
- [ ] 优化内存管理，避免内存不足
- [ ] 优化GPU利用率，提升性能
- [ ] 添加自动重启机制
- [ ] 实现负载均衡

### 2. 测试工具优化

- [ ] 支持更多测试场景
- [ ] 添加实时性能监控
- [ ] 支持分布式测试
- [ ] 添加性能趋势分析
- [ ] 支持更多模型

### 3. 文档完善

- [ ] 添加更多使用示例
- [ ] 完善故障排除指南
- [ ] 添加性能调优建议
- [ ] 提供API文档

---

## 相关链接

- [cLLM GitHub](https://github.com/your-repo/cllm)
- [Ollama](https://ollama.ai/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Qwen3模型](https://github.com/QwenLM/Qwen3)

---

## 许可证

本项目遵循MIT许可证。

---

**最后更新**: 2026-01-19  
**版本**: 1.0.0  
**作者**: TraeAI Assistant
