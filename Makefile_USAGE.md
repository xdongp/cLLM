# cLLM Makefile 使用指南

本文档说明如何使用Makefile来启动、管理和测试cLLM服务器。

---

## 快速开始

### 1. 构建项目

```bash
# 构建项目（Release模式）
make build

# 或使用Debug模式
make build-debug
```

### 2. 启动服务器

```bash
# 默认启动（GPU模式，前台运行）
make start

# 或明确指定GPU模式
make start-gpu

# CPU模式
make start-cpu

# 后台运行
make start-bg
```

### 3. 查看服务器状态

```bash
# 查看服务器状态
make status

# 查看日志
make tail-logs
```

### 4. 停止服务器

```bash
# 停止服务器
make stop
```

---

## Makefile 目标说明

### 构建目标

| 目标 | 说明 |
|------|------|
| `make build` | 构建项目（Release模式） |
| `make build-debug` | 构建项目（Debug模式） |
| `make clean` | 清理构建文件 |
| `make rebuild` | 重新构建（clean + build） |

### 服务器管理目标

| 目标 | 说明 |
|------|------|
| `make start` | 启动服务器（GPU模式，前台） |
| `make start-gpu` | 启动服务器（GPU模式，前台） |
| `make start-cpu` | 启动服务器（CPU模式，前台） |
| `make start-bg` | 后台启动服务器（GPU模式） |
| `make stop` | 停止服务器 |
| `make status` | 查看服务器状态 |
| `make tail-logs` | 查看服务器日志 |

### 测试目标

| 目标 | 说明 |
|------|------|
| `make test` | 运行单元测试 |
| `make integration-test` | 运行集成测试 |
| `make test-all` | 运行所有测试 |
| `make test-tokenizer` | 运行tokenizer测试 |

### 其他目标

| 目标 | 说明 |
|------|------|
| `make install-deps` | 安装依赖 |
| `make setup-env` | 设置开发环境 |
| `make check-env` | 检查构建环境 |
| `make check-model` | 检查模型路径 |

---

## 服务器配置

### GPU模式（默认）

```bash
make start-gpu
```

**配置**:
- 配置文件: `config/config_gpu.yaml`
- 端口: 18085
- GPU层数: 99（全部使用GPU）
- 日志文件: `logs/cllm_server.log`
- 错误日志: `logs/cllm_server_error.log`

**适用场景**:
- 需要最佳性能
- 有可用的GPU资源
- 生产环境

### CPU模式

```bash
make start-cpu
```

**配置**:
- 配置文件: `config/config_cpu.yaml`
- 端口: 18085
- GPU层数: 0（纯CPU）
- 日志文件: `logs/cllm_server_cpu.log`
- 错误日志: `logs/cllm_server_cpu_error.log`

**适用场景**:
- 无GPU资源
- 测试CPU性能
- 开发和调试

### 后台模式

```bash
make start-bg
```

**特点**:
- 服务器在后台运行，不阻塞终端
- 自动保存PID到 `logs/cllm_server.pid`
- 日志输出到 `logs/cllm_server.log`
- 使用 `make stop` 停止服务器

**适用场景**:
- 持续运行的服务
- CI/CD环境
- 自动化测试

---

## 日志管理

### 查看日志

```bash
# 实时查看日志
make tail-logs

# 或直接使用tail
tail -f logs/cllm_server.log
```

### 日志文件

| 日志文件 | 说明 |
|----------|------|
| `logs/cllm_server.log` | GPU模式服务器日志 |
| `logs/cllm_server_error.log` | GPU模式错误日志 |
| `logs/cllm_server_cpu.log` | CPU模式服务器日志 |
| `logs/cllm_server_cpu_error.log` | CPU模式错误日志 |
| `logs/cllm_server.pid` | 后台运行的PID文件 |

### 清理日志

```bash
# 清理所有日志
rm -rf logs/

# 或保留日志文件，清空内容
> logs/cllm_server.log
> logs/cllm_server_error.log
```

---

## 常见使用场景

### 场景1: 开发和调试

```bash
# 1. 构建项目
make build-debug

# 2. 启动服务器（前台运行，方便查看日志）
make start-gpu

# 3. 在另一个终端测试
curl http://localhost:18085/health

# 4. 按Ctrl+C停止服务器
```

### 场景2: 持续运行服务

```bash
# 1. 构建项目
make build

# 2. 后台启动
make start-bg

# 3. 查看状态
make status

# 4. 查看日志
make tail-logs

# 5. 停止服务
make stop
```

### 场景3: 性能测试

```bash
# 1. 测试GPU性能
make start-gpu
# 在另一个终端运行测试工具
# 按Ctrl+C停止

# 2. 测试CPU性能
make start-cpu
# 在另一个终端运行测试工具
# 按Ctrl+C停止

# 3. 对比性能
# 查看测试报告
```

### 场景4: 自动化测试

```bash
# 1. 后台启动服务器
make start-bg

# 2. 等待服务器启动
sleep 5

# 3. 运行测试
make test

# 4. 运行性能测试
python3 tools/run_cllm_ollama_comparison.py

# 5. 停止服务器
make stop
```

---

## 环境变量

可以通过环境变量覆盖默认配置:

```bash
# 构建参数
make build BUILD_TYPE=Debug

# 测试参数
make test TEST_FILTER=test_name
```

---

## 故障排除

### 问题1: 端口被占用

**错误**: `Address already in use`

**解决**:
```bash
# 1. 查找占用端口的进程
lsof -i :18085

# 2. 停止进程
kill -9 <PID>

# 3. 或使用make stop
make stop
```

### 问题2: 服务器未启动

**错误**: `Connection refused`

**解决**:
```bash
# 1. 检查服务器状态
make status

# 2. 如果未运行，启动服务器
make start

# 3. 查看日志
make tail-logs
```

### 问题3: 构建失败

**解决**:
```bash
# 1. 清理构建文件
make clean

# 2. 重新构建
make build

# 3. 检查环境
make check-env
```

### 问题4: 日志文件过大

**解决**:
```bash
# 1. 清空日志
> logs/cllm_server.log
> logs/cllm_server_error.log

# 2. 或删除日志目录
rm -rf logs/

# 3. 重新启动服务器
make start
```

---

## 相关文档

- [自动化测试指南](docs/testing/AUTOMATED_TESTING_GUIDE.md)
- [性能测试报告](docs/testing/cllm_vs_ollama_comparison_report_v3.md)
- [配置文件说明](config/README.md)

---

## 贡献

如果您发现Makefile有问题或有改进建议，欢迎提交Issue或Pull Request。

---

**最后更新**: 2026-01-19  
**版本**: 1.0.0
