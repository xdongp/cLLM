# cLLM - C++ Large Language Model Inference Engine

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.15+-green.svg)](https://cmake.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

**高性能 C++ 大语言模型推理引擎**

[快速开始](#-快速开始) • [文档](#-文档) • [架构](#-架构) • [贡献](#-贡献)

</div>

---

## 📖 项目简介

cLLM 是一个基于 C++17 开发的高性能大语言模型推理引擎，从 Python 版本重构而来，专注于：

- 🚀 **高性能**: 推理速度 20+ tokens/s，比 Python 版本提升 3-5 倍
- 💾 **低内存**: 优化的内存管理和 KV 缓存策略
- 🔄 **高并发**: 基于 Drogon 异步框架，支持大规模并发请求
- 🎯 **生产就绪**: 完整的 HTTP API、动态批处理、流式输出

---

## ✨ 核心特性

### 推理能力
- ✅ LibTorch 后端（PyTorch C++ API）
- ✅ HuggingFace Tokenizers 支持
- ✅ 多种采样策略（Temperature, Top-K, Top-P, Repetition Penalty）
- ✅ KV Cache 优化
- ✅ 动态批处理（Dynamic Batching）

### 服务能力
- ✅ RESTful HTTP API
- ✅ 流式生成（Server-Sent Events）
- ✅ 健康检查端点
- ✅ 请求队列管理
- ✅ 异步处理框架

### 开发体验
- ✅ 模块化设计，易于扩展
- ✅ 完善的单元测试
- ✅ 详细的文档系统
- ✅ CodeBuddy AI 辅助开发

---

## 🚀 快速开始

### 前置条件

- C++17 或更高版本编译器（GCC 7+, Clang 5+）
- CMake 3.15+
- LibTorch 1.9+
- 其他依赖：Drogon, Eigen3, nlohmann-json, spdlog

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/YOUR_USERNAME/cLLM.git
cd cLLM

# 2. 安装依赖 (macOS)
brew install cmake libtorch eigen nlohmann-json spdlog drogon

# 或 (Ubuntu)
sudo apt-get install cmake libtorch-dev libeigen3-dev \
    nlohmann-json3-dev libspdlog-dev libdrogon-dev

# 3. 编译项目
mkdir build && cd build
cmake ..
make -j$(nproc)

# 4. 运行测试
ctest --output-on-failure
```

### 快速运行

```bash
# 启动 HTTP 服务器
./bin/cllm_server --config ../config/default.yaml

# 测试健康检查
curl http://localhost:8080/health

# 测试文本生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_tokens": 50}'
```

**详细步骤**: 查看 [快速开始指南](docs/guides/快速开始.md)

---

## 📚 文档

### 入门指南
- [快速开始](docs/guides/快速开始.md) - 5分钟上手
- [开发环境搭建](docs/guides/开发环境搭建.md) - 完整环境配置
- [配置快速参考](docs/guides/配置快速参考.md) - 配置文件说明
- [服务器使用指南](docs/guides/服务器使用指南.md) - HTTP API 使用

### 架构文档
- [cLLM详细设计](docs/architecture/cLLM详细设计.md) - 完整系统架构
- [组件交互设计](docs/architecture/组件交互设计.md) - 模块关系
- [工程编译设计](docs/architecture/工程编译设计.md) - 编译系统

### 模块设计
- [Tokenizer模块设计](docs/modules/Tokenizer模块设计.md) - 分词器
- [调度器模块设计](docs/modules/调度器模块设计.md) - 请求调度
- [HTTP服务器模块设计](docs/modules/HTTP服务器模块设计.md) - Web 服务
- [更多模块...](docs/modules/)

### 开发规范
- [C++编程规范](docs/specifications/C++编程规范_团队版.md) - 编码标准
- [CodeBuddy使用指南](docs/guides/CodeBuddy使用指南.md) - AI 辅助开发
- [文档命名规范](docs/specifications/文档命名规范.md) - 文档标准

### 完整导航
📋 [文档导航](docs/00_文档导航.md) - 所有文档索引

---

## 🏗️ 架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server Layer                     │
│ (RESTful API Endpoints, Request Handling, Validation)    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Request Scheduler                       │
│  (Request Management, Dynamic Batching, Execution)       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Model Executor                           │
│  (Model Loading, Inference, Quantization, Optimization)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Core Components Layer                       │
│ Tokenizer | Sampler | KV Cache | Memory Management       │
└─────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 职责 | 文件 |
|------|------|------|
| **HTTP Server** | RESTful API 服务 | `src/http/` |
| **Scheduler** | 请求调度和批处理 | `src/scheduler/` |
| **Model Executor** | 模型加载和推理 | `src/model/` |
| **Tokenizer** | 文本编码/解码 | `src/tokenizer/` |
| **Sampler** | Token 采样策略 | `src/sampler/` |
| **KV Cache** | 键值缓存管理 | `src/kv_cache/` |

**详细架构**: 查看 [cLLM详细设计](docs/architecture/cLLM详细设计.md)

---

## 🔌 API 参考

### HTTP 端点

#### 1. 健康检查
```bash
curl http://localhost:18080/health
```

#### 2. 文本生成
```bash
# 基本生成测试
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "hello",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# 中文生成测试
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# 短文本生成（快速测试）
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "hello",
    "max_tokens": 10,
    "temperature": 0.7
  }'

# 带响应时间测量的生成测试
time curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "hello",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

#### 3. 流式生成
```bash
POST /v1/generate/stream
Content-Type: application/json

{
  "prompt": "讲一个故事",
  "max_tokens": 200,
  "stream": true
}
```

#### 4. Token 编码
```bash
# 文本编码测试
curl -X POST http://localhost:18080/encode \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world"
  }'

# 中文编码测试
curl -X POST http://localhost:18080/encode \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界"
  }'
```

**完整 API 文档**: 查看 [服务器使用指南](docs/guides/服务器使用指南.md)

---

## 🛠️ 开发指南

### 项目结构

```
cLLM/
├── src/                    # 源代码
│   ├── http/              # HTTP 服务器
│   ├── scheduler/         # 请求调度器
│   ├── model/             # 模型执行器
│   ├── tokenizer/         # 分词器
│   ├── sampler/           # 采样器
│   └── kv_cache/          # KV 缓存
├── include/                # 头文件
│   └── cllm/              # 公共接口
├── tests/                  # 测试代码
├── examples/               # 示例代码
├── docs/                   # 文档
├── config/                 # 配置文件
└── scripts/                # 工具脚本
```

### 添加新功能

```bash
# 1. 创建开发分支
git checkout -b feature/your-feature

# 2. 编写代码（遵守 C++17 标准）
# include/cllm/your_module.h
# src/your_module.cpp

# 3. 添加测试
# tests/test_your_module.cpp

# 4. 编译和测试
mkdir build && cd build
cmake ..
make
ctest

# 5. 提交代码
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

### 编码规范

遵循 [C++编程规范](docs/specifications/C++编程规范_团队版.md):

- ✅ C++17 标准
- ✅ 类名使用 PascalCase
- ✅ 函数/变量使用 snake_case
- ✅ 成员变量使用 `_` 后缀
- ✅ 使用智能指针管理内存
- ✅ 完善的错误处理
- ✅ 详细的注释和文档

---

## 🧪 测试

### 运行测试

```bash
cd build

# 运行所有测试
ctest --output-on-failure

# 运行特定测试
./bin/test_tokenizer
./bin/test_scheduler
./bin/test_model_executor

# 运行集成测试
./bin/integrated_test
```

### 测试覆盖率

```bash
# 生成覆盖率报告
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
make coverage
```

**测试文档**: 查看 [测试指南](docs/tests/)

---

## 📊 性能

### 基准测试

| 指标 | Python 版本 | C++ 版本 | 提升 |
|------|------------|----------|------|
| **推理速度** | 5-8 tokens/s | 20+ tokens/s | **3-5x** |
| **内存占用** | ~2GB | ~800MB | **2.5x** |
| **并发能力** | 10 req/s | 100+ req/s | **10x** |
| **启动时间** | 3-5s | <1s | **5x** |

### 优化技术

- ✅ KV Cache 复用
- ✅ 动态批处理（Batch Size: 1-32）
- ✅ 异步 I/O（Drogon + Asio）
- ✅ 零拷贝内存管理
- ✅ LibTorch JIT 优化

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 贡献流程

1. **Fork** 本仓库
2. **创建分支** (`git checkout -b feature/amazing-feature`)
3. **提交更改** (`git commit -m 'feat: add amazing feature'`)
4. **推送分支** (`git push origin feature/amazing-feature`)
5. **创建 Pull Request**

### 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: 新功能
fix: 修复 Bug
docs: 文档更新
test: 添加测试
refactor: 重构代码
perf: 性能优化
chore: 构建/工具链更新
```

**详细指南**: 查看 [贡献指南](docs/guides/贡献指南.md)

---

## 📋 技术栈

### 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| **LibTorch** | 1.9+ | 深度学习推理 |
| **Drogon** | 1.7+ | HTTP 服务器 |
| **Eigen3** | 3.3+ | 线性代数 |
| **nlohmann-json** | 3.2+ | JSON 处理 |
| **spdlog** | 1.8+ | 日志系统 |
| **Asio** | 1.18+ | 异步 I/O |
| **yaml-cpp** | 0.6+ | YAML 配置 |

### 开发工具

- **构建**: CMake 3.15+
- **测试**: Google Test + Google Mock
- **CI/CD**: GitHub Actions
- **文档**: Markdown
- **代码质量**: clang-format, clang-tidy

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 👥 贡献者

感谢所有贡献者！

---

## 🔗 相关链接

- **文档**: [docs/](docs/)
- **问题反馈**: [Issues](https://github.com/YOUR_USERNAME/cLLM/issues)
- **讨论**: [Discussions](https://github.com/YOUR_USERNAME/cLLM/discussions)

---

## 📞 联系方式

- **Email**: xdongp@gmail.com
- **GitHub**: [@xdongp](https://github.com/xdongp)

---

## 🎉 致谢

感谢以下开源项目：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Drogon](https://github.com/drogonframework/drogon) - C++ Web 框架
- [HuggingFace](https://huggingface.co/) - Tokenizers 库
- [nlohmann-json](https://github.com/nlohmann/json) - JSON 库

---

<div align="center">

**⭐ 如果觉得有帮助，请给个 Star！**

Made with ❤️ by cLLM Team

</div>
