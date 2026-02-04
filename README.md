# cLLM - C++ Large Language Model Inference Engine

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![GCC](https://img.shields.io/badge/GCC-10+-green.svg)](https://gcc.gnu.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

**高性能 C++ 大语言模型推理引擎**

[快速开始](#-快速开始) • [文档](#-文档) • [架构](#-架构) • [部署](#-部署指南) • [贡献](#-贡献)

</div>

---

## 📖 项目简介

cLLM 是一个基于 C++17 开发的高性能大语言模型推理引擎，专注于生产环境部署：

- 🚀 **高性能**: 推理速度 20+ tokens/s，比 Python 版本提升 3-5 倍
- 💾 **低内存**: 优化的内存管理和 KV 缓存策略
- 🔄 **高并发**: 基于原生异步 HTTP 服务器，支持大规模并发请求
- 🎯 **生产就绪**: 完整的 HTTP API、动态批处理、流式输出
- 🖥️ **多后端**: 支持 llama.cpp (GGUF)、Kylin (SafeTensors)、LibTorch

---

## ✨ 核心特性

### 推理能力
- ✅ **llama.cpp 后端**（推荐）- GGUF 模型，Metal/CUDA 加速
- ✅ **Kylin 后端** - 自研引擎，支持 HuggingFace SafeTensors
- ✅ **LibTorch 后端** - PyTorch C++ API，TorchScript 模型
- ✅ 多种采样策略（Temperature, Top-K, Top-P）
- ✅ KV Cache 优化
- ✅ 动态批处理（Dynamic Batching）
- ✅ 真流式输出（TTFB < 0.1s）

### 服务能力
- ✅ RESTful HTTP API
- ✅ 流式生成（Server-Sent Events）
- ✅ 健康检查端点
- ✅ 请求队列管理
- ✅ 异步处理框架
- ✅ 模型热加载

### 部署支持
- ✅ CentOS 7/8 一键部署脚本
- ✅ Ubuntu/Debian 支持
- ✅ macOS 开发环境
- ✅ Docker 容器化
- ✅ systemd 服务管理

---

## 🚀 快速开始

### 前置条件

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|----------|----------|------|
| C++ 标准 | C++17 | C++17 | 语言标准，必需支持 |
| GCC | 10.0 | 10+ | CentOS 7 已验证可用 |
| Clang | 12.0 | 14+ | macOS 推荐 |
| CMake | 3.20 | 3.28+ | 必需 |
| Python | 3.8 | 3.10+ | 用于构建工具 |

**注意**：
- **C++17** 是语言标准要求，需要编译器支持（GCC 7+ 已支持，但本项目采用GCC 10+已经编译成功 ）
- **CentOS 7** 用户请使用 `devtoolset-10` 或更高版本（部署脚本会自动安装）
- **macOS** 用户请使用 Xcode 12+ 或 Homebrew 安装的 Clang
- CMake 3.20+ 是必需的，用于支持现代 CMake 特性

**依赖库**：nlohmann-json, yaml-cpp, spdlog, sentencepiece

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/xdongp/cLLM.git
cd cLLM

# 2. 安装依赖 (macOS)
brew install cmake nlohmann-json yaml-cpp spdlog

# 或 (Ubuntu)
sudo apt-get install cmake nlohmann-json3-dev libyaml-cpp-dev libspdlog-dev

# 3. 编译项目
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. 运行测试
ctest --output-on-failure
```

### 快速运行

```bash
# 启动 HTTP 服务器
./bin/cllm_server --config ../config/config_llama_cpp_cpu.yaml

# 测试健康检查
curl http://localhost:8080/health

# 测试文本生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_tokens": 50}'
```

**详细步骤**: 查看 [快速开始指南](docs/guides/快速开始.md)

---

## �️ 部署指南

### CentOS 7 生产部署（推荐）

我们提供完整的一键部署脚本，支持 x86_64 和 ARM64 架构：

```bash
# 1. 下载部署脚本
wget https://raw.githubusercontent.com/xdongp/cLLM/main/scripts/deploy_centos7.sh
chmod +x deploy_centos7.sh

# 2. CPU 模式部署
sudo ./deploy_centos7.sh

# 3. GPU 模式部署（需要 CUDA）
sudo ./deploy_centos7.sh --gpu

# 4. 本地源码部署
sudo ./deploy_centos7.sh --local
```

**部署脚本功能**：
- 自动安装 GCC 10/11、CMake 3.28+、Python 3.12
- 安装所有依赖库（OpenBLAS、nlohmann-json、yaml-cpp、spdlog）
- 编译 llama.cpp 和 cLLM
- 创建 systemd 服务
- 配置日志轮转

### 部署选项

| 选项 | 说明 |
|------|------|
| `--gpu` | 启用 GPU 模式（需要 CUDA） |
| `--local` | 使用当前目录的源码 |
| `--skip-deps` | 跳过依赖安装 |

### 部署后配置

```bash
# 编辑配置文件
sudo vim /opt/cllm/config/config_llama_cpp_cpu.yaml

# 修改模型路径
model:
  path: "/opt/models/your-model.gguf"

# 启动服务
sudo systemctl start cllm
sudo systemctl enable cllm

# 查看状态
sudo systemctl status cllm
curl http://localhost:8080/health
```

### 其他部署方式

- **CentOS 7 + GCC 11**: [scripts/deploy_centos7_gcc11.sh](scripts/deploy_centos7_gcc11.sh)
- **Docker 部署**: 查看 [Docker 部署指南](docs/deployment/Docker部署指南.md)
- **Linux 生产环境**: 查看 [Linux生产环境部署指南](docs/deployment/Linux生产环境部署指南.md)

---

## �📚 文档

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

### 部署文档
- [Linux生产环境部署指南](docs/deployment/Linux生产环境部署指南.md)
- [Docker部署指南](docs/deployment/Docker部署指南.md)
- [性能优化指南](docs/deployment/性能优化指南.md)

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
│     (RESTful API, Request Handling, Streaming)          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Request Scheduler                       │
│       (Dynamic Batching, Request Management)            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Model Executor                           │
│         (Inference, KV Cache, Sampling)                 │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Inference Engine                            │
│   ┌──────────┬──────────┬──────────┐                    │
│   │llama.cpp │  Kylin   │LibTorch  │                    │
│   │ (GGUF)   │(SafeT.)  │(TorchS.) │                    │
│   └──────────┴──────────┴──────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
HTTP Request → Handler → Scheduler → BatchProcessor → ModelExecutor → InferenceEngine
      ↑                                                                      │
      └──────────────────── Streaming Response ←─────────────────────────────┘
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

### 多后端架构

| 后端 | 模型格式 | GPU 加速 | 适用场景 |
|------|---------|---------|---------|
| **llama.cpp** | GGUF | Metal/CUDA | 生产环境，量化模型 |
| **Kylin** | SafeTensors | CPU/Metal | HuggingFace 模型 |
| **LibTorch** | TorchScript | CUDA | PyTorch 模型 |

**详细架构**: 查看 [cLLM详细设计](docs/architecture/cLLM详细设计.md)

---

## 🔌 API 参考

### HTTP 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 发现 |
| GET | `/health` | 健康检查 |
| POST | `/generate` | 文本生成（非流式） |
| POST | `/generate_stream` | 文本生成（流式） |
| POST | `/encode` | 文本编码 |
| POST | `/benchmark` | 性能测试 |
| GET | `/model/info` | 模型信息 |

### 文本生成示例

```bash
# 基本生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "hello",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# 中文生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# 流式生成
curl -X POST http://localhost:8080/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "讲一个故事",
    "max_tokens": 200,
    "stream": true
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
│   ├── config_llama_cpp_cpu.yaml
│   ├── config_llama_cpp_gpu.yaml
│   ├── config_kylin_cpu.yaml
│   └── config_kylin_gpu.yaml
├── scripts/                # 工具脚本
│   ├── deploy_centos7.sh
│   └── deploy_centos7_gcc11.sh
└── third_party/            # 第三方库
    └── llama.cpp/         # llama.cpp 源码
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

### 基准测试

```bash
# 使用内置 benchmark 端点
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "requests": 100,
    "concurrency": 10,
    "max_tokens": 50
  }'
```

---

## 📊 性能

### 基准测试结果

| 指标 | Python 版本 | C++ 版本 | 提升 |
|------|------------|----------|------|
| **推理速度** | 5-8 tokens/s | 20+ tokens/s | **3-5x** |
| **内存占用** | ~2GB | ~800MB | **2.5x** |
| **并发能力** | 10 req/s | 100+ req/s | **10x** |
| **启动时间** | 3-5s | <1s | **5x** |

### 优化技术

- ✅ KV Cache 复用
- ✅ 动态批处理（Batch Size: 1-32）
- ✅ 异步 I/O（基于 epoll/kqueue）
- ✅ 零拷贝内存管理
- ✅ GGUF 量化支持（Q4_K_M、Q5_K_M）

---

## 📋 技术栈

### 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| **llama.cpp** | latest | GGUF 模型推理 |
| **nlohmann-json** | 3.11+ | JSON 处理 |
| **yaml-cpp** | 0.8+ | YAML 配置 |
| **spdlog** | 1.12+ | 日志系统 |
| **OpenBLAS** | 0.3+ | CPU 加速（可选） |

### 开发工具

- **构建**: CMake 3.15+
- **测试**: Google Test
- **CI/CD**: GitHub Actions
- **文档**: Markdown
- **代码质量**: clang-format, clang-tidy

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

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 👥 贡献者

感谢所有贡献者！

---

## 🔗 相关链接

- **文档**: [docs/](docs/)
- **问题反馈**: [Issues](https://github.com/xdongp/cLLM/issues)
- **部署脚本**: [scripts/](scripts/)
