# cLLM 架构设计文档

## 1. 项目概述

### 1.1 项目目标
将 xLLM Python 版本重构为 C++ 实现，以获得更高的性能和更好的资源利用率：
- 推理性能 20+ tokens/s
- 降低内存占用
- 提高并发处理能力
- 支持真流式输出

### 1.2 技术栈
| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 编程语言 | C++17 | 现代 C++ 特性 |
| HTTP 服务器 | 自研 | 基于 epoll/kqueue，支持真流式 |
| LLM 推理 | llama.cpp | 高性能 GGUF 模型推理 |
| 分词器 | tokenizers-cpp | Hugging Face tokenizers |
| JSON | nlohmann/json | 现代 C++ JSON 库 |
| 日志 | spdlog | 高性能日志库 |

## 2. 系统架构

### 2.1 整体架构图
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
│              Inference Engine (llama.cpp)               │
│        (GGUF Model, Metal/CUDA Acceleration)            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流
```
HTTP Request → Handler → Scheduler → BatchProcessor → ModelExecutor → InferenceEngine
      ↑                                                                      │
      └──────────────────── Streaming Response ──────────────────────────────┘
```

## 3. 模块设计

### 3.1 HTTP Server 模块

**设计原则**：
- 基于 epoll (Linux) / kqueue (macOS) 的事件驱动架构
- 支持真流式输出（chunked transfer encoding）
- 每生成一个 token 立即发送，TTFB < 0.1s

**API 端点**：

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 发现（显示所有端点） |
| GET | `/health` | 健康检查 |
| POST | `/generate` | 文本生成（非流式） |
| POST | `/generate_stream` | 文本生成（真流式） |
| POST | `/encode` | 文本编码 |
| POST | `/benchmark` | 性能测试 |
| GET | `/model/info` | 模型信息 |

**响应格式**：
```json
{
  "success": true,
  "data": { ... }
}
```

### 3.2 Scheduler 模块

**设计原则**：
- 动态批处理，提高 GPU 利用率
- 请求优先级管理
- 超时控制和资源保护

**核心流程**：
1. 请求入队（`addRequest`）
2. 批处理组装（`formBatch`）
3. 批处理执行（`processBatch`）
4. 结果分发（`waitForRequest`）

**流式生成**：
- 注册 token 回调（`setStreamingTokenCallback`）
- 每生成一个 token 立即触发回调
- 支持客户端断开检测

### 3.3 Model Executor 模块

**设计原则**：
- 统一的模型执行接口
- 支持多后端（通过工厂模式创建）
- KV Cache 管理

**推理流程**：
1. 输入 token IDs
2. 模型 forward（prefill / decode）
3. 获取 logits
4. 采样生成下一个 token

### 3.4 多后端架构

**支持的后端**：

| 后端 | 模型格式 | GPU 加速 | 特点 |
|------|---------|---------|------|
| **llama.cpp** | GGUF | Metal/CUDA | 高性能量化推理，推荐使用 |
| **Kylin** | SafeTensors | CPU/Metal | 自研推理引擎，支持 HuggingFace 模型 |
| **LibTorch** | TorchScript | CUDA | PyTorch C++ API |

**后端选择**：
```
┌─────────────────────────────────────────────────────────┐
│                  Model Executor                          │
│              (统一推理接口)                              │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
┌─────────▼────┐  ┌──────▼─────┐  ┌─────▼──────┐
│  llama.cpp   │  │   Kylin    │  │  LibTorch  │
│   Backend    │  │   Backend  │  │   Backend  │
│  (GGUF)      │  │ (SafeT.)   │  │ (TorchS.)  │
└──────────────┘  └────────────┘  └────────────┘
```

**llama.cpp 后端**（推荐）：
- 支持 GGUF 模型格式（量化模型）
- Metal (macOS) / CUDA (Linux) GPU 加速
- 自动 KV Cache 管理
- 高性能量化支持（Q4_K_M、Q5_K_M 等）

**Kylin 后端**（自研）：
- 支持 HuggingFace SafeTensors 格式
- 可选 GGML 或原生算子
- 支持 CPU 和 Metal 加速

### 3.5 Inference Engine 配置

**llama.cpp 主要参数**：
- `n_gpu_layers`: GPU 层数（0=CPU，99=全 GPU）
- `n_batch`: 批处理大小（推荐 512）
- `n_ctx`: 上下文长度
- `n_threads`: CPU 线程数

### 3.6 Tokenizer 模块

**设计原则**：
- 使用 Hugging Face tokenizers-cpp
- 支持 GGUF 内嵌 tokenizer
- UTF-8 编码处理

**核心功能**：
- `encode`: 文本 → token IDs
- `decode`: token IDs → 文本
- 特殊 token 处理（BOS、EOS、PAD）

### 3.7 Sampler 模块

**采样策略**：
- Temperature 采样
- Top-K 采样
- Top-P (Nucleus) 采样
- 组合采样

## 4. 并发架构

### 4.1 线程模型
```
┌─────────────────────────────────────────────────────────┐
│                   Main Thread                            │
│              (HTTP Server Event Loop)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼───────┐ ┌───▼───┐ ┌───────▼───────┐
│  Connection   │ │  ...  │ │  Connection   │
│   Handler     │ │       │ │   Handler     │
└───────────────┘ └───────┘ └───────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Scheduler Thread                            │
│         (Batch Formation & Execution)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│            Inference Engine Thread                       │
│              (GPU/CPU Inference)                        │
└─────────────────────────────────────────────────────────┘
```

### 4.2 同步机制
- HTTP 连接：非阻塞 I/O + 事件驱动
- 请求队列：线程安全队列 + 条件变量
- 流式回调：回调注册表 + 互斥锁

## 5. 性能优化

### 5.1 推理优化
- **GPU 加速**：Metal/CUDA 自动检测
- **批处理**：动态批处理提高吞吐量
- **KV Cache**：避免重复计算

### 5.2 内存优化
- **量化模型**：Q4_K_M 减少内存占用
- **内存映射**：mmap 加载模型
- **缓存复用**：KV Cache 跨请求复用

### 5.3 延迟优化
- **真流式**：每 token 立即发送
- **预热**：服务启动时预热模型
- **连接复用**：HTTP Keep-Alive

## 6. 配置文件

### 6.1 配置文件结构

配置文件使用 YAML 格式，主要配置文件：
- `config/config_llama_cpp.yaml` - llama.cpp 后端（推荐）
- `config/config_gpu.yaml` - GPU 加速配置
- `config/config_cpu.yaml` - CPU 模式配置

### 6.2 主要配置项

```yaml
# 服务器配置
server:
  host: "0.0.0.0"          # 监听地址
  port: 8085               # 监听端口
  num_threads: 8           # HTTP 工作线程数

# 模型配置
model:
  path: "/path/to/model.gguf"  # 模型文件路径
  vocab_size: 151936           # 词表大小
  max_context_length: 8192     # 最大上下文长度

# 后端配置（关键）
backend:
  type: "llama_cpp"        # 后端类型: llama_cpp | kylin | libtorch
  
  llama_cpp:
    n_batch: 512           # 批处理大小
    n_threads: 8           # CPU 线程数
    n_gpu_layers: 99       # GPU 层数（0=CPU，99=全GPU）
    n_seq_max: 2           # 最大并发序列数
    use_mmap: true         # 内存映射加载

  kylin:
    device_backend: "cpu"  # cpu | metal
    operator_backend: "ggml"
    n_threads: 8

# 调度器配置
scheduler:
  max_batch_size: 16       # 最大批处理大小
  request_timeout: 600.0   # 请求超时（秒）
  default_max_tokens: 2048 # 默认生成长度

# 资源配置
resources:
  max_context_length: 8192
  kv_cache_max_size: 16
  memory_limit_mb: 16384

# 日志配置
logging:
  level: "info"            # debug | info | warn | error
```

### 6.3 后端配置说明

**llama.cpp 后端**（推荐）：
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `n_gpu_layers` | GPU 层数，0=CPU，99=全GPU | 99 (有 GPU) |
| `n_batch` | prompt 处理批大小 | 512 |
| `n_threads` | CPU 线程数 | CPU 核心数 |
| `n_seq_max` | 最大并发序列 | 2-4 |

**Kylin 后端**：
| 参数 | 说明 | 选项 |
|------|------|------|
| `device_backend` | 计算设备 | cpu / metal |
| `operator_backend` | 算子实现 | ggml / native |

## 7. 部署与运行

### 7.1 构建
```bash
cd cLLM
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 7.2 运行
```bash
./build/bin/cllm_server --config config/config_llama_cpp.yaml
```

### 7.3 测试
```bash
# 查看所有端点
curl http://localhost:8085/

# 健康检查
curl http://localhost:8085/health

# 模型信息
curl http://localhost:8085/model/info

# 文本生成（流式）
curl -X POST http://localhost:8085/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","max_tokens":50,"stream":true}'
```

## 8. 详细实现

各模块的详细代码实现请参考：
- **HTTP Server**: `src/http/http_server.cpp`
- **Scheduler**: `src/scheduler/scheduler.cpp`
- **Model Executor**: `src/model/executor.cpp`
- **Tokenizer**: `src/tokenizer/`
- **Sampler**: `src/sampler/sampler.cpp`

或查看模块设计文档：`docs/modules/`
